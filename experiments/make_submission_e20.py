"""E20 submission: fit winning ensemble on all training months, score ranking DEP.

Winning blend (NNLS weights fit on Jan+Jul holdout, validated on Dec):
  A:0.0  B:0.0  C(CatBoost residual):0.456  D(XGB residual):0.053  E(airport LGB):0.491

E18-H hygiene everywhere: matched-only geo_mean, LIRF-override rows dropped
from every expert fit, always-on LIRF MVT-SCHED override on the output.
Fits use data/training_*.parquet only. Ranking is the prediction-time
information set (no DEP BLOCK/TAXITIME in any fit).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, load_dep  # noqa: E402
from run_e16a import load_arr_delay  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_e4_e5 import load_arr  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_submitting_check import (  # noqa: E402
    SUBMIT_PATH,
    featurize,
    load_ranking_score,
    verify_submitting,
)
from run_e20_ensemble import (  # noqa: E402
    CAT_COLS,
    NUM_FEATS,
    apply_override,
    calibrate,
    cat_pdf,
    lgb_pdf,
    xgb_pdf,
    residual_target,
    time_es_split,
)

WEIGHTS = {"A": 0.0, "B": 0.0, "C": 0.456, "D": 0.053, "E": 0.491}

OUT_PATH = ROOT / "likable-eagle_v4.parquet"
OUT_COPY = ROOT / "analysis" / "submitting_check" / "likable-eagle_v4.parquet"
PRED_PATH = ROOT / "analysis" / "submitting_check" / "tables" / "submitting_predictions_e20.parquet"


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pack_submission(pred: pl.DataFrame) -> pl.DataFrame:
    sub = pl.read_parquet(SUBMIT_PATH)
    filled = (
        sub.with_row_index("_i")
        .drop("TAXITIME_SEC_mvt")
        .join(pred.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt", how="left")
        .sort("_i").drop("_i").select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    )
    errors = []
    if filled.height != 344841:
        errors.append(f"row count {filled.height}")
    if filled["MVT_ID_mvt"].n_unique() != filled.height:
        errors.append("MVT_ID not unique")
    if not (filled["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        errors.append("ID order/values mismatch")
    if filled["TAXITIME_SEC_mvt"].null_count() or int(filled["TAXITIME_SEC_mvt"].is_nan().sum()):
        errors.append("null/nan predictions")
    if errors:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(errors))
    return filled


def main():
    log("E20 submission: verify submitting.parquet...")
    verify_submitting()

    log("load ranking prediction-time frames...")
    score_dep, score_arr_t, score_arr_d, rank_meta = load_ranking_score()
    sub_ids = pl.read_parquet(SUBMIT_PATH, columns=["MVT_ID_mvt"])
    n_join = sub_ids.join(score_dep.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="inner").height
    if n_join != sub_ids.height:
        raise SystemExit("submitting IDs do not fully match ranking DEP")

    log("featurize training_*.parquet...")
    train_raw = load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    train_feat = featurize(train_raw, load_arr(), load_arr_delay())
    log(f"  training DEP {train_feat.height:,}")

    log("featurize ranking DEP (causal within ranking stream)...")
    score_feat = featurize(score_dep, score_arr_t, score_arr_d)
    log(f"  ranking DEP {score_feat.height:,}")

    log("matched-only geometry + hour baseline (fit on training)...")
    tabs = geometry_tables(train_feat.filter(~pl.col("unmatched")))
    tr = attach_geometry(train_feat, tabs, 30)
    sc = attach_geometry(score_feat, tabs, 30)
    from run_e16a import attach_hour_baseline
    tr = attach_hour_baseline(tr, tr)
    sc = attach_hour_baseline(tr, sc)

    p_tr, p_sc = calibrate(tr, sc)
    tr = tr.with_columns(pl.Series("p_cal", p_tr))
    sc = sc.with_columns(pl.Series("p_cal", p_sc))

    um_tr = tr["unmatched"].to_numpy().astype(bool)
    ap_tr = tr["airport"].to_numpy()
    ov = um_tr & (ap_tr == "LIRF")
    log(f"  hygiene: dropping {int(ov.sum()):,} LIRF-override rows from expert training")
    tr_res = tr.filter(~pl.Series(ov))
    tr_fit, tr_es = time_es_split(tr_res)
    yres_tr = residual_target(tr_fit)
    yres_es = residual_target(tr_es)

    pdf_tr = lgb_pdf(tr_fit, NUM_FEATS)
    pdf_es = lgb_pdf(tr_es, NUM_FEATS)
    cat_tr = cat_pdf(tr_fit, NUM_FEATS)
    cat_es = cat_pdf(tr_es, NUM_FEATS)
    cat_sc = cat_pdf(sc, NUM_FEATS)
    xgb_tr = xgb_pdf(tr_fit, NUM_FEATS)
    xgb_es = xgb_pdf(tr_es, NUM_FEATS)
    xgb_sc = xgb_pdf(sc, NUM_FEATS)

    # C: CatBoost residual
    log("fit expert C (CatBoost residual, all training months)...")
    from catboost import CatBoostRegressor, Pool
    mC = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                           loss_function="RMSE", random_seed=1, verbose=False, thread_count=-1)
    mC.fit(Pool(cat_tr[NUM_FEATS + CAT_COLS], yres_tr, cat_features=CAT_COLS),
           eval_set=Pool(cat_es[NUM_FEATS + CAT_COLS], yres_es, cat_features=CAT_COLS),
           early_stopping_rounds=60)
    predC = p_sc + np.asarray(mC.predict(cat_sc[NUM_FEATS + CAT_COLS]), dtype=np.float64)

    # D: XGBoost residual
    log("fit expert D (XGBoost residual, all training months)...")
    import xgboost as xgb
    mD = xgb.train(
        {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 7, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 80, "lambda": 1.0, "seed": 1, "tree_method": "hist"},
        xgb.DMatrix(xgb_tr[NUM_FEATS], yres_tr), num_boost_round=2000,
        evals=[(xgb.DMatrix(xgb_es[NUM_FEATS], yres_es), "es")],
        early_stopping_rounds=60, verbose_eval=False)
    predD = p_sc + np.asarray(mD.predict(xgb.DMatrix(xgb_sc[NUM_FEATS])), dtype=np.float64)

    # E: airport experts
    log("fit expert E (airport-specific residual LGB, all training months)...")
    from run_e20_ensemble import fit_lgb
    AIRPORTS = sorted(set(tr["airport"].to_list()))
    predE = np.full(len(p_sc), np.nan)
    sc_ap = sc["airport"].to_numpy()
    for a in AIRPORTS:
        tr_a = tr_res.filter(pl.col("airport") == a)
        sc_a = sc.filter(pl.col("airport") == a)
        if tr_a.height < 20000 or sc_a.height == 0:
            continue
        ta_fit, ta_es = time_es_split(tr_a)
        pda = lgb_pdf(ta_fit, NUM_FEATS)
        pde = lgb_pdf(ta_es, NUM_FEATS)
        pds = lgb_pdf(sc_a, NUM_FEATS)
        ma = fit_lgb(pda[NUM_FEATS + CAT_COLS], residual_target(ta_fit),
                     pde[NUM_FEATS + CAT_COLS], residual_target(ta_es), seed=1)
        predE[sc_ap == a] = sc_a["p_cal"].to_numpy() + np.asarray(ma.predict(pds[NUM_FEATS + CAT_COLS]), dtype=np.float64)
    miss = ~np.isfinite(predE)
    if miss.any():
        log(f"  E fallback (no airport model) rows: {int(miss.sum())}")
        predE[miss] = predC[miss]

    blended = WEIGHTS["C"] * predC + WEIGHTS["D"] * predD + WEIGHTS["E"] * predE
    um = sc["unmatched"].to_numpy().astype(bool)
    ap = sc["airport"].to_numpy()
    sched = sc["mvt_sched"].to_numpy().astype(float)
    p = apply_override(blended, um, ap, sched)
    if int(np.isfinite(p).sum()) != len(p):
        raise SystemExit("non-finite blended predictions")
    log(f"  ranking preds n={len(p):,} mean={p.mean():.1f} med={np.median(p):.1f} "
        f"LIRF_fallback={int((um & (ap == 'LIRF')).sum()):,} neg={int((p < 0).sum()):,} gt60m={int((p >= 3600).sum()):,}")

    pred_pl = sc.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", p))
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_pl.write_parquet(PRED_PATH)
    log(f"  wrote {PRED_PATH}")

    filled = pack_submission(pred_pl)
    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(OUT_PATH)
    filled.write_parquet(OUT_COPY)

    reread = pl.read_parquet(OUT_PATH)
    rp = reread["TAXITIME_SEC_mvt"].to_numpy()
    print("WROTE", OUT_PATH)
    print("COPY", OUT_COPY)
    print("model", "E20-nnls (0.456 C + 0.053 D + 0.491 E)")
    print("rows", reread.height)
    print("columns", reread.columns)
    print("unique_ids", reread["MVT_ID_mvt"].n_unique())
    print("null_count", reread["TAXITIME_SEC_mvt"].null_count())
    print("pred_min", float(np.min(rp)))
    print("pred_max", float(np.max(rp)))
    print("pred_mean", float(np.mean(rp)))
    print("pred_median", float(np.median(rp)))
    print("n_negative", int((rp < 0).sum()))
    print("n_gt_60m", int((rp >= 3600).sum()))
    print("n_lirf_fallback", int((um & (ap == "LIRF")).sum()))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
