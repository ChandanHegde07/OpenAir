"""E24 submission (v6): quantile blend L2 + Q90 residual LightGBM.

NNLS weights from Jan+Jul holdout (E24): L2=0.732, Q90=0.268 (Q10=Q50=0).
Fits on all training_*.parquet months. Ranking-safe clocks only.
LIRF unmatched → MVT−SCHED. No METAR.
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
from run_e16a import attach_hour_baseline, load_arr_delay  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_e4_e5 import load_arr  # noqa: E402
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
    lgb_pdf,
    residual_target,
    time_es_split,
)
from run_e24_quantile import fit_lgb_obj  # noqa: E402
from make_submission_e20 import pack_submission  # noqa: E402

W_L2 = 0.7320472451509163
W_Q90 = 0.2679527548490836
COLS = list(NUM_FEATS) + list(CAT_COLS)

OUT_PATH = ROOT / "likable-eagle_v6.parquet"
OUT_COPY = ROOT / "analysis" / "submitting_check" / "likable-eagle_v6.parquet"
PRED_PATH = ROOT / "analysis" / "submitting_check" / "tables" / "submitting_predictions_e24.parquet"


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("E24 v6: L2 + Q90 quantile blend. verify submitting.parquet...")
    verify_submitting()

    log("load ranking prediction-time frames...")
    score_dep, score_arr_t, score_arr_d, _ = load_ranking_score()
    sub_ids = pl.read_parquet(SUBMIT_PATH, columns=["MVT_ID_mvt"])
    n_join = sub_ids.join(score_dep.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="inner").height
    if n_join != sub_ids.height:
        raise SystemExit("submitting IDs do not fully match ranking DEP")

    log("featurize training_*.parquet...")
    train_raw = load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    train_feat = featurize(train_raw, load_arr(), load_arr_delay())
    log(f"  training DEP {train_feat.height:,}")

    log("featurize ranking DEP...")
    score_feat = featurize(score_dep, score_arr_t, score_arr_d)
    log(f"  ranking DEP {score_feat.height:,}")

    log("matched-only geometry + hour baseline (fit on training)...")
    tabs = geometry_tables(train_feat.filter(~pl.col("unmatched")))
    tr = attach_geometry(train_feat, tabs, 30)
    sc = attach_geometry(score_feat, tabs, 30)
    tr = attach_hour_baseline(tr, tr)
    sc = attach_hour_baseline(tr, sc)

    p_tr, p_sc = calibrate(tr, sc)
    tr = tr.with_columns(pl.Series("p_cal", p_tr))
    sc = sc.with_columns(pl.Series("p_cal", p_sc))

    um_tr = tr["unmatched"].to_numpy().astype(bool)
    ap_tr = tr["airport"].to_numpy()
    ov = um_tr & (ap_tr == "LIRF")
    log(f"  hygiene: dropping {int(ov.sum()):,} LIRF-override rows from residual train")
    tr_res = tr.filter(~pl.Series(ov))
    tr_fit, tr_es = time_es_split(tr_res)
    yres_tr = residual_target(tr_fit)
    yres_es = residual_target(tr_es)

    pdf_tr = lgb_pdf(tr_fit, NUM_FEATS)
    pdf_es = lgb_pdf(tr_es, NUM_FEATS)
    pdf_sc = lgb_pdf(sc, NUM_FEATS)
    xtr, xes, xsc = pdf_tr[COLS], pdf_es[COLS], pdf_sc[COLS]

    log("fit L2 residual LightGBM...")
    m_l2 = fit_lgb_obj(xtr, yres_tr, xes, yres_es, objective="regression", seed=1)
    pred_l2 = p_sc + np.asarray(m_l2.predict(xsc), dtype=np.float64)
    log(f"  L2 trees={m_l2.best_iteration_}")

    log("fit Q90 residual LightGBM (pinball α=0.9)...")
    m_q = fit_lgb_obj(xtr, yres_tr, xes, yres_es, objective="quantile", alpha=0.9, seed=1)
    pred_q = p_sc + np.asarray(m_q.predict(xsc), dtype=np.float64)
    log(f"  Q90 trees={m_q.best_iteration_}")

    blended = W_L2 * pred_l2 + W_Q90 * pred_q
    um = sc["unmatched"].to_numpy().astype(bool)
    ap = sc["airport"].to_numpy()
    sched = sc["mvt_sched"].to_numpy().astype(float)
    p = apply_override(blended, um, ap, sched)
    if int(np.isfinite(p).sum()) != len(p):
        raise SystemExit("non-finite blended predictions")
    log(
        f"  ranking preds n={len(p):,} mean={p.mean():.1f} med={np.median(p):.1f} "
        f"LIRF_fallback={int((um & (ap == 'LIRF')).sum()):,} neg={int((p < 0).sum()):,} "
        f"gt60m={int((p >= 3600).sum()):,}"
    )

    pred_pl = sc.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", p))
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_pl.write_parquet(PRED_PATH)

    filled = pack_submission(pred_pl)
    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(OUT_PATH)
    filled.write_parquet(OUT_COPY)

    reread = pl.read_parquet(OUT_PATH)
    rp = reread["TAXITIME_SEC_mvt"].to_numpy()
    print("WROTE", OUT_PATH)
    print("COPY", OUT_COPY)
    print("model", f"E24 Qblend L2={W_L2:.3f} + Q90={W_Q90:.3f}")
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
