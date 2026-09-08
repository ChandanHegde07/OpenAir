"""Reusable E20 fit/predict. Does not modify the original E20 experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
sys.path.insert(0, str(EXP))

from common import AIRPORTS, airport_mean_fallback, fill_with_fallback  # noqa: E402
from run_e12_e9 import CAT_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from run_e20_ensemble import (  # noqa: E402
    NUM_FEATS,
    apply_override,
    cat_pdf,
    lgb_pdf,
    nnls_weights,
    residual_target,
    xgb_pdf,
)

E20_OOF_DIR = EXP / "results" / "E20"
E20_NNLS_WEIGHTS = np.array([0.0, 0.0, 0.456, 0.053, 0.491], dtype=np.float64)
BLEND_ORDER = ["A", "B", "C", "D", "E"]


def blend_from_experts(df: pl.DataFrame, weights: np.ndarray | None = None) -> np.ndarray:
    P = np.column_stack([df[f"pred_{k}"].to_numpy().astype(np.float64) for k in BLEND_ORDER])
    w = E20_NNLS_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    p = P @ w
    um = df["unmatched"].to_numpy().astype(bool)
    ap = df["airport"].to_numpy()
    sched = df["mvt_sched"].to_numpy().astype(np.float64)
    return apply_override(p, um, ap, sched)


def exact_nnls_weights(oof: pl.DataFrame) -> np.ndarray:
    y = oof["y"].to_numpy().astype(np.float64)
    P = np.column_stack([oof[f"pred_{k}"].to_numpy().astype(np.float64) for k in BLEND_ORDER])
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    return nnls_weights(P[ok], y[ok])


def load_cached_e20_oof(split: str) -> pl.DataFrame:
    path = E20_OOF_DIR / f"oof_predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path)
    w = exact_nnls_weights(pl.read_parquet(E20_OOF_DIR / "oof_predictions_janjul.parquet"))
    pred = blend_from_experts(df, w)
    return df.with_columns(pl.Series("e20_pred", pred))


def _calibrate(tr: pl.DataFrame, va: pl.DataFrame):
    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal_va = fill_with_fallback(
        fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, va),
    )
    p_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal_tr = fill_with_fallback(
        fill_with_fallback(p_t, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    return p_cal_tr, p_cal_va


def fit_predict_e20(
    feat: pl.DataFrame,
    train_months: list[int],
    pred_months: list[int],
    seed: int = 1,
    log=print,
    lite: bool = False,
) -> pl.DataFrame:
    """Fit E20 on train_months, predict pred_months. feat must already be fully featurized."""
    months = list(dict.fromkeys(list(train_months) + list(pred_months)))
    sub = feat.filter(pl.col("month").is_in(months))
    tr0 = sub.filter(pl.col("month").is_in(train_months))
    va0 = sub.filter(pl.col("month").is_in(pred_months))
    if tr0.height < 10000 or va0.height == 0:
        raise RuntimeError(f"bad fold train={tr0.height} pred={va0.height}")
    pack = prepare_split(sub, list(pred_months), matched_geo=True)
    tr, va = pack["tr"], pack["va"]
    if tr.height != tr0.height:
        tr = pack["tr"].filter(pl.col("month").is_in(train_months))
    p_tr, p_va = _calibrate(tr, va)
    tr = tr.with_columns(pl.Series("p_cal", p_tr))
    va = va.with_columns(pl.Series("p_cal", p_va))

    um_tr = tr["unmatched"].to_numpy().astype(bool)
    ap_tr = tr["airport"].to_numpy()
    ov = um_tr & (ap_tr == "LIRF")
    tr_res = tr.filter(~pl.Series(ov))
    tr_fit, tr_es = time_es_split(tr_res)
    log(f"    E20 fold train_months={train_months} pred={pred_months} "
        f"fit={tr_fit.height:,} es={tr_es.height:,} va={va.height:,}")

    yres_tr = residual_target(tr_fit)
    yres_es = residual_target(tr_es)
    um_va = va["unmatched"].to_numpy().astype(bool)
    ap_va = va["airport"].to_numpy()
    sched_va = va["mvt_sched"].to_numpy().astype(np.float64)
    p_va_np = va["p_cal"].to_numpy().astype(np.float64)

    from catboost import CatBoostRegressor, Pool
    import xgboost as xgb

    cat_tr = cat_pdf(tr_fit, NUM_FEATS)
    cat_es = cat_pdf(tr_es, NUM_FEATS)
    cat_va = cat_pdf(va, NUM_FEATS)
    mC = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )
    mC.fit(
        Pool(cat_tr[NUM_FEATS + CAT_COLS], yres_tr, cat_features=CAT_COLS),
        eval_set=Pool(cat_es[NUM_FEATS + CAT_COLS], yres_es, cat_features=CAT_COLS),
        early_stopping_rounds=60,
    )
    predC = apply_override(
        p_va_np + np.asarray(mC.predict(cat_va[NUM_FEATS + CAT_COLS]), dtype=np.float64),
        um_va, ap_va, sched_va,
    )

    if lite:
        predD = predC.copy()
    else:
        xgb_tr = xgb_pdf(tr_fit, NUM_FEATS)
        xgb_es = xgb_pdf(tr_es, NUM_FEATS)
        xgb_va = xgb_pdf(va, NUM_FEATS)
        mD = xgb.train(
            {
                "objective": "reg:squarederror",
                "eta": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 80,
                "lambda": 1.0,
                "seed": seed,
                "tree_method": "hist",
            },
            xgb.DMatrix(xgb_tr[NUM_FEATS], yres_tr),
            num_boost_round=2000,
            evals=[(xgb.DMatrix(xgb_es[NUM_FEATS], yres_es), "es")],
            early_stopping_rounds=60,
            verbose_eval=False,
        )
        predD = apply_override(
            p_va_np + np.asarray(mD.predict(xgb.DMatrix(xgb_va[NUM_FEATS])), dtype=np.float64),
            um_va, ap_va, sched_va,
        )

    predE = np.full(len(p_va_np), np.nan)
    for a in AIRPORTS:
        tr_a = tr_res.filter(pl.col("airport") == a)
        va_a = va.filter(pl.col("airport") == a)
        va_idx = np.where(ap_va == a)[0]
        if tr_a.height < 20000 or va_a.height == 0:
            predE[va_idx] = predC[va_idx]
            continue
        ta_fit, ta_es = time_es_split(tr_a)
        pda = lgb_pdf(ta_fit, NUM_FEATS)
        pde = lgb_pdf(ta_es, NUM_FEATS)
        pds = lgb_pdf(va_a, NUM_FEATS)
        ma = fit_lgb(
            pda[NUM_FEATS + CAT_COLS],
            residual_target(ta_fit),
            pde[NUM_FEATS + CAT_COLS],
            residual_target(ta_es),
            seed=seed,
        )
        resid = np.asarray(ma.predict(pds[NUM_FEATS + CAT_COLS]), dtype=np.float64)
        predE[va_idx] = apply_override(
            va_a["p_cal"].to_numpy() + resid, um_va[va_idx], ap_va[va_idx], sched_va[va_idx]
        )
    miss = ~np.isfinite(predE)
    if miss.any():
        predE[miss] = predC[miss]

    # dummy A/B for blend_from_experts (zero weight)
    dummy = predC.copy()
    out = va.select(["MVT_ID_mvt", "y", "unmatched", "airport", "mvt_sched", "p_cal", "month"])
    out = out.with_columns(
        pl.Series("pred_A", dummy),
        pl.Series("pred_B", dummy),
        pl.Series("pred_C", predC),
        pl.Series("pred_D", predD),
        pl.Series("pred_E", predE),
    )
    e20 = blend_from_experts(out, E20_NNLS_WEIGHTS)
    return out.with_columns(pl.Series("e20_pred", e20))
