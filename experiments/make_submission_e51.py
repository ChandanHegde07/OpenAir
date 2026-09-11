"""E51 v14 — v13 recipe with 0.7 L2 + 0.3 quantile(0.65) Δ.

OOF rec_mix_gh25: 238.58 / 211.70 vs v13 238.52 / 212.08.
Jan+Jul tied; December matched −0.38. Unmatched = v8. No LIRF G.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, load_dep  # noqa: E402
from run_submitting_check import SUBMIT_PATH  # noqa: E402
from make_submission_e50 import NUM0, NUM, CAT, P90, HTHR, add_extra, e20_col, pdf, log  # noqa: E402

OUT = ROOT / "experiments" / "results" / "E51" / "likable-eagle_v14.parquet"
OUT_ROOT = ROOT / "likable-eagle_v14.parquet"
OUT_SUB = ROOT / "submissions" / "likable-eagle_v14.parquet"
V8_PATH = ROOT / "submissions" / "likable-eagle_v8.parquet"
SEED = 1
LAM_REC, LAM_HAT = 0.5, 0.25
MIX_Q = 0.30


def fit_lgb(X, y, alpha=None, seed=1):
    kw = dict(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1,
    )
    if alpha is None:
        m = lgb.LGBMRegressor(**kw)
    else:
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **kw)
    m.fit(X, y, categorical_feature=CAT)
    return m


def main():
    if not V8_PATH.exists():
        raise SystemExit(f"missing {V8_PATH}")
    log("v14 train...")
    dep = add_extra(load_dep()).filter(~pl.col("unmatched")).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
    oof = pl.concat(
        [
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_janjul.parquet")),
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_dec.parquet")),
        ]
    )
    dep = dep.join(oof, on="MVT_ID_mvt", how="left")
    have = dep.filter(pl.col("e20_pred").is_not_null())
    miss = dep.filter(pl.col("e20_pred").is_null())
    fill = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    fill.fit(pdf(have, NUM0), have["e20_pred"].to_numpy().astype(float), categorical_feature=CAT)
    miss = miss.with_columns(pl.Series("e20_pred", np.asarray(fill.predict(pdf(miss, NUM0)), float)))
    tr = pl.concat([have, miss], how="vertical_relaxed")
    y = tr["y"].to_numpy().astype(float)
    P = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    e_tr = tr["e20_pred"].to_numpy().astype(float)
    X = pdf(tr, NUM)
    log("  Δ L2 + quantile + residual...")
    d_l2 = fit_lgb(X, y - P, seed=1)
    d_q = fit_lgb(X, y - P, alpha=0.65, seed=2)
    hmod = fit_lgb(X, y - e_tr, seed=7)

    log("ranking...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").collect()
    rank = add_extra(rank)
    v8 = pl.read_parquet(V8_PATH)
    matched = rank.filter(pl.col("AOBT_3_flt").is_not_null())
    matched = matched.join(v8.rename({"TAXITIME_SEC_mvt": "e20_pred"}), on="MVT_ID_mvt", how="left")
    Xm = pdf(matched, NUM)
    Pm = np.clip(matched["mvt_aobt"].to_numpy().astype(float), 0, None)
    rec = np.clip(Pm + (1 - MIX_Q) * np.asarray(d_l2.predict(Xm), float) + MIX_Q * np.asarray(d_q.predict(Xm), float), 0, None)
    hat = np.asarray(hmod.predict(Xm), float)
    e8 = matched["e20_pred"].to_numpy().astype(float)
    gate = (e8 > P90) | ((rec - e8) > HTHR)
    grec = e8 + LAM_REC * (rec - e8) * gate.astype(float)
    new = np.maximum(grec + LAM_HAT * hat, 0.0)

    j = v8.rename({"TAXITIME_SEC_mvt": "v8"}).join(
        matched.select("MVT_ID_mvt").with_columns(pl.Series("new", new)),
        on="MVT_ID_mvt",
        how="left",
    )
    final = np.where(j["new"].is_not_null().to_numpy(), j["new"].to_numpy(), j["v8"].to_numpy())
    out = j.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", final.astype(float)))
    sub = pl.read_parquet(SUBMIT_PATH)
    if out.height != 344841 or not (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        raise SystemExit("id check failed")
    nchg = int((np.abs(final - j["v8"].to_numpy()) > 1e-6).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    out.write_parquet(OUT_ROOT)
    out.write_parquet(OUT_SUB)
    print("WROTE", OUT)
    print("COPY", OUT_ROOT, OUT_SUB)
    print("v14: 0.7 L2 + 0.3 q65 Δ, grec 0.5 + hat 0.25; unmatched=v8")
    print("rows", out.height, "changed", nchg)
    print("mean", float(np.mean(final)), "median", float(np.median(final)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
