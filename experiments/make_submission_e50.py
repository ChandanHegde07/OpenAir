"""E50 v13 — rich gated P+Δ plus leftover residual (λ=0.25).

OOF grec_hat_l0.25: matched 238.52 / 212.08 vs v12-like grec 239.02 / 212.38.
Unmatched stays v8. No LIRF unmatched G.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, load_dep  # noqa: E402
from run_submitting_check import SUBMIT_PATH  # noqa: E402

OUT = ROOT / "experiments" / "results" / "E50" / "likable-eagle_v13.parquet"
OUT_ROOT = ROOT / "likable-eagle_v13.parquet"
OUT_SUB = ROOT / "submissions" / "likable-eagle_v13.parquet"
V8_PATH = ROOT / "submissions" / "likable-eagle_v8.parquet"
SEED = 1
LAM_REC, LAM_HAT = 0.5, 0.25
P90, HTHR = 1462.0, 200.0
NUM0 = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
]
NUM = NUM0 + ["e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def add_extra(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_aobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("aobt_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_lobt"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_iobt"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
            pl.col("ADEP_mvt").alias("airport"),
            pl.col("AIRCRAFT_TYPE_mvt").str.slice(0, 3).alias("ac_family"),
            pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
        ]
    )
    df = df.with_columns(pl.col("aobt_eobt").abs().alias("abs_aobt_eobt"))
    clocks = df.select(["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"])
    arr = []
    for c in clocks.columns:
        arr.append(clocks[c].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[ns]").astype(np.float64))
    stack = np.stack(arr, axis=1) / 1e9
    with np.errstate(all="ignore"):
        clock_std = np.nanstd(stack, axis=1, ddof=0)
        clock_range = np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)
        n_clocks = np.sum(np.isfinite(stack), axis=1)
    clock_std[~np.isfinite(clock_std)] = np.nan
    clock_range[~np.isfinite(clock_range)] = np.nan
    return df.with_columns(
        pl.Series("clock_std", clock_std),
        pl.Series("clock_range", clock_range),
        pl.Series("n_clocks", n_clocks.astype(np.int32)),
        pl.col("ac_family").fill_null("NA"),
    )


def e20_col(oof):
    e = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
    return oof.select("MVT_ID_mvt").with_columns(pl.Series("e20_pred", e))


def main():
    if not V8_PATH.exists():
        raise SystemExit(f"missing {V8_PATH}")
    log("train frames...")
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
    log(f"  n={tr.height:,} fit Δ and residual...")
    dmod = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    dmod.fit(pdf(tr, NUM), y - P, categorical_feature=CAT)
    hmod = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED + 2, n_jobs=-1, verbose=-1,
    )
    hmod.fit(pdf(tr, NUM), y - e_tr, categorical_feature=CAT)

    log("ranking...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").collect()
    rank = add_extra(rank)
    v8 = pl.read_parquet(V8_PATH)
    matched = rank.filter(pl.col("AOBT_3_flt").is_not_null())
    matched = matched.join(v8.rename({"TAXITIME_SEC_mvt": "e20_pred"}), on="MVT_ID_mvt", how="left")
    Pm = np.clip(matched["mvt_aobt"].to_numpy().astype(float), 0, None)
    rec = np.clip(Pm + np.asarray(dmod.predict(pdf(matched, NUM)), float), 0, None)
    hat = np.asarray(hmod.predict(pdf(matched, NUM)), float)
    e8 = matched["e20_pred"].to_numpy().astype(float)
    gate = (e8 > P90) | ((rec - e8) > HTHR)
    grec = e8 + LAM_REC * (rec - e8) * gate.astype(float)
    new = grec + LAM_HAT * hat
    new = np.maximum(new, 0.0)

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
    if out["TAXITIME_SEC_mvt"].null_count():
        raise SystemExit("nulls")
    nchg = int((np.abs(final - j["v8"].to_numpy()) > 1e-6).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    out.write_parquet(OUT_ROOT)
    out.write_parquet(OUT_SUB)
    print("WROTE", OUT)
    print("COPY", OUT_ROOT, OUT_SUB)
    print("grec λ=0.5 + leftover hat λ=0.25; unmatched=v8")
    print("rows", out.height, "changed", nchg)
    print("mean", float(np.mean(final)), "median", float(np.median(final)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
