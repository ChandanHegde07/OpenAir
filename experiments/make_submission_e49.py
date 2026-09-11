"""E49 v12 — gated P+Δ with E20 as a Δ feature (grec λ=0.5).

OOF: grec_l0.5 matched 239.32 / 212.50 vs E20 244.76 / 214.78 vs v11-gate ~242.5 / 213.2.
NNLS with CatBoost was slightly better but 2M-row CatBoost is too slow; LGB grec
captured almost all of the gain.

Train Δ = y−P with e20_pred from cached OOF (Jan/Jul/Dec) plus a filler model
for other months. Ranking uses v8 as e20_pred. Unmatched rows stay v8.
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

OUT = ROOT / "experiments" / "results" / "E49" / "likable-eagle_v12.parquet"
OUT_ROOT = ROOT / "likable-eagle_v12.parquet"
OUT_SUB = ROOT / "submissions" / "likable-eagle_v12.parquet"
V8_PATH = ROOT / "submissions" / "likable-eagle_v8.parquet"
SEED = 1
LAM = 0.5
P90 = 1462.0
HTHR = 200.0
NUM0 = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt", "hour", "dow", "month"]
NUM = NUM0 + ["e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def add_clocks(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_aobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("aobt_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
            pl.col("ADEP_mvt").alias("airport"),
        ]
    )


def e20_col(oof: pl.DataFrame) -> pl.DataFrame:
    e = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
    return oof.select("MVT_ID_mvt").with_columns(pl.Series("e20_pred", e))


def main():
    if not V8_PATH.exists():
        raise SystemExit(f"missing {V8_PATH}")
    log("attach OOF e20 + fill other months...")
    dep = add_clocks(load_dep()).filter(~pl.col("unmatched")).filter(pl.col("y").is_finite())
    dep = dep.filter(pl.col("mvt_aobt").is_finite())
    oof = pl.concat(
        [
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_janjul.parquet")),
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_dec.parquet")),
        ]
    )
    dep = dep.join(oof, on="MVT_ID_mvt", how="left")
    have = dep.filter(pl.col("e20_pred").is_not_null())
    miss = dep.filter(pl.col("e20_pred").is_null())
    log(f"  oof e20 n={have.height:,} fill n={miss.height:,}")
    fill = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    fill.fit(pdf(have, NUM0), have["e20_pred"].to_numpy().astype(float), categorical_feature=CAT)
    filled = np.asarray(fill.predict(pdf(miss, NUM0)), float)
    miss = miss.with_columns(pl.Series("e20_pred", filled))
    tr = pl.concat([have, miss], how="vertical_relaxed")
    y = tr["y"].to_numpy().astype(float)
    P = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    log("train Δ = y − P with e20 feature...")
    dmod = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    dmod.fit(pdf(tr, NUM), y - P, categorical_feature=CAT)

    log("ranking...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").collect()
    rank = add_clocks(rank)
    v8 = pl.read_parquet(V8_PATH)
    matched = rank.filter(pl.col("AOBT_3_flt").is_not_null())
    matched = matched.join(v8.rename({"TAXITIME_SEC_mvt": "e20_pred"}), on="MVT_ID_mvt", how="left")
    Pm = np.clip(matched["mvt_aobt"].to_numpy().astype(float), 0, None)
    rec = np.clip(Pm + np.asarray(dmod.predict(pdf(matched, NUM)), float), 0, None)
    e8 = matched["e20_pred"].to_numpy().astype(float)
    gate = (e8 > P90) | ((rec - e8) > HTHR)
    new = e8 + LAM * (rec - e8) * gate.astype(float)
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
    print("grec λ=0.5 on matched; unmatched=v8; Δ uses e20 feature")
    print("rows", out.height, "changed", nchg, "gate_frac_matched", float(gate.mean()))
    print("mean", float(np.mean(final)), "median", float(np.median(final)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
