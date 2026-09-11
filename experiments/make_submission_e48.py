"""E48 matched-tail splice onto v8 (LB 288.90).

Do NOT touch LIRF unmatched G (v9/v10 hurt the ranking set).
Matched rows only: gated residual toward a y-model / (P+Δ) blend.

Internal OOF (E48b allgate λ=0.5): matched Jan+Jul 244.76→242.49, Dec 214.78→213.17,
top-1% SSE −4.7% / −4.1%. Train here on all 2025 matched; ranking uses v8 as the
E20 prediction for the gate.
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

OUT = ROOT / "experiments" / "results" / "E48" / "likable-eagle_v11.parquet"
OUT_ROOT = ROOT / "likable-eagle_v11.parquet"
OUT_SUB = ROOT / "submissions" / "likable-eagle_v11.parquet"
V8_PATH = ROOT / "submissions" / "likable-eagle_v8.parquet"
SEED = 1
LAM = 0.5
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt", "hour", "dow", "month"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df: pl.DataFrame):
    p = df.select(NUM + CAT).to_pandas()
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


def main():
    if not V8_PATH.exists():
        raise SystemExit(f"missing {V8_PATH}")
    log("train matched y-model (2025)...")
    dep = load_dep()
    tr = add_clocks(dep).filter(~pl.col("unmatched")).filter(pl.col("y").is_finite())
    tr = tr.filter(pl.col("mvt_aobt").is_finite())
    y = tr["y"].to_numpy().astype(float)
    p90 = float(np.quantile(y, 0.90))  # gate uses v8 at ranking; 1460 was e20 p90
    # residual vs P so the model focuses on AOBT-BLOCK gap
    P = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(pdf(tr), y - P, categorical_feature=CAT)
    log(f"  n={tr.height:,} p90_y={p90:.0f}")

    log("ranking matched...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").collect()
    rank = add_clocks(rank)
    matched = rank.filter(pl.col("AOBT_3_flt").is_not_null())
    Pm = np.clip(matched["mvt_aobt"].to_numpy().astype(float), 0, None)
    dhat = np.asarray(model.predict(pdf(matched)), float)
    rec = np.clip(Pm + dhat, 0, None)

    v8 = pl.read_parquet(V8_PATH)
    j = v8.rename({"TAXITIME_SEC_mvt": "v8"}).join(matched.select(["MVT_ID_mvt"]).with_columns(pl.Series("rec", rec)), on="MVT_ID_mvt", how="left")
    v8v = j["v8"].to_numpy().astype(float)
    recj = j["rec"].to_numpy()
    gate = j["rec"].is_not_null().to_numpy() & ((v8v > 1460.0) | ((recj - v8v) > 200.0))
    # unmatched / no rec: keep v8
    rec_fill = np.where(j["rec"].is_not_null().to_numpy(), recj, v8v)
    gate = gate & j["rec"].is_not_null().to_numpy()
    final = v8v + LAM * (rec_fill - v8v) * gate.astype(float)
    final = np.maximum(final, 0.0)
    out = j.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", final))
    sub = pl.read_parquet(SUBMIT_PATH)
    if out.height != 344841 or not (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        raise SystemExit("id check failed")
    if out["TAXITIME_SEC_mvt"].null_count():
        raise SystemExit("nulls")
    nchg = int((np.abs(final - v8v) > 1e-6).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    out.write_parquet(OUT_ROOT)
    out.write_parquet(OUT_SUB)
    print("WROTE", OUT)
    print("COPY", OUT_ROOT, OUT_SUB)
    print("base v8; matched gated Δ blend λ=0.5; unmatched untouched")
    print("rows", out.height, "changed", nchg, "gate_frac", float(gate.mean()))
    print("mean", float(np.mean(final)), "median", float(np.median(final)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
