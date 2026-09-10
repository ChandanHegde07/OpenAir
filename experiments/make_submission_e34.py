"""E34 submission — likable-eagle_v8.parquet.

Base = likable-eagle_v7. Replace the LIRF-unmatched slice with the enriched
gate-delay decomposition: T = D - G_hat, G_hat = CatBoost trained on all 2025
LIRF departures with enriched D/surface features (alpha=1.0). All other rows
keep their v7 (== E20) prediction. Inference-only ranking; no forbidden fields.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, TRAIN_FILES  # noqa: E402
from run_submitting_check import SUBMIT_PATH  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402
from run_e34d_enriched import EXTRA, enrich  # noqa: E402

OUT = ROOT / "experiments" / "results" / "E34" / "likable-eagle_v8.parquet"
OUT_ROOT = ROOT / "likable-eagle_v8.parquet"
V7 = ROOT / "analysis" / "submitting_check" / "likable-eagle_v4.parquet"  # E20 base
V7_PATH = ROOT / "likable-eagle_v7.parquet"
ALPHA = 1.0
SEED = 1
FEATS = NUM + EXTRA


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def base(non_lirf_needed=True):
    return None


def main():
    for p in (SUBMIT_PATH, V7_PATH):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    log("E34: build 2025 LIRF training frame...")
    dep = pl.scan_parquet(TRAIN_FILES).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")) \
        .select(["MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"), "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                 "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "TAXITIME_SEC_mvt"]).collect()
    arr25 = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
             .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
             .sort(["airport", "MVT_TIME_UTC_mvt"]))
    D = pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")
    dep = dep.with_columns([
        D.dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
    ])
    dep = add_surface(dep, arr25)
    dep = enrich(dep, dep["mvt_sched"].to_numpy().astype(float))
    Dt = dep["mvt_sched"].to_numpy().astype(float)
    log(f"  LIRF train rows {dep.height:,}")
    cdf = dep.select(FEATS + CAT).to_pandas()
    for c in CAT:
        cdf[c] = cdf[c].fillna("NA").astype(str)
    model = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, loss_function="RMSE",
                              random_seed=SEED, verbose=False, thread_count=-1, od_type="Iter", od_wait=80)
    model.fit(cdf, Dt - dep["y"].to_numpy().astype(float), cat_features=CAT)

    log("  featurize ranking LIRF...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet")
    rdep = rank.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).select([
        "MVT_ID_mvt", "FLIGHT_mvt", pl.col("ADEP_mvt").alias("airport"), "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt"]).collect()
    rarr = (rank.filter(pl.col("PHASE_mvt") == "ARR")
            .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
            .sort(["airport", "MVT_TIME_UTC_mvt"]))
    Dr = pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")
    rdep = rdep.with_columns([
        Dr.dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"),
    ])
    rdep = add_surface(rdep, rarr)
    rdep = enrich(rdep, Dt)
    lu = rdep.filter(pl.col("AOBT_3_flt").is_null())
    log(f"  ranking LIRF unmatched {lu.height}")
    pdf = lu.select(FEATS + CAT).to_pandas()
    for c in CAT:
        pdf[c] = pdf[c].fillna("NA").astype(str)
    g = np.asarray(model.predict(pdf), float)
    dr = lu["mvt_sched"].to_numpy().astype(float)
    t_new = np.maximum(dr - ALPHA * g, 0.0)

    log("  blend into v7 base (LIRF unmatched only)...")
    v7 = pl.read_parquet(V7_PATH)
    repl = lu.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", t_new))
    basev7 = v7.with_row_index("_i")
    final = (basev7.join(repl.rename({"TAXITIME_SEC_mvt": "_r"}), on="MVT_ID_mvt", how="left")
                   .with_columns(pl.when(pl.col("_r").is_not_null()).then(pl.col("_r")).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"))
                   .drop("_r").sort("_i").drop("_i").select("MVT_ID_mvt", "TAXITIME_SEC_mvt"))
    sub = pl.read_parquet(SUBMIT_PATH)
    if final.height != 344841 or final["MVT_ID_mvt"].n_unique() != final.height:
        raise SystemExit("row/id check failed")
    if not (final["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        raise SystemExit("id order mismatch")
    if final["TAXITIME_SEC_mvt"].null_count() or int(final["TAXITIME_SEC_mvt"].is_nan().sum()):
        raise SystemExit("null/nan")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(OUT)
    final.write_parquet(OUT_ROOT)
    rp = final["TAXITIME_SEC_mvt"].to_numpy()
    print("WROTE", OUT)
    print("COPY", OUT_ROOT)
    print("model E34: v7 + enriched CatBoost gate-delay decomposition (alpha=1.0, LIRF unmatched)")
    print("rows", final.height, "changed_rows", int((rp != v7["TAXITIME_SEC_mvt"].to_numpy()).sum()), "nulls", final["TAXITIME_SEC_mvt"].null_count())
    print("mean", float(np.mean(rp)), "median", float(np.median(rp)), "min", float(np.min(rp)), "max", float(np.max(rp)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
