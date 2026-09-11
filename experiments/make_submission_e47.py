"""E47 submission — likable-eagle_v10.parquet.

Base = v9. Replace LIRF-unmatched rows with G trained on unmatched LIRF 2025
using E35 features + stand-release + in_opdi (OPDI open ADS-B flight list).
ISR/ETH prefixes are essentially T=D on both holdouts; force T=D there.

in_opdi on ranking uses OPDI 2026-01 and 2026-07 flight lists (open, documented).
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
from run_e45_stand_release import REL, add_stand_release  # noqa: E402
from run_e47_opdi_regime import attach_in_opdi, load_opdi_lirf  # noqa: E402

OUT = ROOT / "experiments" / "results" / "E47" / "likable-eagle_v10.parquet"
OUT_ROOT = ROOT / "likable-eagle_v10.parquet"
V9_PATH = ROOT / "submissions" / "likable-eagle_v9.parquet"
if not V9_PATH.exists():
    V9_PATH = ROOT / "experiments" / "results" / "E35" / "likable-eagle_v9.parquet"
SEED = 1
FEATS = NUM + EXTRA + REL + ["in_opdi"]
FORCE_D_PREFIX = {"ISR", "ETH"}
LIST_DIR = ROOT / "data" / "external" / "opdi" / "flight_list"


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pick(df):
    x = df.select(FEATS + CAT).to_pandas()
    for c in CAT:
        x[c] = x[c].fillna("NA").astype(str)
    return x


def load_opdi_lirf_years(years: tuple[int, ...]) -> pl.DataFrame:
    frames = []
    for p in sorted(LIST_DIR.glob("flight_list_*.parquet")):
        ym = p.stem.replace("flight_list_", "")
        if not ym[:4].isdigit() or int(ym[:4]) not in years:
            continue
        df = pl.read_parquet(p).select(["flt_id", "adep", "first_seen"])
        if df.schema["first_seen"] == pl.String:
            df = df.with_columns(pl.col("first_seen").str.to_datetime(strict=False))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no OPDI lists for years {years}")
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("adep") == "LIRF")
        .with_columns(pl.col("flt_id").fill_null("").str.to_uppercase().str.replace_all(" ", "").alias("cs"))
        .filter(pl.col("cs") != "")
        .select(["cs", "first_seen"])
    )


def prep_lirf(dep: pl.DataFrame, arr_surf: pl.DataFrame, arr_blk: pl.DataFrame, op: pl.DataFrame, dref: np.ndarray):
    dep = dep.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
            pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"),
        ]
    )
    dep = add_surface(dep, arr_surf)
    dep = enrich(dep, dref)
    dep = add_stand_release(dep, arr_blk)
    dep = attach_in_opdi(dep.with_columns(pl.col("AOBT_3_flt").is_null().alias("unmatched")), op)
    return dep


def main():
    if not V9_PATH.exists():
        raise SystemExit(f"missing {V9_PATH}")
    log("E47: train unmatched LIRF G...")
    dep = (
        pl.scan_parquet(TRAIN_FILES)
        .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
        .select(
            [
                "MVT_ID_mvt",
                pl.col("ADEP_mvt").alias("airport"),
                "ADES_mvt",
                "MVT_TIME_UTC_mvt",
                "SCHED_TIME_UTC_mvt",
                "RUNWAY_mvt",
                "STAND_mvt",
                "AIRCRAFT_TYPE_mvt",
                "FLIGHT_mvt",
                "TAXITIME_SEC_mvt",
                "AOBT_3_flt",
            ]
        )
        .collect()
        .with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
    )
    arr_surf = (
        pl.scan_parquet(TRAIN_FILES)
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select([pl.lit("LIRF").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    arr_blk = (
        pl.scan_parquet(TRAIN_FILES)
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select(["STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        .collect()
    )
    op25 = load_opdi_lirf_years((2025,))
    dep = prep_lirf(dep, arr_surf, arr_blk, op25, dep.select((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).to_series().to_numpy().astype(float))
    tr_u = dep.filter(pl.col("unmatched"))
    Dt = tr_u["mvt_sched"].to_numpy().astype(float)
    yt = tr_u["y"].to_numpy().astype(float)
    log(f"  unmatched train {tr_u.height:,} in_opdi {float(tr_u['in_opdi'].mean()):.3f}")
    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
        od_type="Iter",
        od_wait=80,
    )
    model.fit(pick(tr_u), Dt - yt, cat_features=CAT)

    log("  ranking LIRF...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet")
    rdep = rank.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).select(
        [
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            pl.col("ADEP_mvt").alias("airport"),
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "AOBT_3_flt",
        ]
    ).collect()
    rarr_surf = (
        rank.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select([pl.lit("LIRF").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    rarr_blk = (
        rank.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select(["STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        .collect()
    )
    op26 = load_opdi_lirf_years((2026,))
    rdep = prep_lirf(rdep, rarr_surf, rarr_blk, op26, Dt)
    lu = rdep.filter(pl.col("AOBT_3_flt").is_null())
    g = np.asarray(model.predict(pick(lu)), float)
    dr = lu["mvt_sched"].to_numpy().astype(float)
    t_new = np.maximum(dr - g, 0.0)
    pref = lu["flt_prefix"].to_numpy()
    force = np.isin(pref, list(FORCE_D_PREFIX))
    t_new = np.where(force, np.maximum(dr, 0.0), t_new)
    log(f"  ranking LIRF unmatched {lu.height} ISR/ETH force-D {int(force.sum())} in_opdi {float(lu['in_opdi'].mean()):.3f}")

    v9 = pl.read_parquet(V9_PATH)
    repl = lu.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", t_new))
    final = (
        v9.with_row_index("_i")
        .join(repl.rename({"TAXITIME_SEC_mvt": "_r"}), on="MVT_ID_mvt", how="left")
        .with_columns(
            pl.when(pl.col("_r").is_not_null()).then(pl.col("_r")).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt")
        )
        .drop("_r")
        .sort("_i")
        .drop("_i")
        .select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    )
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
    nchg = int((rp != v9["TAXITIME_SEC_mvt"].to_numpy()).sum())
    print("WROTE", OUT)
    print("COPY", OUT_ROOT)
    print("model E47: unmatched-only CatBoost G + stand-release + in_opdi, ISR/ETH -> D")
    print("rows", final.height, "changed_rows", nchg)
    print("mean", float(np.mean(rp)), "median", float(np.median(rp)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
