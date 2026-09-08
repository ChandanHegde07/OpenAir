"""Ranking-safe movement stream. Never loads TAXITIME or BLOCK into sequences."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))

from common import TRAIN_FILES, sec  # noqa: E402

MOVEMENT_COLS = [
    "MVT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
]

FORBIDDEN_SEQ_COLS = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y"}


def _encode_stream(df: pl.DataFrame) -> pl.DataFrame:
    if any(c in df.columns for c in FORBIDDEN_SEQ_COLS):
        raise RuntimeError(f"sequence stream contains forbidden columns: {FORBIDDEN_SEQ_COLS & set(df.columns)}")
    df = df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("airport"),
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int8).alias("is_dep"),
        (pl.col("PHASE_mvt") == "ARR").cast(pl.Int8).alias("is_arr"),
        pl.col("AOBT_3_flt").is_null().cast(pl.Int8).alias("unmatched"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"),
        pl.col("AOBT_3_flt").is_null().cast(pl.Float32).alias("aobt_miss"),
        pl.col("EOBT_1_flt").is_null().cast(pl.Float32).alias("eobt_miss"),
        pl.col("SCHED_TIME_UTC_mvt").is_null().cast(pl.Float32).alias("sched_miss"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_aobt"),
        sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt_eobt"),
        sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_eobt"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_sched"),
        pl.col("RUNWAY_mvt").alias("runway"),
        pl.col("STAND_mvt").alias("stand"),
        pl.col("AIRCRAFT_OPERATOR_flt").alias("airline"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("actype"),
        pl.col("WK_TBL_CAT_flt").alias("wtc"),
        pl.col("ADES_mvt").alias("ades"),
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.lit("DEP")).otherwise(pl.lit("ARR")).alias("phase"),
    )
    # ARR origin-AOBT is not destination surface state. Zero those clocks.
    df = df.with_columns(
        pl.when(pl.col("is_dep") == 1).then(pl.col("mvt_aobt")).otherwise(None).alias("mvt_aobt"),
        pl.when(pl.col("is_dep") == 1).then(pl.col("aobt_eobt")).otherwise(None).alias("aobt_eobt"),
        pl.when(pl.col("is_dep") == 1).then(pl.col("mvt_eobt")).otherwise(None).alias("mvt_eobt"),
        pl.when(pl.col("is_dep") == 1).then(pl.col("aobt_miss")).otherwise(pl.lit(1.0)).alias("aobt_miss"),
        pl.when(pl.col("is_dep") == 1).then(pl.col("eobt_miss")).otherwise(pl.lit(1.0)).alias("eobt_miss"),
    )
    df = df.with_columns(
        ((pl.col("hour").cast(pl.Float32) * (2.0 * math.pi / 24.0)).sin()).alias("hour_sin"),
        ((pl.col("hour").cast(pl.Float32) * (2.0 * math.pi / 24.0)).cos()).alias("hour_cos"),
        ((pl.col("dow").cast(pl.Float32) * (2.0 * math.pi / 7.0)).sin()).alias("dow_sin"),
        ((pl.col("dow").cast(pl.Float32) * (2.0 * math.pi / 7.0)).cos()).alias("dow_cos"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t_sec"),
    )
    df = df.sort(["airport", "t_sec", "MVT_ID_mvt"])
    gap = (
        pl.col("t_sec") - pl.col("t_sec").shift(1).over("airport")
    ).fill_null(0).clip(0, 24 * 3600)
    df = df.with_columns(((gap + 1.0).log()).cast(pl.Float32).alias("prev_gap_log"))
    return df


def load_training_movements() -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(p).select(MOVEMENT_COLS) for p in TRAIN_FILES])
    return _encode_stream(lf.collect())


def load_ranking_movements(path: Path) -> pl.DataFrame:
    """Inference-only. Must not be used for vocabs, norms, or model selection."""
    raw = pl.scan_parquet(str(path)).select(MOVEMENT_COLS).collect()
    return _encode_stream(raw)
