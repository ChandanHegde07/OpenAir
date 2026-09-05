"""Part 1-4: dataset structure, target identity, timestamps, leakage audit."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

DATA = Path("data")
OUT = Path("analysis/output")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = sorted(glob.glob(str(DATA / "training_*.parquet")))
TS_COLS = [
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
]
CAT_COLS = [
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
]


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def file_overview(fh):
    log(fh, "=" * 80)
    log(fh, "FILE OVERVIEW")
    log(fh, "=" * 80)
    for path in TRAIN_FILES + [str(DATA / "ranking.parquet"), str(DATA / "submitting.parquet")]:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
        n = lf.select(pl.len()).collect().item()
        size = os.path.getsize(path)
        log(fh, f"{Path(path).name:50s} rows={n:>10,} cols={len(schema):2d} bytes={size:>12,}")
        log(fh, "  columns: " + ", ".join(schema.names()))


def compare_schemas(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "SCHEMA COMPARISON")
    log(fh, "=" * 80)
    schemas = {}
    for path in TRAIN_FILES + [str(DATA / "ranking.parquet"), str(DATA / "submitting.parquet")]:
        schemas[Path(path).name] = list(pl.scan_parquet(path).collect_schema().items())
    train_ref = schemas[Path(TRAIN_FILES[0]).name]
    log(fh, "Training month schemas identical:", all(schemas[Path(p).name] == train_ref for p in TRAIN_FILES))
    rank = schemas["ranking.parquet"]
    log(fh, "Ranking schema identical to training:", rank == train_ref)
    log(fh, "Submitting schema:", schemas["submitting.parquet"])
    # types
    log(fh, "\nTraining/ranking dtypes:")
    for name, dtype in train_ref:
        log(fh, f"  {name:28s} {dtype}")


def ranking_vs_training(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "RANKING vs TRAINING: blanked columns, phases, dates")
    log(fh, "=" * 80)
    rank = pl.scan_parquet(DATA / "ranking.parquet")
    schema = rank.collect_schema()
    n = rank.select(pl.len()).collect().item()
    log(fh, "ranking rows", n)

    # missing by phase
    phase = rank.group_by("PHASE_mvt").agg(pl.len().alias("n")).collect()
    log(fh, "\nranking PHASE counts:")
    log(fh, phase)

    miss_exprs = []
    for c in schema.names():
        miss_exprs.append(pl.col(c).is_null().sum().alias(c))
    miss = rank.select(miss_exprs).collect()
    log(fh, "\nranking overall missing counts:")
    for c in schema.names():
        m = miss[c][0]
        log(fh, f"  {c:28s} missing={m:>10,} ({100*m/n:6.2f}%)")

    log(fh, "\nranking missing by PHASE:")
    miss_by = (
        rank.group_by("PHASE_mvt")
        .agg(
            pl.len().alias("n"),
            *[pl.col(c).is_null().sum().alias(c) for c in schema.names() if c != "PHASE_mvt"],
        )
        .collect()
    )
    log(fh, miss_by)

    # date ranges
    log(fh, "\nranking date ranges:")
    for c in TS_COLS:
        s = rank.select(
            pl.col(c).min().alias("min"),
            pl.col(c).max().alias("max"),
            pl.col(c).is_null().sum().alias("nulls"),
        ).collect()
        log(fh, f"  {c:28s} min={s['min'][0]} max={s['max'][0]} nulls={s['nulls'][0]}")

    # unique months of MVT_TIME
    months = (
        rank.filter(pl.col("MVT_TIME_UTC_mvt").is_not_null())
        .select(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("n"))
        .sort("month")
        .collect()
    )
    log(fh, "\nranking MVT_TIME months:")
    log(fh, months)

    # airports
    ap = (
        rank.with_columns(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt"))
            .otherwise(pl.col("ADES_mvt"))
            .alias("airport")
        )
        .group_by(["airport", "PHASE_mvt"])
        .agg(pl.len().alias("n"))
        .sort(["airport", "PHASE_mvt"])
        .collect()
    )
    log(fh, "\nranking airport x phase:")
    log(fh, ap)

    # submitting
    sub = pl.scan_parquet(DATA / "submitting.parquet")
    subn = sub.select(pl.len()).collect().item()
    log(fh, "\nsubmitting rows", subn)
    sub_miss = sub.select(pl.col("TAXITIME_SEC_mvt").is_null().sum()).collect().item()
    log(fh, "submitting TAXITIME nulls", sub_miss)
    log(fh, "submitting TAXITIME unique", sub.select(pl.col("TAXITIME_SEC_mvt").n_unique()).collect().item())
    log(fh, "submitting TAXITIME sample", sub.select("TAXITIME_SEC_mvt").head(10).collect())

    # ID overlap
    rank_dep_ids = rank.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt")
    sub_ids = sub.select("MVT_ID_mvt")
    n_dep = rank_dep_ids.select(pl.len()).collect().item()
    n_match = rank_dep_ids.join(sub_ids, on="MVT_ID_mvt", how="inner").select(pl.len()).collect().item()
    n_sub_only = sub_ids.join(rank_dep_ids, on="MVT_ID_mvt", how="anti").select(pl.len()).collect().item()
    log(fh, f"ranking DEP={n_dep:,} submitting={subn:,} inner={n_match:,} sub_only={n_sub_only:,}")

    # AOBT vs BLOCK on ranking DEP
    log(fh, "\nranking DEP: AOBT_3 / LOBT / BLOCK / TAXITIME presence")
    dep = rank.filter(pl.col("PHASE_mvt") == "DEP")
    n_dep = dep.select(pl.len()).collect().item()
    for c in [
        "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "LOBT_flt",
        "IOBT_flt",
        "EOBT_1_flt",
        "AOBT_3_flt",
        "ARVT_1_flt",
        "ARVT_3_flt",
        "RUNWAY_mvt",
        "STAND_mvt",
    ]:
        nn = dep.select(pl.col(c).is_not_null().sum()).collect().item()
        log(fh, f"  {c:28s} non-null={nn:>10,} ({100*nn/n_dep:6.2f}%)")


def train_profile(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "TRAINING PROFILE (all months)")
    log(fh, "=" * 80)
    train = pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
    n = train.select(pl.len()).collect().item()
    log(fh, "total training rows", n)

    schema = train.collect_schema()
    miss = train.select([pl.col(c).is_null().sum().alias(c) for c in schema.names()]).collect()
    log(fh, "\nmissing overall:")
    for c in schema.names():
        m = miss[c][0]
        log(fh, f"  {c:28s} missing={m:>10,} ({100*m/n:6.2f}%)")

    log(fh, "\nmissing by PHASE:")
    miss_by = (
        train.group_by("PHASE_mvt")
        .agg(
            pl.len().alias("n"),
            *[pl.col(c).is_null().sum().alias(c) for c in schema.names() if c != "PHASE_mvt"],
        )
        .collect()
    )
    log(fh, miss_by)

    log(fh, "\nunique counts:")
    uniques = train.select([pl.col(c).n_unique().alias(c) for c in schema.names()]).collect()
    for c in schema.names():
        log(fh, f"  {c:28s} unique={uniques[c][0]:,}")

    log(fh, "\nPHASE counts:")
    log(fh, train.group_by("PHASE_mvt").agg(pl.len().alias("n")).collect())

    log(fh, "\ncategorical value counts (top):")
    for c in [
        "FLIGHT_RULE_mvt",
        "PHASE_mvt",
        "MARKET_SEGMENT_flt",
        "FLIGHT_RULE_flt",
        "FLIGHT_TYPE_flt",
        "WK_TBL_CAT_flt",
        "ADEP_mvt",
        "ADES_mvt",
    ]:
        vc = train.group_by(c).agg(pl.len().alias("n")).sort("n", descending=True).head(20).collect()
        log(fh, f"\n--- {c} ---")
        log(fh, vc)

    # airport from phase
    train_ap = train.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("airport")
    )
    log(fh, "\nairport x phase:")
    log(
        fh,
        train_ap.group_by(["airport", "PHASE_mvt"]).agg(pl.len().alias("n")).sort(["airport", "PHASE_mvt"]).collect(),
    )

    log(fh, "\ndate ranges training:")
    for c in TS_COLS:
        s = train.select(
            pl.col(c).min().alias("min"),
            pl.col(c).max().alias("max"),
            pl.col(c).is_null().sum().alias("nulls"),
        ).collect()
        log(fh, f"  {c:28s} min={s['min'][0]} max={s['max'][0]} nulls={s['nulls'][0]}")

    months = (
        train.filter(pl.col("MVT_TIME_UTC_mvt").is_not_null())
        .select(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo").alias("month"), "PHASE_mvt")
        .group_by(["month", "PHASE_mvt"])
        .agg(pl.len().alias("n"))
        .sort(["month", "PHASE_mvt"])
        .collect()
    )
    log(fh, "\ntraining MVT_TIME months x phase:")
    log(fh, months)

    # ID uniqueness
    log(fh, "\nID uniqueness:")
    log(fh, "MVT_ID unique", train.select(pl.col("MVT_ID_mvt").n_unique()).collect().item())
    log(fh, "FLIGHT_ID unique", train.select(pl.col("FLIGHT_ID_mvt").n_unique()).collect().item())
    dups = train.group_by("MVT_ID_mvt").agg(pl.len().alias("n")).filter(pl.col("n") > 1).select(pl.len()).collect().item()
    log(fh, "duplicate MVT_IDs", dups)
    fid_dups = (
        train.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
        .group_by("FLIGHT_ID_mvt")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    log(fh, "FLIGHT_IDs appearing >1 time", fid_dups)


def target_identity(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "TARGET IDENTITY: TAXITIME vs MVT-BLOCK")
    log(fh, "=" * 80)
    train = pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
    df = train.select(
        "PHASE_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "AOBT_3_flt",
        "LOBT_flt",
        "EOBT_1_flt",
        "IOBT_flt",
        "SCHED_TIME_UTC_mvt",
        "ARVT_1_flt",
        "ARVT_3_flt",
        "ADEP_mvt",
        "ADES_mvt",
        "MVT_ID_mvt",
        "FLIGHT_ID_mvt",
    ).with_columns(
        ((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()).alias("mvt_minus_block"),
        ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()).alias("mvt_minus_aobt"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()).alias("block_minus_aobt"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds()).alias("block_minus_lobt"),
        ((pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds()).alias("aobt_minus_lobt"),
        ((pl.col("EOBT_1_flt") - pl.col("IOBT_flt")).dt.total_seconds()).alias("eobt_minus_iobt"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("EOBT_1_flt")).dt.total_seconds()).alias("block_minus_eobt"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).alias("block_minus_sched"),
        ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).alias("mvt_minus_sched"),
        ((pl.col("ARVT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()).alias("arvt3_minus_mvt"),
        ((pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds()).alias("planned_duration"),
        ((pl.col("ARVT_3_flt") - pl.col("AOBT_3_flt")).dt.total_seconds()).alias("actual_block_to_arr"),
    )

    for phase in ["DEP", "ARR"]:
        sub = df.filter(pl.col("PHASE_mvt") == phase)
        n = sub.select(pl.len()).collect().item()
        log(fh, f"\n===== PHASE {phase} n={n:,} =====")

        # taxitime vs mvt-block
        both = sub.filter(
            pl.col("TAXITIME_SEC_mvt").is_not_null()
            & pl.col("mvt_minus_block").is_not_null()
        )
        nb = both.select(pl.len()).collect().item()
        exact = both.filter(pl.col("TAXITIME_SEC_mvt") == pl.col("mvt_minus_block")).select(pl.len()).collect().item()
        absdiff = both.select(
            (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_block")).abs().alias("ad")
        )
        log(fh, f"rows with both TAXITIME and MVT-BLOCK: {nb:,}")
        log(fh, f"exact equality TAXITIME == MVT-BLOCK: {exact:,} ({100*exact/nb if nb else 0:.4f}%)")
        qs = absdiff.select(
            pl.col("ad").mean().alias("mean"),
            pl.col("ad").median().alias("median"),
            pl.col("ad").quantile(0.9).alias("p90"),
            pl.col("ad").quantile(0.99).alias("p99"),
            pl.col("ad").max().alias("max"),
            (pl.col("ad") == 0).mean().alias("frac0"),
            (pl.col("ad") <= 1).mean().alias("frac_le1"),
            (pl.col("ad") <= 60).mean().alias("frac_le60"),
        ).collect()
        log(fh, "abs(TAXITIME - (MVT-BLOCK)) stats:", qs)

        # sign of mvt-block
        sign = (
            sub.filter(pl.col("mvt_minus_block").is_not_null())
            .select(
                (pl.col("mvt_minus_block") > 0).sum().alias("pos"),
                (pl.col("mvt_minus_block") == 0).sum().alias("zero"),
                (pl.col("mvt_minus_block") < 0).sum().alias("neg"),
                pl.col("mvt_minus_block").min().alias("min"),
                pl.col("mvt_minus_block").max().alias("max"),
            )
            .collect()
        )
        log(fh, "MVT-BLOCK sign:", sign)

        # AOBT vs BLOCK
        aobt = sub.filter(pl.col("block_minus_aobt").is_not_null())
        na = aobt.select(pl.len()).collect().item()
        if na:
            aqs = aobt.select(
                pl.col("block_minus_aobt").abs().mean().alias("mean_abs"),
                pl.col("block_minus_aobt").abs().median().alias("med_abs"),
                (pl.col("block_minus_aobt").abs() == 0).mean().alias("frac0"),
                (pl.col("block_minus_aobt").abs() <= 60).mean().alias("frac_le60"),
                (pl.col("block_minus_aobt").abs() <= 300).mean().alias("frac_le300"),
                pl.col("block_minus_aobt").min().alias("min"),
                pl.col("block_minus_aobt").max().alias("max"),
            ).collect()
            log(fh, f"BLOCK - AOBT_3  n={na:,}", aqs)

        lobt = sub.filter(pl.col("block_minus_lobt").is_not_null())
        nl = lobt.select(pl.len()).collect().item()
        if nl:
            lqs = lobt.select(
                pl.col("block_minus_lobt").abs().mean().alias("mean_abs"),
                pl.col("block_minus_lobt").abs().median().alias("med_abs"),
                (pl.col("block_minus_lobt").abs() == 0).mean().alias("frac0"),
                (pl.col("block_minus_lobt").abs() <= 60).mean().alias("frac_le60"),
                (pl.col("block_minus_lobt").abs() <= 300).mean().alias("frac_le300"),
                pl.col("block_minus_lobt").min().alias("min"),
                pl.col("block_minus_lobt").max().alias("max"),
            ).collect()
            log(fh, f"BLOCK - LOBT  n={nl:,}", lqs)

        # MVT vs AOBT as taxi proxy
        ma = sub.filter(pl.col("mvt_minus_aobt").is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null())
        nma = ma.select(pl.len()).collect().item()
        if nma:
            maqs = ma.select(
                (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt")).abs().mean().alias("mean_abs"),
                (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt")).abs().median().alias("med_abs"),
                ((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt")).abs() == 0).mean().alias("frac0"),
                ((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt")).abs() <= 60).mean().alias("frac_le60"),
            ).collect()
            log(fh, f"TAXITIME vs MVT-AOBT n={nma:,}", maqs)

        # target stats
        t = sub.filter(pl.col("TAXITIME_SEC_mvt").is_not_null()).select("TAXITIME_SEC_mvt")
        nt = t.select(pl.len()).collect().item()
        stats = t.select(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
            pl.col("TAXITIME_SEC_mvt").min().alias("min"),
            pl.col("TAXITIME_SEC_mvt").max().alias("max"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.01).alias("p01"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.05).alias("p05"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.25).alias("p25"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.75).alias("p75"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.90).alias("p90"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.95).alias("p95"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.99).alias("p99"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.999).alias("p999"),
            (pl.col("TAXITIME_SEC_mvt") < 0).sum().alias("neg"),
            (pl.col("TAXITIME_SEC_mvt") == 0).sum().alias("zero"),
            (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("gt_1h"),
            (pl.col("TAXITIME_SEC_mvt") > 7200).sum().alias("gt_2h"),
        ).collect()
        log(fh, f"TAXITIME stats n={nt:,}")
        log(fh, stats)

        # RMSE contribution of tails: share of SSE
        # We'll compute later in numpy if needed

    # sample rows
    log(fh, "\n===== SAMPLE DEP ROWS =====")
    log(
        fh,
        train.filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "LOBT_flt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_1_flt",
            "ARVT_3_flt",
            "CALLSIGN_flt",
            "MARKET_SEGMENT_flt",
            "WK_TBL_CAT_flt",
            "AIRCRAFT_OPERATOR_flt",
        )
        .head(8)
        .collect(),
    )
    log(fh, "\n===== SAMPLE ARR ROWS =====")
    log(
        fh,
        train.filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
            "EOBT_1_flt",
        )
        .head(8)
        .collect(),
    )


def ranking_samples(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "RANKING SAMPLE ROWS")
    log(fh, "=" * 80)
    rank = pl.scan_parquet(DATA / "ranking.parquet")
    log(fh, "\n===== RANKING DEP =====")
    log(
        fh,
        rank.filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "LOBT_flt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_1_flt",
            "ARVT_3_flt",
        )
        .head(8)
        .collect(),
    )
    log(fh, "\n===== RANKING ARR =====")
    log(
        fh,
        rank.filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
            "EOBT_1_flt",
        )
        .head(8)
        .collect(),
    )


def monthly_missing(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "PER-MONTH MISSING % (to detect schema drift in values)")
    log(fh, "=" * 80)
    cols = [
        "TAXITIME_SEC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "RUNWAY_mvt",
        "STAND_mvt",
        "AIRCRAFT_TYPE_mvt",
        "LOBT_flt",
        "IOBT_flt",
        "EOBT_1_flt",
        "AOBT_3_flt",
        "ARVT_1_flt",
        "ARVT_3_flt",
        "FLIGHT_ID_mvt",
        "CALLSIGN_flt",
        "MARKET_SEGMENT_flt",
        "WK_TBL_CAT_flt",
        "AIRCRAFT_OPERATOR_flt",
        "ADES_FILED_flt",
    ]
    for path in TRAIN_FILES:
        lf = pl.scan_parquet(path)
        n = lf.select(pl.len()).collect().item()
        miss = lf.select([pl.col(c).is_null().mean().alias(c) for c in cols]).collect()
        log(fh, f"\n{Path(path).name} n={n:,}")
        for c in cols:
            log(fh, f"  {c:28s} miss={100*miss[c][0]:6.2f}%")


def main():
    out_path = OUT / "01_structure_leakage.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        file_overview(fh)
        compare_schemas(fh)
        ranking_vs_training(fh)
        train_profile(fh)
        target_identity(fh)
        ranking_samples(fh)
        monthly_missing(fh)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
