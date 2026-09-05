"""Extreme taxi-out cases and LIRF unmatched diagnostics."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data")
OUT = Path("analysis/output")
TRAIN_FILES = sorted(glob.glob(str(DATA / "training_*.parquet")))


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def main():
    out = OUT / "05_extremes.txt"
    with open(out, "w", encoding="utf-8") as fh:
        dep = pl.concat(
            [
                pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP")
                for p in TRAIN_FILES
            ]
        ).collect()
        y = dep["TAXITIME_SEC_mvt"]
        log(fh, "n", dep.height)
        for thr in [3600, 7200, 10800, 21600, 43200, 86400]:
            sub = dep.filter(pl.col("TAXITIME_SEC_mvt") > thr)
            log(fh, f"\n> {thr}s n={sub.height} airports", sub.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))
            log(fh, "null AOBT", sub.select(pl.col("AOBT_3_flt").is_null().sum()).item())
            log(fh, "null FLIGHT_ID", sub.select(pl.col("FLIGHT_ID_mvt").is_null().sum()).item())

        log(fh, "\n===== sample >2h =====")
        cols = [
            "ADEP_mvt",
            "ADES_mvt",
            "FLIGHT_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "AOBT_3_flt",
            "LOBT_flt",
            "EOBT_1_flt",
            "MARKET_SEGMENT_flt",
        ]
        log(fh, dep.filter(pl.col("TAXITIME_SEC_mvt") > 7200).select(cols).sort("TAXITIME_SEC_mvt", descending=True).head(20))

        log(fh, "\n===== LIRF unmatched sample =====")
        lirf_u = dep.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
        log(fh, lirf_u.select(cols).sort("TAXITIME_SEC_mvt", descending=True).head(15))
        log(fh, "LIRF unmatched taxi quantiles")
        s = lirf_u["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            log(fh, q, np.quantile(s, q))
        log(fh, "flight types unmatched LIRF", lirf_u.group_by("MARKET_SEGMENT_flt").agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("med"), pl.col("TAXITIME_SEC_mvt").mean().alias("mean")))
        log(fh, "FLIGHT_RULE", lirf_u.group_by("FLIGHT_RULE_mvt").agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("med")))

        # negative taxi
        log(fh, "\n===== negative taxi =====")
        neg = dep.filter(pl.col("TAXITIME_SEC_mvt") < 0)
        log(fh, "n", neg.height, "airports", neg.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))
        log(fh, neg.select(cols).head(10))

        # ranking stands/runways unseen?
        rank = pl.scan_parquet(DATA / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt").collect()
        tr_sr = dep.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").unique()
        unseen = rank.join(tr_sr, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="anti")
        log(fh, f"\nranking DEP with unseen stand-rwy combo: {unseen.height}/{rank.height}")
        tr_st = dep.select("ADEP_mvt", "STAND_mvt").unique()
        unseen_st = rank.join(tr_st, on=["ADEP_mvt", "STAND_mvt"], how="anti")
        log(fh, f"unseen stand: {unseen_st.height}")
        tr_rw = dep.select("ADEP_mvt", "RUNWAY_mvt").unique()
        unseen_rw = rank.join(tr_rw, on=["ADEP_mvt", "RUNWAY_mvt"], how="anti")
        log(fh, f"unseen runway: {unseen_rw.height}")

        # ADEP of DEP always in 10 airports?
        log(fh, "\nDEP ADEP values", dep.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))
        log(fh, "ARR ADES values", pl.concat([pl.scan_parquet(p).filter(pl.col("PHASE_mvt")=="ARR").select("ADES_mvt") for p in TRAIN_FILES]).group_by("ADES_mvt").agg(pl.len().alias("n")).sort("n", descending=True).collect())

        # memory-ish: training dep unique operators
        log(fh, "operators", dep.select(pl.col("AIRCRAFT_OPERATOR_flt").n_unique()).item())
        log(fh, "top operators", dep.group_by("AIRCRAFT_OPERATOR_flt").agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").mean().alias("mean"), pl.col("TAXITIME_SEC_mvt").median().alias("med")).sort("n", descending=True).head(15))

        # ranking has both Jan and Jul - confirm no Feb-Jun
        r = pl.scan_parquet(DATA / "ranking.parquet").select(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), "PHASE_mvt").group_by(["m", "PHASE_mvt"]).agg(pl.len().alias("n")).sort(["m", "PHASE_mvt"]).collect()
        log(fh, "ranking months", r)

        # SCHED is always 5-min?
        log(fh, "SCHED minute of hour unique", dep.select(pl.col("SCHED_TIME_UTC_mvt").dt.minute().n_unique()).item())
        log(fh, "SCHED minutes", dep.group_by(pl.col("SCHED_TIME_UTC_mvt").dt.minute().alias("min")).agg(pl.len()).sort("min").head(20))

        # contribution of unmatched to SSE vs mean
        ynp = dep["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        mean = ynp.mean()
        sse = ((ynp - mean) ** 2).sum()
        um = dep["AOBT_3_flt"].is_null().to_numpy()
        log(fh, f"unmatched share of rows {um.mean():.4f} share of mean-model SSE {((ynp[um]-mean)**2).sum()/sse:.4f}")
        lirf = (dep["ADEP_mvt"] == "LIRF").to_numpy()
        log(fh, f"LIRF share of rows {lirf.mean():.4f} share of SSE {((ynp[lirf]-mean)**2).sum()/sse:.4f}")
        log(fh, f"LIRF unmatched share of SSE {((ynp[lirf & um]-mean)**2).sum()/sse:.4f}")
        gt1h = ynp > 3600
        log(fh, f">1h share SSE {((ynp[gt1h]-mean)**2).sum()/sse:.4f}")
        log(fh, f"LIRF >1h share SSE {((ynp[lirf & gt1h]-mean)**2).sum()/sse:.4f}")


if __name__ == "__main__":
    main()
