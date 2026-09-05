"""E7: FLIGHT_ID / turnaround semantics. Do not assume it is a tail number."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, TRAIN_FILES, load_dep, save_result, split_by_months, metrics_block, fmt, airport_mean_fallback  # noqa: E402


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def load_all_movements() -> pl.DataFrame:
    cols = [
        "MVT_ID_mvt",
        "FLIGHT_ID_mvt",
        "FLIGHT_mvt",
        "PHASE_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt",
        "TAXITIME_SEC_mvt",
        "CALLSIGN_flt",
        "AOBT_3_flt",
    ]
    return (
        pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES])
        .with_columns(
            pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("airport"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        )
        .collect()
    )


def main():
    out = Path(__file__).resolve().parent / "results" / "E7.txt"
    with open(out, "w", encoding="utf-8") as fh:
        log(fh, "========== E7 FLIGHT_ID / TURNAROUND ==========")
        df = load_all_movements()
        log(fh, f"all movements {df.height:,}")

        # --- multiplicity ---
        g = (
            df.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
            .group_by("FLIGHT_ID_mvt")
            .agg(
                pl.len().alias("n"),
                pl.col("PHASE_mvt").n_unique().alias("n_phase"),
                pl.col("airport").n_unique().alias("n_airport"),
                pl.col("AIRCRAFT_TYPE_mvt").n_unique().alias("n_type"),
                pl.col("FLIGHT_mvt").n_unique().alias("n_flightno"),
                pl.col("CALLSIGN_flt").n_unique().alias("n_cs"),
            )
        )
        log(fh, "FLIGHT_ID multiplicity")
        log(fh, g.group_by("n").agg(pl.len().alias("n_ids")).sort("n"))
        log(fh, "n==2 distinct phases", g.filter((pl.col("n") == 2) & (pl.col("n_phase") == 2)).height)
        log(fh, "n==2 same phase", g.filter((pl.col("n") == 2) & (pl.col("n_phase") == 1)).height)
        log(fh, "n==2 n_airport", g.filter(pl.col("n") == 2).group_by("n_airport").agg(pl.len()))
        log(fh, "n==2 n_type>1", g.filter((pl.col("n") == 2) & (pl.col("n_type") > 1)).height)
        log(fh, "n==2 n_flightno>1", g.filter((pl.col("n") == 2) & (pl.col("n_flightno") > 1)).height)

        # pair rows for n==2
        two_ids = g.filter(pl.col("n") == 2).select("FLIGHT_ID_mvt")
        pairs = df.join(two_ids, on="FLIGHT_ID_mvt").sort(["FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"])
        # first vs second
        first = pairs.group_by("FLIGHT_ID_mvt").agg(
            pl.col("PHASE_mvt").first().alias("ph1"),
            pl.col("PHASE_mvt").last().alias("ph2"),
            pl.col("airport").first().alias("ap1"),
            pl.col("airport").last().alias("ap2"),
            pl.col("ADEP_mvt").first().alias("adep1"),
            pl.col("ADES_mvt").first().alias("ades1"),
            pl.col("ADEP_mvt").last().alias("adep2"),
            pl.col("ADES_mvt").last().alias("ades2"),
            pl.col("MVT_TIME_UTC_mvt").first().alias("t1"),
            pl.col("MVT_TIME_UTC_mvt").last().alias("t2"),
        )
        first = first.with_columns((pl.col("t2") - pl.col("t1")).dt.total_seconds().alias("dt"))
        log(fh, "phase order among n==2 (time-sorted)")
        log(fh, first.group_by(["ph1", "ph2"]).agg(pl.len().alias("n")).sort("n", descending=True))
        log(fh, "same airport both rows", first.filter(pl.col("ap1") == pl.col("ap2")).height)
        log(fh, "DEP then ARR and ADEP1==ADES2 and ADES1==ADEP2 (city-pair outbound+inbound SAME airports)? wait")
        # Same flight A->B: DEP at A (ADEP=A, ADES=B) and ARR at B (ADEP=A, ADES=B)
        same_od = first.filter(
            (pl.col("ph1") == "DEP")
            & (pl.col("ph2") == "ARR")
            & (pl.col("adep1") == pl.col("adep2"))
            & (pl.col("ades1") == pl.col("ades2"))
        )
        log(fh, "DEP then ARR, identical ADEP/ADES (same city-pair flight at two challenge airports)", same_od.height)
        log(fh, "dt seconds for those", same_od.select(pl.col("dt").mean().alias("mean"), pl.col("dt").median().alias("med"), pl.col("dt").min().alias("min"), pl.col("dt").max().alias("max")))
        # ARR then DEP same FLIGHT_ID would be weird
        log(fh, "ARR then DEP", first.filter((pl.col("ph1") == "ARR") & (pl.col("ph2") == "DEP")).height)

        # --- turnaround via STAND: previous ARR at same airport+stand ---
        log(fh, "\n--- stand-based previous ARR (turnaround candidate) ---")
        dep = df.filter(pl.col("PHASE_mvt") == "DEP").select(
            "MVT_ID_mvt",
            "airport",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "FLIGHT_ID_mvt",
            "FLIGHT_mvt",
            "CALLSIGN_flt",
            "BLOCK_TIME_UTC_mvt",
            "MVT_TIME_UTC_mvt",
            "y",
        )
        arr = df.filter(pl.col("PHASE_mvt") == "ARR").select(
            "airport",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "FLIGHT_ID_mvt",
            "FLIGHT_mvt",
            "CALLSIGN_flt",
            pl.col("MVT_TIME_UTC_mvt").alias("arr_land"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("arr_inblock"),
            pl.col("y").alias("taxiin"),
        )
        dep_k = dep.sort(["airport", "STAND_mvt", "BLOCK_TIME_UTC_mvt"])
        arr_k = arr.sort(["airport", "STAND_mvt", "arr_inblock"])
        joined = dep_k.join_asof(
            arr_k,
            left_on="BLOCK_TIME_UTC_mvt",
            right_on="arr_inblock",
            by=["airport", "STAND_mvt"],
            strategy="backward",
        )
        joined = joined.with_columns(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("arr_inblock")).dt.total_seconds().alias("turnaround_s"),
            (pl.col("AIRCRAFT_TYPE_mvt") == pl.col("AIRCRAFT_TYPE_mvt_right")).alias("same_type"),
            (pl.col("CALLSIGN_flt") == pl.col("CALLSIGN_flt_right")).alias("same_cs"),
            (pl.col("FLIGHT_ID_mvt") == pl.col("FLIGHT_ID_mvt_right")).alias("same_fid"),
        )
        has = joined.filter(pl.col("arr_inblock").is_not_null())
        log(fh, f"DEP with a previous ARR at same stand (asof): {has.height:,}/{dep.height:,} ({100*has.height/dep.height:.1f}%)")
        log(fh, "same aircraft type among those", float(has["same_type"].mean() or 0))
        log(fh, "same callsign among those", float(has.filter(pl.col("CALLSIGN_flt").is_not_null())["same_cs"].mean() or 0))
        log(fh, "same FLIGHT_ID among those (should be ~0 if FLIGHT_ID is city-pair)", float(has["same_fid"].mean() or 0))
        ta = has["turnaround_s"].to_numpy().astype(float)
        ta = ta[np.isfinite(ta) & (ta >= 0)]
        log(fh, f"turnaround seconds n={ta.size:,} mean={ta.mean():.0f} med={np.median(ta):.0f} p10={np.quantile(ta,0.1):.0f} p90={np.quantile(ta,0.9):.0f}")
        # same-type only, 20min-8h plausible turnaround
        plaus = has.filter(
            pl.col("same_type")
            & pl.col("turnaround_s").is_between(20 * 60, 8 * 3600)
        )
        log(fh, f"plausible same-type turnaround 20min-8h: {plaus.height:,} ({100*plaus.height/dep.height:.1f}% of DEP)")
        y = plaus["y"].to_numpy()
        ti = plaus["taxiin"].to_numpy().astype(float)
        # corr taxi-out vs previous taxi-in
        m = np.isfinite(y) & np.isfinite(ti)
        if m.sum() > 100:
            log(fh, f"corr(this taxi-out, prev taxi-in) same stand plausible: {np.corrcoef(y[m], ti[m])[0,1]:.4f}")
        # corr with turnaround duration
        tt = plaus["turnaround_s"].to_numpy().astype(float)
        m2 = np.isfinite(y) & np.isfinite(tt)
        log(fh, f"corr(this taxi-out, turnaround_s): {np.corrcoef(y[m2], tt[m2])[0,1]:.4f}")

        # --- callsign chain at airport (different FLIGHT_ID) ---
        log(fh, "\n--- callsign: previous DEP with same CALLSIGN at same airport ---")
        dcs = dep.filter(pl.col("CALLSIGN_flt").is_not_null()).sort(["airport", "CALLSIGN_flt", "MVT_TIME_UTC_mvt"])
        dcs = dcs.with_columns(
            pl.col("y").shift(1).over(["airport", "CALLSIGN_flt"]).alias("prev_same_cs_y"),
            pl.col("FLIGHT_ID_mvt").shift(1).over(["airport", "CALLSIGN_flt"]).alias("prev_fid"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt").shift(1).over(["airport", "CALLSIGN_flt"]))
            .dt.total_seconds()
            .alias("dt_cs"),
        )
        has_cs = dcs.filter(pl.col("prev_same_cs_y").is_not_null())
        log(fh, f"DEP with previous same callsign at airport: {has_cs.height:,}")
        log(fh, "frac different FLIGHT_ID", float((has_cs["FLIGHT_ID_mvt"] != has_cs["prev_fid"]).mean()))
        yy = has_cs["y"].to_numpy()
        py = has_cs["prev_same_cs_y"].to_numpy()
        log(fh, f"corr(taxi, prev same-callsign taxi at airport): {np.corrcoef(yy, py)[0,1]:.4f}")
        # typically next day rotation: dt ~ 20h?
        dt = has_cs["dt_cs"].to_numpy()
        log(fh, f"dt same cs hours: med={np.median(dt)/3600:.2f} p10={np.quantile(dt,0.1)/3600:.2f} p90={np.quantile(dt,0.9)/3600:.2f}")

        # --- Jan+Jul diagnostic: does prev taxi-in / turnaround predict residual of a simple airport mean? ---
        log(fh, "\n--- leak check: previous taxi-in uses ARR TAXITIME which IS in ranking ---")
        log(fh, "ARR taxi-in is ranking-safe. Turnaround time uses ARR BLOCK (in-block) which is ranking-safe.")
        log(fh, "DEP BLOCK is NOT available at test. We used BLOCK as asof key — THAT IS LEAKAGE for ranking DEP.")
        log(fh, "Safe asof key alternatives: SCHED_TIME or AOBT_3 (matched only).")

        # rebuild asof on AOBT for matched, SCHED for all
        dep2 = (
            df.filter(pl.col("PHASE_mvt") == "DEP")
            .select(
                "MVT_ID_mvt",
                "airport",
                "STAND_mvt",
                "AIRCRAFT_TYPE_mvt",
                "SCHED_TIME_UTC_mvt",
                "AOBT_3_flt",
                "y",
            )
            .with_columns(pl.col("SCHED_TIME_UTC_mvt").dt.month().alias("month"))
        )
        arr2 = (
            df.filter(pl.col("PHASE_mvt") == "ARR")
            .select(
                "airport",
                "STAND_mvt",
                "AIRCRAFT_TYPE_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("arr_inblock"),
                pl.col("y").alias("taxiin"),
            )
            .sort(["airport", "STAND_mvt", "arr_inblock"])
        )
        dep_s = dep2.sort(["airport", "STAND_mvt", "SCHED_TIME_UTC_mvt"])
        jsafe = dep_s.join_asof(
            arr2,
            left_on="SCHED_TIME_UTC_mvt",
            right_on="arr_inblock",
            by=["airport", "STAND_mvt"],
            strategy="backward",
        )
        jsafe = jsafe.with_columns(
            (pl.col("SCHED_TIME_UTC_mvt") - pl.col("arr_inblock")).dt.total_seconds().alias("sched_minus_inblock"),
            (pl.col("AIRCRAFT_TYPE_mvt") == pl.col("AIRCRAFT_TYPE_mvt_right")).alias("same_type"),
        )
        # Jan+Jul corr
        va = jsafe.filter(pl.col("month").is_in([1, 7]))
        for lab, sub in [
            ("any prev ARR at stand", va.filter(pl.col("taxiin").is_not_null())),
            (
                "same type, 20m-8h by SCHED-inblock",
                va.filter(pl.col("same_type") & pl.col("sched_minus_inblock").is_between(20 * 60, 8 * 3600)),
            ),
        ]:
            if sub.height < 100:
                log(fh, lab, "too few", sub.height)
                continue
            yv = sub["y"].to_numpy()
            tv = sub["taxiin"].to_numpy().astype(float)
            m = np.isfinite(yv) & np.isfinite(tv)
            log(fh, f"Jan+Jul {lab} n={m.sum():,} corr taxi vs prev taxiin={np.corrcoef(yv[m], tv[m])[0,1]:.4f}")

        save_result("E7", {"note": "see E7.txt"})
    print("WROTE", out)


if __name__ == "__main__":
    main()
