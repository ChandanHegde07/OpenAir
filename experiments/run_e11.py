"""E11: unmatched / LIRF investigation. Understand first; do not drop rows or jump to a specialist."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AIRPORTS,
    DATA,
    load_dep,
    metrics_block,
    save_result,
    split_by_months,
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
)


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def dist(fh, name, s):
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        log(fh, f"  [{name}] empty")
        return
    qs = np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    log(
        fh,
        f"  [{name}] n={s.size:,} mean={s.mean():.1f} med={np.median(s):.1f} std={s.std():.1f} "
        f"p5={qs[0]:.0f} p25={qs[1]:.0f} p50={qs[2]:.0f} p75={qs[3]:.0f} p90={qs[4]:.0f} "
        f"p95={qs[5]:.0f} p99={qs[6]:.0f} max={s.max():.0f} "
        f">30m={(s>1800).mean()*100:.2f}% >1h={(s>3600).mean()*100:.2f}% >2h={(s>7200).mean()*100:.2f}%",
    )


def main():
    out = Path(__file__).resolve().parent / "results" / "E11.txt"
    with open(out, "w", encoding="utf-8") as fh:
        log(fh, "========== E11 UNMATCHED / LIRF ==========")
        dep = load_dep()
        log(fh, f"DEP n={dep.height:,} unmatched={int(dep['unmatched'].sum()):,}")

        # 1. missingness: are NM fields jointly missing?
        log(fh, "\n--- missingness co-occurrence (DEP) ---")
        nm_cols = [
            "AOBT_3_flt",
            "EOBT_1_flt",
            "IOBT_flt",
            "LOBT_flt",
            "FLIGHT_ID_mvt",
            "CALLSIGN_flt" if "CALLSIGN_flt" in dep.columns else "MARKET_SEGMENT_flt",
            "WK_TBL_CAT_flt",
            "AIRCRAFT_OPERATOR_flt",
        ]
        # CALLSIGN not in load_dep. use MARKET_SEGMENT
        for c in ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "FLIGHT_ID_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "ARVT_1_flt"]:
            nnull = dep[c].is_null().sum()
            both = dep.filter(pl.col("unmatched") & pl.col(c).is_null()).height
            log(fh, f"  {c:28s} null={nnull:,}  null_and_unmatched={both:,}  unmatched_but_present={dep.filter(pl.col('unmatched') & pl.col(c).is_not_null()).height:,}")

        # FLIGHT_ID null but AOBT present?
        log(
            fh,
            "FLIGHT_ID null, AOBT present",
            dep.filter(pl.col("FLIGHT_ID_mvt").is_null() & pl.col("AOBT_3_flt").is_not_null()).height,
        )
        log(
            fh,
            "FLIGHT_ID present, AOBT null",
            dep.filter(pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("AOBT_3_flt").is_null()).height,
        )

        # 2. unmatched vs matched by airport
        log(fh, "\n--- unmatched vs matched by airport ---")
        log(fh, f"{'AP':6s} {'n_u':>7s} {'pct':>6s} {'u_mean':>8s} {'u_med':>8s} {'u_std':>8s} {'m_mean':>8s} {'m_med':>8s} {'u>1h':>6s} {'m>1h':>6s}")
        for ap in AIRPORTS:
            u = dep.filter((pl.col("airport") == ap) & pl.col("unmatched"))["y"].to_numpy()
            m = dep.filter((pl.col("airport") == ap) & (~pl.col("unmatched")))["y"].to_numpy()
            log(
                fh,
                f"{ap:6s} {len(u):7d} {100*len(u)/(len(u)+len(m)):5.2f}% "
                f"{u.mean():8.1f} {np.median(u):8.1f} {u.std():8.1f} "
                f"{m.mean():8.1f} {np.median(m):8.1f} "
                f"{100*(u>3600).mean():5.1f}% {100*(m>3600).mean():5.2f}%",
            )

        dist(fh, "all unmatched y", dep.filter(pl.col("unmatched"))["y"].to_numpy())
        dist(fh, "all matched y", dep.filter(~pl.col("unmatched"))["y"].to_numpy())
        dist(fh, "LIRF unmatched y", dep.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))["y"].to_numpy())
        dist(fh, "LIRF matched y", dep.filter((pl.col("airport") == "LIRF") & (~pl.col("unmatched")))["y"].to_numpy())

        # 3. LIRF unmatched vs matched: clocks we DO have (mvt, block, sched)
        log(fh, "\n--- LIRF unmatched vs matched: airport clocks ---")
        lirf = dep.filter(pl.col("airport") == "LIRF").with_columns(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("block_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("mvt_sched"),
            pl.col("BLOCK_TIME_UTC_mvt").dt.date().alias("block_date"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("mvt_date"),
            pl.col("SCHED_TIME_UTC_mvt").dt.date().alias("sched_date"),
            (pl.col("MVT_TIME_UTC_mvt").dt.date() != pl.col("BLOCK_TIME_UTC_mvt").dt.date()).alias("cross_day"),
            ((pl.col("y") > 21600).alias("gt6h")),
        )
        for flag, lab in [(True, "UNMATCHED"), (False, "MATCHED")]:
            sub = lirf.filter(pl.col("unmatched") == flag)
            log(fh, f"\n LIRF {lab} n={sub.height:,}")
            dist(fh, "y", sub["y"].to_numpy())
            dist(fh, "block-sched", sub["block_sched"].to_numpy())
            dist(fh, "mvt-sched", sub["mvt_sched"].to_numpy())
            log(fh, "  cross_day BLOCK vs MVT", int(sub["cross_day"].sum()), f"({100*sub['cross_day'].mean():.1f}%)")
            log(fh, "  y>6h", int((sub["y"] > 21600).sum()), "y>12h", int((sub["y"] > 43200).sum()), "y>24h", int((sub["y"] > 86400).sum()))

        # 4. hour / month / dow
        log(fh, "\n--- LIRF unmatched hour (UTC) ---")
        log(
            fh,
            lirf.filter(pl.col("unmatched"))
            .group_by("hour")
            .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), pl.col("y").mean().alias("mean"))
            .sort("hour"),
        )
        log(fh, "\n--- LIRF unmatched month ---")
        log(
            fh,
            lirf.filter(pl.col("unmatched"))
            .group_by("month")
            .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), (pl.col("y") > 3600).sum().alias("gt1h"))
            .sort("month"),
        )
        log(fh, "matched LIRF by month >1h", lirf.filter(~pl.col("unmatched")).group_by("month").agg(pl.len().alias("n"), (pl.col("y") > 3600).sum().alias("gt1h")).sort("month"))

        # 5. runway / stand / type
        log(fh, "\n--- LIRF unmatched runway ---")
        log(
            fh,
            lirf.group_by(["unmatched", "RUNWAY_mvt"])
            .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), pl.col("y").mean().alias("mean"))
            .sort(["unmatched", "n"], descending=[False, True]),
        )
        log(fh, "\n--- LIRF unmatched top stands ---")
        log(
            fh,
            lirf.filter(pl.col("unmatched"))
            .group_by("STAND_mvt")
            .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), pl.col("y").mean().alias("mean"))
            .sort("n", descending=True)
            .head(20),
        )
        log(fh, "\n--- LIRF unmatched aircraft type ---")
        log(
            fh,
            lirf.filter(pl.col("unmatched"))
            .group_by("AIRCRAFT_TYPE_mvt")
            .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), pl.col("y").mean().alias("mean"))
            .sort("n", descending=True)
            .head(20),
        )
        log(fh, "LIRF matched top types", lirf.filter(~pl.col("unmatched")).group_by("AIRCRAFT_TYPE_mvt").agg(pl.len().alias("n")).sort("n", descending=True).head(10))

        # 6. flight number patterns
        log(fh, "\n--- LIRF unmatched FLIGHT_mvt samples / patterns ---")
        u = lirf.filter(pl.col("unmatched"))
        log(fh, "null FLIGHT_mvt", u.filter(pl.col("FLIGHT_mvt").is_null()).height)
        log(fh, "unique FLIGHT_mvt", u.select(pl.col("FLIGHT_mvt").n_unique()).item())
        # prefix 3 letters vs 4
        u2 = u.with_columns(
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("p3"),
            pl.col("FLIGHT_mvt").str.slice(0, 2).alias("p2"),
            pl.col("FLIGHT_mvt").str.contains(r"^[A-Z]{2,3}\d").alias("looks_airline"),
        )
        log(fh, "top 3-char prefixes unmatched", u2.group_by("p3").agg(pl.len().alias("n"), pl.col("y").median().alias("med")).sort("n", descending=True).head(20))
        log(fh, "matched top prefixes", lirf.filter(~pl.col("unmatched")).with_columns(pl.col("FLIGHT_mvt").str.slice(0, 3).alias("p3")).group_by("p3").agg(pl.len().alias("n")).sort("n", descending=True).head(10))

        # 7. destination mix
        log(fh, "\n--- LIRF unmatched ADES ---")
        log(fh, u.group_by("ADES_mvt").agg(pl.len().alias("n"), pl.col("y").median().alias("med")).sort("n", descending=True).head(15))
        log(fh, "matched ADES top", lirf.filter(~pl.col("unmatched")).group_by("ADES_mvt").agg(pl.len().alias("n")).sort("n", descending=True).head(10))

        # 8. does same FLIGHT_mvt appear as MATCHED at LIRF on other days?
        um_fl = u.select("FLIGHT_mvt").unique()
        n_also_matched = (
            lirf.filter(~pl.col("unmatched"))
            .join(um_fl, on="FLIGHT_mvt", how="inner")
            .select(pl.col("FLIGHT_mvt").n_unique())
            .item()
        )
        log(fh, f"unmatched LIRF flight numbers that also appear as matched LIRF: {n_also_matched} / {um_fl.height}")

        # 9. previous movement at same stand: is BLOCK equal to a previous BLOCK or ARR in-block?
        # compare time since previous DEP at same stand
        log(fh, "\n--- LIRF stand occupancy / previous DEP gap ---")
        srt = lirf.sort(["STAND_mvt", "MVT_TIME_UTC_mvt"]).with_columns(
            pl.col("MVT_TIME_UTC_mvt").shift(1).over("STAND_mvt").alias("prev_mvt"),
            pl.col("BLOCK_TIME_UTC_mvt").shift(1).over("STAND_mvt").alias("prev_block"),
            pl.col("unmatched").shift(1).over("STAND_mvt").alias("prev_unmatched"),
            pl.col("y").shift(1).over("STAND_mvt").alias("prev_y"),
        )
        srt = srt.with_columns(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("prev_mvt")).dt.total_seconds().alias("block_minus_prev_mvt"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("prev_block")).dt.total_seconds().alias("block_minus_prev_block"),
        )
        uu = srt.filter(pl.col("unmatched"))
        dist(fh, "unmatched BLOCK - prev DEP MVT (same stand)", uu["block_minus_prev_mvt"].to_numpy())
        dist(fh, "matched BLOCK - prev DEP MVT (same stand)", srt.filter(~pl.col("unmatched"))["block_minus_prev_mvt"].to_numpy())
        # fraction where BLOCK is within 60s of previous DEP BLOCK (possible copy)
        near = uu.filter(pl.col("block_minus_prev_block").abs() <= 60).height
        log(fh, f"unmatched BLOCK within 60s of previous DEP BLOCK same stand: {near}/{uu.height}")
        near2 = uu.filter(pl.col("block_minus_prev_mvt").abs() <= 60).height
        log(fh, f"unmatched BLOCK within 60s of previous DEP MVT same stand: {near2}/{uu.height}")

        # 10. sample extreme unmatched LIRF
        log(fh, "\n--- sample LIRF unmatched y>2h ---")
        log(
            fh,
            u.filter(pl.col("y") > 7200)
            .select(
                "FLIGHT_mvt",
                "ADES_mvt",
                "AIRCRAFT_TYPE_mvt",
                "RUNWAY_mvt",
                "STAND_mvt",
                "BLOCK_TIME_UTC_mvt",
                "MVT_TIME_UTC_mvt",
                "SCHED_TIME_UTC_mvt",
                "y",
                "hour",
                "month",
            )
            .sort("y", descending=True)
            .head(15),
        )
        log(fh, "\n--- sample LIRF unmatched y<20min ---")
        log(
            fh,
            u.filter(pl.col("y") < 1200)
            .select("FLIGHT_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "y", "hour")
            .head(10),
        )

        # 11. other airports unmatched: are they similar to LIRF or mild?
        log(fh, "\n--- unmatched at other airports: are they also extreme? ---")
        for ap in AIRPORTS:
            if ap == "LIRF":
                continue
            sub = dep.filter((pl.col("airport") == ap) & pl.col("unmatched"))
            dist(fh, f"{ap} unmatched", sub["y"].to_numpy())

        # Ranking/submitting files are ignored in this research phase.
        # Unmatched diagnostics below use TRAINING holdouts only.

        # 13. diagnostic ceiling: if we used train unmatched-LIRF median as a constant for unmatched LIRF, and airport unmatched median elsewhere
        log(fh, "\n--- DIAGNOSTIC (not a specialist model): unmatched constants from TRAIN applied to VAL ---")
        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr, va = split_by_months(dep, months)
            y = va["y"].to_numpy()
            um = va["unmatched"].to_numpy()
            ap = va["airport"].to_numpy()
            # baseline: airport mean for everyone (no clocks)
            fb = airport_mean_fallback(tr, va)
            # oracle-ish: unmatched get train unmatched median by airport; matched keep airport mean (NOT the best matched model)
            med_u = (
                tr.filter(pl.col("unmatched"))
                .group_by("airport")
                .agg(pl.col("y").median().alias("u_med"), pl.col("y").mean().alias("u_mean"))
            )
            j = va.join(med_u, on="airport", how="left")
            u_med = j["u_med"].to_numpy().astype(float)
            u_mean = j["u_mean"].to_numpy().astype(float)
            pred_med = np.where(um, np.where(np.isfinite(u_med), u_med, fb), fb)
            pred_mean = np.where(um, np.where(np.isfinite(u_mean), u_mean, fb), fb)
            m0 = metrics_block(y, fb, um, ap)
            m1 = metrics_block(y, pred_med, um, ap)
            m2 = metrics_block(y, pred_mean, um, ap)
            log(fh, f"\n{split_name} airport_mean all rows:     {fmt(m0)}")
            log(fh, f"{split_name} unmatched=train u_median: {fmt(m1)}")
            log(fh, f"{split_name} unmatched=train u_mean:   {fmt(m2)}")
            # LIRF-only constant vs others airport mean
            lirf_med = float(tr.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))["y"].median())
            lirf_p75 = float(tr.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))["y"].quantile(0.75))
            pred_lirf = fb.copy()
            pred_lirf[(ap == "LIRF") & um] = lirf_med
            m3 = metrics_block(y, pred_lirf, um, ap)
            log(fh, f"{split_name} only LIRF unmatched -> train LIRF u_med={lirf_med:.0f}: {fmt(m3)}")
            pred_lirf75 = fb.copy()
            pred_lirf75[(ap == "LIRF") & um] = lirf_p75
            m4 = metrics_block(y, pred_lirf75, um, ap)
            log(fh, f"{split_name} only LIRF unmatched -> train LIRF u_p75={lirf_p75:.0f}: {fmt(m4)}")

        # 14. SSE share of unmatched in val
        tr, va = split_by_months(dep, [1, 7])
        y = va["y"].to_numpy()
        um = va["unmatched"].to_numpy()
        mean = y.mean()
        sse = ((y - mean) ** 2).sum()
        log(fh, f"\nJan+Jul val unmatched share of rows {um.mean():.4f} share of mean-model SSE {((y[um]-mean)**2).sum()/sse:.4f}")
        lirf_u = um & (va["airport"].to_numpy() == "LIRF")
        log(fh, f"Jan+Jul val LIRF unmatched share of rows {lirf_u.mean():.4f} share of SSE {((y[lirf_u]-mean)**2).sum()/sse:.4f}")
        log(fh, f"n unmatched val={um.sum()} n LIRF unmatched val={lirf_u.sum()}")

        save_result(
            "E11",
            {
                "n_dep": dep.height,
                "n_unmatched": int(dep["unmatched"].sum()),
                "lirf_unmatched": int(dep.filter((pl.col("airport") == "LIRF") & pl.col("unmatched")).height),
            },
        )
    print("WROTE", out)


if __name__ == "__main__":
    main()
