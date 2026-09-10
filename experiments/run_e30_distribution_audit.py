"""E30 — distribution / schema / leaderboard-gap audit (C31, C32).

Question: why can the leaderboard be ~246 while our internal is ~368 (LB 317)?
Uses prediction-time fields only (no ranking targets). Compares schema,
availability, and feature distributions across 2025 training, Dec 2025,
Jan/Jul 2025 holdout, and 2026 ranking.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, load_dep  # noqa: E402
from run_submitting_check import RANK_PATH  # noqa: E402

RES = HERE / "results" / "E30"
RES.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def psi(a, b, bins=10):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(ha.sum(), 1) + 1e-6
    pb = hb / max(hb.sum(), 1) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main():
    log("E30 audit: schema + availability...")
    tr_schema = pl.scan_parquet(TRAIN_FILES[0]).collect_schema()
    rk_schema = pl.scan_parquet(RANK_PATH).collect_schema()
    schema = {
        "same_columns": list(tr_schema.names()) == list(rk_schema.names()),
        "in_training_not_ranking": sorted(set(tr_schema.names()) - set(rk_schema.names())),
        "in_ranking_not_training": sorted(set(rk_schema.names()) - set(tr_schema.names())),
    }
    rank = pl.scan_parquet(RANK_PATH).filter(pl.col("PHASE_mvt") == "DEP").select(
        ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
         "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
         "ARVT_3_flt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "ADES_FILED_flt", "FLIGHT_TYPE_flt"]
    ).collect()
    rank = rank.with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ms"),
    )
    n = rank.height
    avail = {c: {"null": int(rank[c].null_count()), "coverage": float(1 - rank[c].null_count() / n)} for c in rank.columns}
    unmatched = rank.filter(pl.col("AOBT_3_flt").is_null())
    log(f"  ranking DEP {n:,}  unmatched(all flt null) {unmatched.height:,} ({100*unmatched.height/n:.2f}%)")
    avail["_unmatched_n"] = int(unmatched.height)

    # forbidden-field check: ARVT_3 vs MVT (arrival at destination should be AFTER takeoff)
    m = rank.filter(pl.col("ARVT_3_flt").is_not_null() & pl.col("MVT_TIME_UTC_mvt").is_not_null())
    d = (m["ARVT_3_flt"] - m["MVT_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
    avail["_ARVT_after_MVT_frac"] = float(np.mean(d > 0))

    # ---- distribution comparison on movements available everywhere ----
    dep = load_dep().with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ms"))
    splits = {
        "train2025": dep,
        "dec2025": dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 12),
        "jan2025": dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1),
        "jul2025": dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7),
    }
    dist = {}
    for name, df in splits.items():
        dist[name] = {
            "n": int(df.height),
            "ms_median": float(df["ms"].median()),
            "ms_p90": float(df["ms"].quantile(0.9)),
            "unmatched_frac": float(df["unmatched"].mean()),
        }
    dist["ranking2026"] = {
        "n": int(n),
        "ms_median": float(rank["ms"].median()),
        "ms_p90": float(rank["ms"].quantile(0.9)),
        "unmatched_frac": float(unmatched.height / n),
    }

    # LIRF unmatched MVT-SCHED: ranking vs 2025 holdout (the dominant error driver)
    lirf_cmp = {}
    lirf_cmp["ranking2026"] = {
        "n": int(unmatched.filter(pl.col("ADEP_mvt") == "LIRF").height),
        "ms_median": float(unmatched.filter(pl.col("ADEP_mvt") == "LIRF")["ms"].median()),
        "ms_p90": float(unmatched.filter(pl.col("ADEP_mvt") == "LIRF")["ms"].quantile(0.9)),
        "ms_max": float(unmatched.filter(pl.col("ADEP_mvt") == "LIRF")["ms"].max()),
    }
    for lab, mth in [("janjul2025", [1, 7]), ("dec2025", [12])]:
        v = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(mth))
        l = v.filter(pl.col("unmatched") & (pl.col("airport") == "LIRF"))
        lirf_cmp[lab] = {"n": int(l.height), "ms_median": float(l["ms"].median()),
                         "ms_p90": float(l["ms"].quantile(0.9)), "ms_max": float(l["ms"].max()),
                         "y_median": float(l["y"].median()), "y_p90": float(l["y"].quantile(0.9))}

    # PSI of key numeric columns: ranking vs train
    psi_out = {}
    for col in ["ms"]:
        psi_out[col] = psi(rank[col].to_numpy().astype(float), dep[col].to_numpy().astype(float))
    # categorical overlap (airport, runway)
    def cat_overlap(col_r, col_t):
        r = set(col_r.drop_nulls().unique().to_list())
        t = set(col_t.drop_nulls().unique().to_list())
        return {"ranking_categories": len(r), "train_categories": len(t),
                "ranking_unseen_in_train": sorted(str(x) for x in (r - t))[:20],
                "unseen_rate": float(np.mean([x not in t for x in col_r.drop_nulls().to_list()]))}
    cats = {"airport": cat_overlap(rank["ADEP_mvt"], dep["airport"]),
            "runway": cat_overlap(rank["RUNWAY_mvt"], dep["RUNWAY_mvt"]),
            "stand": cat_overlap(rank["STAND_mvt"], dep["STAND_mvt"]),
            "ac_type": cat_overlap(rank["AIRCRAFT_TYPE_mvt"], dep["AIRCRAFT_TYPE_mvt"]),
            "flt_prefix": cat_overlap(rank["FLIGHT_mvt"].fill_null("").str.slice(0, 3), dep["FLIGHT_mvt"].fill_null("").str.slice(0, 3))}

    payload = {"schema": schema, "ranking_availability": avail, "distributions": dist,
               "lirf_unmatched_ms": lirf_cmp, "psi_ms": psi_out, "categorical_overlap": cats}
    (RES / "E30_distribution_audit.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    log("== distributions (MVT-SCHED) ==")
    for k, v in dist.items():
        log(f"  {k:12s} n={v['n']:>9,} ms_med={v['ms_median']:8.0f} ms_p90={v['ms_p90']:9.0f} unmatched={100*v['unmatched_frac']:.2f}%")
    log("== LIRF unmatched MVT-SCHED ==")
    for k, v in lirf_cmp.items():
        log(f"  {k:12s} n={v['n']:>4} ms_med={v['ms_median']:8.0f} ms_p90={v['ms_p90']:9.0f}")
    log(f"PSI(ms ranking vs train) = {psi_out['ms']:.4f}")
    log(f"unseen rate: runway {cats['runway']['unseen_rate']:.3f} stand {cats['stand']['unseen_rate']:.3f} ac_type {cats['ac_type']['unseen_rate']:.3f}")
    log(f"ARVT after MVT frac = {avail['_ARVT_after_MVT_frac']:.4f}")
    log("WROTE " + str(RES / "E30_distribution_audit.json"))


if __name__ == "__main__":
    main()
