"""E4 traffic/congestion and E5 queue dynamics, on top of best linear clocks+geometry."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AIRPORTS,
    DATA,
    DEP_COLS,
    TRAIN_FILES,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
    load_dep,
    metrics_block,
    ols_predict,
    save_result,
    split_by_months,
)
from run_e2b_e3 import attach_geometry, evaluate, geometry_tables, per_airport_ols


def load_arr() -> pl.DataFrame:
    cols = ["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]
    return (
        pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .with_columns(pl.col("ADES_mvt").alias("airport"))
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )


def add_traffic(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    windows = [5, 15, 30, 60]
    out = dep
    for w in windows:
        rolled = dep.rolling(index_column="MVT_TIME_UTC_mvt", period=f"{w}m", group_by="airport").agg(
            pl.len().alias(f"_d{w}")
        )
        out = out.with_columns((rolled[f"_d{w}"] - 1).alias(f"dep_{w}m"))
        rwy = dep.sort(["airport", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
        rr = rwy.rolling(
            index_column="MVT_TIME_UTC_mvt", period=f"{w}m", group_by=["airport", "RUNWAY_mvt"]
        ).agg(pl.len().alias("_r"))
        rwy = rwy.with_columns((rr["_r"] - 1).alias(f"dep_rwy_{w}m"))
        out = out.join(rwy.select("MVT_ID_mvt", f"dep_rwy_{w}m"), on="MVT_ID_mvt", how="left")

    arr_c = (
        arr.select("airport", pl.col("MVT_TIME_UTC_mvt").alias("arr_t"))
        .sort(["airport", "arr_t"])
        .with_columns(pl.int_range(1, pl.len() + 1).over("airport").alias("cum_arr"))
    )
    keys = out.select("MVT_ID_mvt", "airport", "MVT_TIME_UTC_mvt").sort(["airport", "MVT_TIME_UTC_mvt"])
    now = keys.join_asof(arr_c, left_on="MVT_TIME_UTC_mvt", right_on="arr_t", by="airport", strategy="backward")
    now = now.rename({"cum_arr": "cum_arr_now"})
    for w in windows:
        left = keys.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=w)).alias("t0"))
        jw = left.join_asof(arr_c, left_on="t0", right_on="arr_t", by="airport", strategy="backward").select(
            "MVT_ID_mvt", pl.col("cum_arr").alias(f"cum_arr_{w}")
        )
        now = now.join(jw, on="MVT_ID_mvt", how="left")
        now = now.with_columns(
            (pl.col("cum_arr_now").fill_null(0) - pl.col(f"cum_arr_{w}").fill_null(0)).alias(f"arr_{w}m")
        )
    out = out.join(now.select(["MVT_ID_mvt"] + [f"arr_{w}m" for w in windows]), on="MVT_ID_mvt", how="left")
    out = out.with_columns(
        (pl.col("dep_15m") + pl.col("arr_15m")).alias("mov_15m"),
        (pl.col("dep_60m") + pl.col("arr_60m")).alias("mov_60m"),
        (pl.col("dep_15m").cast(pl.Float64) / pl.col("dep_60m").clip(lower_bound=1)).alias("dep_ratio_15_60"),
        (pl.col("dep_5m").cast(pl.Float64) - pl.col("dep_15m") / 3.0).alias("dep_accel"),
        (pl.col("arr_15m").cast(pl.Float64) / (pl.col("dep_15m") + 1)).alias("arr_dep_ratio_15"),
        (pl.col("dep_15m") - pl.col("dep_5m")).alias("dep_5_15"),
    )
    return out


def add_queue(dep: pl.DataFrame) -> pl.DataFrame:
    """Vectorized queue at takeoff from AOBT/MVT overlap. Ranking-safe."""
    d = dep.filter(pl.col("AOBT_3_flt").is_not_null())
    n = dep.height
    queue = np.full(n, np.nan)
    q_rwy = np.full(n, np.nan)
    push5 = np.full(n, np.nan)
    push15 = np.full(n, np.nan)
    ids_all = dep["MVT_ID_mvt"].to_numpy()
    id_to_i = {int(i) if False else i: k for k, i in enumerate(ids_all)}
    # use row index alignment via join later
    q_ids = d["MVT_ID_mvt"].to_numpy()
    q_ap = d["airport"].to_numpy()
    q_mv = d["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    q_ao = d["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    q_rw = d["RUNWAY_mvt"].to_numpy()
    qv = np.full(d.height, np.nan)
    qrv = np.full(d.height, np.nan)
    p5 = np.full(d.height, np.nan)
    p15 = np.full(d.height, np.nan)

    for ap in AIRPORTS:
        idx = np.where(q_ap == ap)[0]
        if idx.size == 0:
            continue
        mv = q_mv[idx]
        ao = q_ao[idx]
        rw = q_rw[idx]
        ao_s = np.sort(ao)
        mv_s = np.sort(mv)
        n_started = np.searchsorted(ao_s, mv, side="left")
        n_taken = np.searchsorted(mv_s, mv, side="left")
        qv[idx] = np.maximum(n_started - n_taken - 1, 0)
        for wns, dest in [(5 * 60 * 10**9, p5), (15 * 60 * 10**9, p15)]:
            hi = np.searchsorted(ao_s, mv, side="right")
            lo = np.searchsorted(ao_s, mv - wns, side="right")
            dest[idx] = np.maximum(hi - lo, 0)
        # same-runway queue
        for r in np.unique(rw):
            ridx = idx[rw == r]
            ao_r = np.sort(q_ao[ridx])
            mv_r = np.sort(q_mv[ridx])
            mv_i = q_mv[ridx]
            ns = np.searchsorted(ao_r, mv_i, side="left")
            nt = np.searchsorted(mv_r, mv_i, side="left")
            qrv[ridx] = np.maximum(ns - nt - 1, 0)

    extra = pl.DataFrame(
        {
            "MVT_ID_mvt": q_ids,
            "queue": qv,
            "queue_rwy": qrv,
            "push_5m": p5,
            "push_15m": p15,
        }
    )
    out = dep.join(extra, on="MVT_ID_mvt", how="left")
    # queue growth: queue minus lag of queue at airport (approx 5/10/15 min via shift of previous flights)
    out = out.sort(["airport", "MVT_TIME_UTC_mvt"]).with_columns(
        (pl.col("queue") - pl.col("queue").shift(5).over("airport")).alias("queue_growth_5fl"),
        (pl.col("queue") - pl.col("queue").shift(10).over("airport")).alias("queue_growth_10fl"),
        (pl.col("queue") / (pl.col("dep_15m") + 1)).alias("queue_per_dep15"),
        (pl.col("queue_rwy") / (pl.col("dep_rwy_15m") + 1)).alias("queue_rwy_per_dep15"),
    )
    return out


def add_relative_traffic(train: pl.DataFrame, val: pl.DataFrame) -> pl.DataFrame:
    hist = train.group_by(["airport", "hour"]).agg(pl.col("dep_15m").mean().alias("dep15_hour_mean"))
    val = val.join(hist, on=["airport", "hour"], how="left")
    return val.with_columns(
        (pl.col("dep_15m").cast(pl.Float64) / pl.col("dep15_hour_mean").clip(lower_bound=1)).alias("dep15_vs_hour")
    )


def main():
    out_path = Path(__file__).resolve().parent / "results" / "E4_E5.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        print("loading dep+arr...", flush=True)
        dep = add_causal_rolling(load_dep())
        arr = load_arr()
        print("traffic...", flush=True)
        dep = add_traffic(dep, arr)
        print("queue...", flush=True)
        dep = add_queue(dep)
        print(f"ready {dep.height:,}", flush=True)

        payload = {"E4": {}, "E5": {}}
        traffic_cols = [
            "dep_5m",
            "dep_15m",
            "dep_30m",
            "dep_60m",
            "dep_rwy_15m",
            "dep_rwy_60m",
            "arr_15m",
            "arr_30m",
            "arr_60m",
            "mov_15m",
            "dep_ratio_15_60",
            "dep_accel",
            "arr_dep_ratio_15",
            "dep15_vs_hour",
        ]
        queue_cols = [
            "queue",
            "queue_rwy",
            "push_5m",
            "push_15m",
            "queue_growth_5fl",
            "queue_growth_10fl",
            "queue_per_dep15",
        ]

        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr, va = split_by_months(dep, months)
            tabs = geometry_tables(tr)
            tr = attach_geometry(tr, tabs, 30)
            va = attach_geometry(va, tabs, 30)
            va = add_relative_traffic(tr, va)
            tr = add_relative_traffic(tr, tr)
            y = va["y"].to_numpy()
            um = va["unmatched"].to_numpy()
            ap = va["airport"].to_numpy()
            fb = airport_mean_fallback(tr, va)
            geo = va["geo_mean"].to_numpy().astype(float)

            print(f"\n========== {split_name} ==========", flush=True)
            fh.write(f"\n========== {split_name} ==========\n")

            # baseline: per-airport aobt+eobt+geo
            base_ap, base_g, coef, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
            base = fill_with_fallback(fill_with_fallback(base_ap, geo), fb)
            payload.setdefault(split_name, {})
            payload[split_name]["base_airport_aobt_eobt_geo"] = evaluate(
                "BASE per-airport aobt+eobt+geo", y, base, um, ap, fh
            )

            print("\n--- E4 add one traffic feature to global aobt+eobt+geo ---", flush=True)
            fh.write("\n--- E4 ---\n")
            block4 = {}
            core = ["mvt_aobt", "aobt_eobt", "geo_mean"]
            for extra in traffic_cols:
                cols = core + [extra]
                trc = tr
                for c in cols:
                    trc = trc.filter(pl.col(c).is_not_null())
                if trc.height < 1000:
                    continue
                pred, coef = ols_predict(
                    np.column_stack([trc[c].to_numpy().astype(float) for c in cols]),
                    trc["y"].to_numpy(),
                    np.column_stack([va[c].to_numpy().astype(float) for c in cols]),
                )
                m = evaluate(f"+{extra}", y, fill_with_fallback(fill_with_fallback(pred, geo), fb), um, ap, fh)
                m["coef_last"] = float(coef[-1])
                block4[extra] = m

            # best combo of traffic
            combo = core + ["dep_rwy_15m", "dep_15m", "arr_15m", "dep_ratio_15_60", "dep15_vs_hour"]
            trc = tr
            for c in combo:
                trc = trc.filter(pl.col(c).is_not_null())
            pred, coef = ols_predict(
                np.column_stack([trc[c].to_numpy().astype(float) for c in combo]),
                trc["y"].to_numpy(),
                np.column_stack([va[c].to_numpy().astype(float) for c in combo]),
            )
            block4["traffic_combo"] = evaluate(
                "aobt+eobt+geo+traffic_combo", y, fill_with_fallback(fill_with_fallback(pred, geo), fb), um, ap, fh
            )
            block4["traffic_combo"]["coef"] = [float(c) for c in coef]
            print(f"    combo coef={coef}", flush=True)
            fh.write(f"    combo coef={coef}\n")

            p_ap, _, _, _ = per_airport_ols(tr, va, combo)
            block4["airport_traffic_combo"] = evaluate(
                "per-airport + traffic_combo", y, fill_with_fallback(fill_with_fallback(p_ap, geo), fb), um, ap, fh
            )
            payload["E4"][split_name] = block4

            print("\n--- E5 add one queue feature ---", flush=True)
            fh.write("\n--- E5 ---\n")
            block5 = {}
            for extra in queue_cols:
                cols = core + [extra]
                trc = tr
                for c in cols:
                    trc = trc.filter(pl.col(c).is_not_null())
                if trc.height < 1000:
                    continue
                pred, coef = ols_predict(
                    np.column_stack([trc[c].to_numpy().astype(float) for c in cols]),
                    trc["y"].to_numpy(),
                    np.column_stack([va[c].to_numpy().astype(float) for c in cols]),
                )
                m = evaluate(f"+{extra}", y, fill_with_fallback(fill_with_fallback(pred, geo), fb), um, ap, fh)
                m["coef_last"] = float(coef[-1])
                block5[extra] = m

            combo5 = core + ["queue", "queue_rwy", "push_15m"]
            trc = tr
            for c in combo5:
                trc = trc.filter(pl.col(c).is_not_null())
            pred, coef = ols_predict(
                np.column_stack([trc[c].to_numpy().astype(float) for c in combo5]),
                trc["y"].to_numpy(),
                np.column_stack([va[c].to_numpy().astype(float) for c in combo5]),
            )
            block5["queue_combo"] = evaluate(
                "aobt+eobt+geo+queue_combo", y, fill_with_fallback(fill_with_fallback(pred, geo), fb), um, ap, fh
            )
            block5["queue_combo"]["coef"] = [float(c) for c in coef]
            print(f"    queue coef={coef}", flush=True)
            fh.write(f"    queue coef={coef}\n")

            both = core + ["dep_rwy_15m", "queue", "queue_rwy"]
            p_ap, _, coef, _ = per_airport_ols(tr, va, both)
            block5["airport_geo_traffic_queue"] = evaluate(
                "per-airport aobt+eobt+geo+rwy15+queue",
                y,
                fill_with_fallback(fill_with_fallback(p_ap, geo), fb),
                um,
                ap,
                fh,
            )
            payload["E5"][split_name] = block5

        save_result("E4_E5", payload)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
