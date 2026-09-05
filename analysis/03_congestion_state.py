"""Temporal density, multi-scale congestion, residual vs traffic, regimes, hard cases.

All rolling features use only information available in ranking:
- DEP: MVT_TIME, AOBT_3, SCHED, EOBT, LOBT, RUNWAY, STAND (NOT other flights' TAXITIME/BLOCK)
- ARR: full BLOCK, MVT, TAXITIME
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data")
OUT = Path("analysis/output")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_FILES = sorted(glob.glob(str(DATA / "training_*.parquet")))
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WINDOWS = [1, 5, 10, 15, 30, 60]


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    if np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def load_all():
    cols = [
        "MVT_ID_mvt",
        "FLIGHT_ID_mvt",
        "PHASE_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "RUNWAY_mvt",
        "STAND_mvt",
        "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt",
        "MARKET_SEGMENT_flt",
        "AIRCRAFT_OPERATOR_flt",
        "AOBT_3_flt",
        "EOBT_1_flt",
        "LOBT_flt",
        "ARVT_3_flt",
    ]
    return pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES]).collect()


def airport_of(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("airport")
    )


def density(fh, df: pl.DataFrame):
    log(fh, "=" * 80)
    log(fh, "TEMPORAL DENSITY")
    log(fh, "=" * 80)
    df = airport_of(df)
    # per airport per hour
    hourly = (
        df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour"))
        .group_by(["airport", "hour"])
        .agg(
            pl.len().alias("n"),
            (pl.col("PHASE_mvt") == "DEP").sum().alias("n_dep"),
            (pl.col("PHASE_mvt") == "ARR").sum().alias("n_arr"),
        )
    )
    log(fh, "\nhourly movement stats by airport:")
    log(
        fh,
        hourly.group_by("airport").agg(
            pl.len().alias("n_hours"),
            pl.col("n").mean().alias("mean_mov_per_h"),
            pl.col("n").median().alias("med_mov_per_h"),
            pl.col("n").quantile(0.9).alias("p90_mov_per_h"),
            pl.col("n").max().alias("max_mov_per_h"),
            pl.col("n_dep").mean().alias("mean_dep_h"),
            pl.col("n_arr").mean().alias("mean_arr_h"),
        ).sort("airport"),
    )
    daily = (
        df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1d").alias("day"))
        .group_by(["airport", "day"])
        .agg(pl.len().alias("n"), (pl.col("PHASE_mvt") == "DEP").sum().alias("n_dep"))
    )
    log(fh, "\ndaily movement stats by airport:")
    log(
        fh,
        daily.group_by("airport").agg(
            pl.len().alias("n_days"),
            pl.col("n").mean().alias("mean_mov_day"),
            pl.col("n_dep").mean().alias("mean_dep_day"),
            pl.col("n").min().alias("min_day"),
            pl.col("n").max().alias("max_day"),
        ).sort("airport"),
    )

    # inter-departure gaps
    log(fh, "\ninter-departure gaps (seconds) by airport:")
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").sort(["airport", "MVT_TIME_UTC_mvt"])
    dep = dep.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").diff().over("airport").dt.total_seconds()).alias("gap")
    )
    log(
        fh,
        dep.filter(pl.col("gap").is_not_null() & (pl.col("gap") >= 0) & (pl.col("gap") < 86400))
        .group_by("airport")
        .agg(
            pl.len().alias("n"),
            pl.col("gap").mean().alias("mean_gap"),
            pl.col("gap").median().alias("med_gap"),
            pl.col("gap").quantile(0.1).alias("p10"),
            pl.col("gap").quantile(0.25).alias("p25"),
        )
        .sort("airport"),
    )
    # resolution: fraction of gaps < 60s, < 120s
    gaps = dep.filter(pl.col("gap").is_not_null() & (pl.col("gap") >= 0) & (pl.col("gap") < 3600))["gap"].to_numpy()
    log(fh, f"dep gaps <1h: n={gaps.size:,} frac<30s={(gaps<30).mean():.3f} frac<60s={(gaps<60).mean():.3f} frac<120s={(gaps<120).mean():.3f}")


def add_rolling_counts(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """For each DEP, count DEP takeoffs and ARR landings in previous W minutes at same airport.

    Uses MVT_TIME as the reference (available in ranking). Excludes the current flight.
    """
    dep = airport_of(dep).sort(["airport", "MVT_TIME_UTC_mvt"])
    arr = airport_of(arr).sort(["airport", "MVT_TIME_UTC_mvt"])

    # polars rolling on dep takeoffs
    out = dep
    for w in WINDOWS:
        rolled = (
            dep.rolling(index_column="MVT_TIME_UTC_mvt", period=f"{w}m", group_by="airport")
            .agg(pl.len().alias(f"dep_prev_{w}m_incl"))
        )
        # rolling result aligns with original rows in polars 1.x when same sort
        out = out.with_columns((rolled[f"dep_prev_{w}m_incl"] - 1).alias(f"dep_prev_{w}m"))

        # same-runway takeoffs
        rwy = dep.sort(["airport", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
        rolled_r = (
            rwy.rolling(index_column="MVT_TIME_UTC_mvt", period=f"{w}m", group_by=["airport", "RUNWAY_mvt"])
            .agg(pl.len().alias(f"_r"))
        )
        rwy = rwy.with_columns((rolled_r["_r"] - 1).alias(f"dep_rwy_prev_{w}m"))
        out = out.join(
            rwy.select("MVT_ID_mvt", f"dep_rwy_prev_{w}m"),
            on="MVT_ID_mvt",
            how="left",
        )

    # arrival counts via join_asof per airport: for each dep, count arr with landing in (t-W, t]
    # Efficient: asof join of cumulative arrival count
    arr_c = arr.select("airport", pl.col("MVT_TIME_UTC_mvt").alias("arr_t")).sort(["airport", "arr_t"])
    arr_c = arr_c.with_columns(pl.int_range(1, pl.len() + 1).over("airport").alias("cum_arr"))
    # for each window, asof join at t and at t-W
    dep_keys = out.select("MVT_ID_mvt", "airport", "MVT_TIME_UTC_mvt").sort(["airport", "MVT_TIME_UTC_mvt"])
    joined_now = dep_keys.join_asof(
        arr_c, left_on="MVT_TIME_UTC_mvt", right_on="arr_t", by="airport", strategy="backward"
    ).rename({"cum_arr": "cum_arr_now", "arr_t": "arr_t_now"})
    for w in WINDOWS:
        left = dep_keys.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=w)).alias("t0"))
        joined_w = left.join_asof(
            arr_c, left_on="t0", right_on="arr_t", by="airport", strategy="backward"
        ).select("MVT_ID_mvt", pl.col("cum_arr").alias(f"cum_arr_{w}"))
        joined_now = joined_now.join(joined_w, on="MVT_ID_mvt", how="left")
        joined_now = joined_now.with_columns(
            (pl.col("cum_arr_now").fill_null(0) - pl.col(f"cum_arr_{w}").fill_null(0)).alias(f"arr_prev_{w}m")
        )
    out = out.join(
        joined_now.select(["MVT_ID_mvt"] + [f"arr_prev_{w}m" for w in WINDOWS]),
        on="MVT_ID_mvt",
        how="left",
    )
    return out


def add_surface_proxy(dep: pl.DataFrame) -> pl.DataFrame:
    """Count other departures whose [AOBT_3, MVT_TIME] overlaps this flight's takeoff instant.

    At this flight's MVT_TIME: others with AOBT_3 < MVT_i <= MVT_j  (started taxi, not yet airborne)
    AND others with AOBT_3 < MVT_i < MVT_j.

    This uses only AOBT_3 and MVT_TIME, both available in ranking.

    Also count others with AOBT_3 in previous W minutes (pushbacks).
    """
    d = airport_of(dep).filter(pl.col("AOBT_3_flt").is_not_null())
    # per airport numpy
    n = d.height
    queue_at_mvt = np.full(n, np.nan)
    push_1 = np.full(n, np.nan)
    push_5 = np.full(n, np.nan)
    push_15 = np.full(n, np.nan)
    push_30 = np.full(n, np.nan)
    # map back via MVT_ID
    ids = d["MVT_ID_mvt"].to_numpy()
    airports = d["airport"].to_numpy()
    mvt = d["MVT_TIME_UTC_mvt"].to_numpy()
    aobt = d["AOBT_3_flt"].to_numpy()
    rwy = d["RUNWAY_mvt"].to_numpy()

    # process per airport
    from collections import defaultdict

    idx_by_ap = defaultdict(list)
    for i, ap in enumerate(airports):
        idx_by_ap[ap].append(i)

    for ap, idxs in idx_by_ap.items():
        idxs = np.array(idxs)
        mv = mvt[idxs].astype("datetime64[ns]").astype(np.int64)
        ao = aobt[idxs].astype("datetime64[ns]").astype(np.int64)
        rw = rwy[idxs]
        order = np.argsort(mv)
        mv_s = mv[order]
        ao_s = ao[order]
        # for each flight in time order: count others with ao < mv_i <= mv_j
        # = count of flights that have started (ao < mv_i) minus count that already took off (mv_j < mv_i)
        # takeoffs strictly before i: position in sorted mv
        # starts before mv_i: searchsorted on sorted ao
        ao_sorted = np.sort(ao_s)
        for k, orig_pos in enumerate(order):
            t = mv_s[k]
            n_started = np.searchsorted(ao_sorted, t, side="left")  # ao < t
            n_taken_off = k  # mv < t because sorted unique? ties: k includes equal? order is argsort mv, so k flights with mv <=? 
            # argsort stable, flights with same mv: k is number with earlier-or-equal depending on ties
            # takeoffs strictly before t:
            n_taken_off = np.searchsorted(mv_s, t, side="left")
            # queue = started but not taken off, exclude self if ao_self < t (always if taxi>0)
            q = n_started - n_taken_off
            # self: ao < t typically, and mv == t so not in taken_off strictly before
            # self is counted in started, not in taken_off => subtract 1
            queue_at_mvt[idxs[orig_pos]] = max(q - 1, 0)

        # pushbacks in previous W min: count ao in (t-W, t)
        # use AOBT as event time for pushbacks
        for wmin, dest in [(1, push_1), (5, push_5), (15, push_15), (30, push_30)]:
            wns = wmin * 60 * 10**9
            for k, orig_pos in enumerate(order):
                t = mv_s[k]
                # count ao in (t-w, t]
                hi = np.searchsorted(ao_sorted, t, side="right")
                lo = np.searchsorted(ao_sorted, t - wns, side="right")
                dest[idxs[orig_pos]] = max(hi - lo, 0)

    extra = pl.DataFrame(
        {
            "MVT_ID_mvt": ids,
            "queue_at_mvt": queue_at_mvt,
            "push_prev_1m": push_1,
            "push_prev_5m": push_5,
            "push_prev_15m": push_15,
            "push_prev_30m": push_30,
        }
    )
    return dep.join(extra, on="MVT_ID_mvt", how="left")


def analyze_congestion(fh, feat: pl.DataFrame):
    log(fh, "\n" + "=" * 80)
    log(fh, "CONGESTION vs TARGET (DEP)")
    log(fh, "=" * 80)
    y = feat["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    cols = []
    for w in WINDOWS:
        cols += [f"dep_prev_{w}m", f"dep_rwy_prev_{w}m", f"arr_prev_{w}m"]
    cols += ["queue_at_mvt", "push_prev_1m", "push_prev_5m", "push_prev_15m", "push_prev_30m"]
    # also derived
    feat = feat.with_columns(
        (pl.col("dep_prev_15m") + pl.col("arr_prev_15m")).alias("mov_prev_15m"),
        (pl.col("dep_prev_5m") + pl.col("arr_prev_5m")).alias("mov_prev_5m"),
        (pl.col("dep_prev_60m") + pl.col("arr_prev_60m")).alias("mov_prev_60m"),
        (pl.col("dep_prev_15m") - pl.col("dep_prev_5m")).alias("dep_5_15"),
        (pl.col("dep_prev_5m").cast(pl.Float64) - pl.col("dep_prev_15m") / 3).alias("dep_accel_approx"),
        (pl.col("dep_prev_15m") / pl.col("dep_prev_60m").clip(lower_bound=1)).alias("dep_ratio_15_60"),
    )
    cols += ["mov_prev_15m", "mov_prev_5m", "mov_prev_60m", "dep_5_15", "dep_accel_approx", "dep_ratio_15_60"]

    # stand-rwy baseline residual
    g = feat.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg(
        pl.col("TAXITIME_SEC_mvt").mean().alias("sr_mean")
    )
    feat = feat.join(g, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
    resid = (feat["TAXITIME_SEC_mvt"] - feat["sr_mean"]).to_numpy().astype(np.float64)
    ap_g = feat.group_by("ADEP_mvt").agg(pl.col("TAXITIME_SEC_mvt").mean().alias("ap_mean"))
    feat = feat.join(ap_g, on="ADEP_mvt", how="left")
    resid_ap = (feat["TAXITIME_SEC_mvt"] - feat["ap_mean"]).to_numpy().astype(np.float64)

    aobt_proxy = (feat["MVT_TIME_UTC_mvt"] - feat["AOBT_3_flt"]).dt.total_seconds().to_numpy().astype(np.float64)
    resid_aobt = y - aobt_proxy

    log(fh, f"stand-rwy in-sample RMSE={rmse(y, feat['sr_mean'].to_numpy()):.2f}")
    log(fh, f"airport in-sample RMSE={rmse(y, feat['ap_mean'].to_numpy()):.2f}")
    log(fh, f"AOBT proxy RMSE={rmse(y, aobt_proxy):.2f}")

    log(fh, "\n--- Pearson corr with TAXITIME, residual_stand_rwy, residual_airport, residual_AOBT ---")
    log(fh, f"{'feature':28s} {'n':>10s} {'corr_y':>8s} {'corr_sr':>8s} {'corr_ap':>8s} {'corr_aobt':>9s} mean    p50    p90")
    for c in cols:
        x = feat[c].to_numpy().astype(np.float64)
        m = np.isfinite(x)
        log(
            fh,
            f"{c:28s} {m.sum():10,} {corr(x,y):8.4f} {corr(x,resid):8.4f} {corr(x,resid_ap):8.4f} "
            f"{corr(x,resid_aobt):9.4f} {np.nanmean(x):7.2f} {np.nanmedian(x):7.2f} {np.nanquantile(x[m],0.9):7.2f}",
        )

    # per-airport corr of 15m dep count
    log(fh, "\n--- per-airport corr(dep_prev_15m, taxi) and corr with stand-rwy residual ---")
    for ap in AIRPORTS:
        sub = feat.filter(pl.col("ADEP_mvt") == ap)
        x = sub["dep_prev_15m"].to_numpy().astype(np.float64)
        yy = sub["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
        rr = (sub["TAXITIME_SEC_mvt"] - sub["sr_mean"]).to_numpy().astype(np.float64)
        xr = sub["dep_rwy_prev_15m"].to_numpy().astype(np.float64)
        xa = sub["arr_prev_15m"].to_numpy().astype(np.float64)
        xq = sub["queue_at_mvt"].to_numpy().astype(np.float64)
        log(
            fh,
            f"{ap} n={sub.height:7,} corr_dep15={corr(x,yy):6.3f} corr_rwy15={corr(xr,yy):6.3f} "
            f"corr_arr15={corr(xa,yy):6.3f} corr_queue={corr(xq,yy):6.3f} corr_dep15_resid={corr(x,rr):6.3f}",
        )

    # regimes by dep_prev_15m quintile
    log(fh, "\n--- taxi distribution by dep_prev_15m quantile (global) ---")
    x = feat["dep_prev_15m"].to_numpy().astype(np.float64)
    qs = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8])
    bins = [-np.inf, *qs, np.inf]
    lab = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    idx = np.digitize(x, qs)
    log(fh, "quintile edges dep_prev_15m", qs)
    for i, name in enumerate(lab):
        m = idx == i
        s = y[m]
        log(
            fh,
            f"  {name:12s} n={m.sum():7,} mean={s.mean():7.1f} med={np.median(s):7.1f} std={s.std():7.1f} "
            f"p90={np.quantile(s,0.9):7.1f} p95={np.quantile(s,0.95):7.1f} p99={np.quantile(s,0.99):7.1f}",
        )

    # same by airport using local quintiles for LTFM, EGLL, EHAM, LFPG
    log(fh, "\n--- local quintiles of dep_prev_15m vs taxi at major airports ---")
    for ap in ["LTFM", "EGLL", "EHAM", "LFPG", "EDDF"]:
        sub = feat.filter(pl.col("ADEP_mvt") == ap)
        x = sub["dep_prev_15m"].to_numpy().astype(np.float64)
        yy = sub["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
        qsl = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8])
        idxi = np.digitize(x, qsl)
        log(fh, f"{ap} edges={qsl}")
        for i, name in enumerate(lab):
            m = idxi == i
            s = yy[m]
            log(
                fh,
                f"  {name:12s} n={m.sum():6,} mean={s.mean():7.1f} med={np.median(s):7.1f} std={s.std():6.1f} p95={np.quantile(s,0.95):7.1f}",
            )

    # stand-rwy residual vs congestion quintile
    log(fh, "\n--- stand-rwy residual by dep_prev_15m quintile ---")
    for i, name in enumerate(lab):
        m = idx == i
        s = resid[m]
        s = s[np.isfinite(s)]
        log(
            fh,
            f"  {name:12s} resid_mean={s.mean():7.1f} resid_med={np.median(s):7.1f} resid_std={s.std():7.1f}",
        )

    # simple linear RMSE: stand-rwy mean + b * dep_prev_15m (in-sample)
    x15 = feat["dep_prev_15m"].to_numpy().astype(np.float64)
    m = np.isfinite(x15) & np.isfinite(resid)
    b = np.polyfit(x15[m], resid[m], 1)
    pred = feat["sr_mean"].to_numpy() + b[0] * x15 + b[1]
    log(fh, f"\nin-sample stand-rwy + linear dep15: slope={b[0]:.3f} intercept={b[1]:.3f} RMSE={rmse(y, pred):.2f}")
    xr = feat["dep_rwy_prev_15m"].to_numpy().astype(np.float64)
    m2 = np.isfinite(xr) & np.isfinite(resid)
    b2 = np.polyfit(xr[m2], resid[m2], 1)
    pred2 = feat["sr_mean"].to_numpy() + b2[0] * xr + b2[1]
    log(fh, f"in-sample stand-rwy + linear rwy15: slope={b2[0]:.3f} intercept={b2[1]:.3f} RMSE={rmse(y, pred2):.2f}")
    xq = feat["queue_at_mvt"].to_numpy().astype(np.float64)
    m3 = np.isfinite(xq) & np.isfinite(resid)
    b3 = np.polyfit(xq[m3], resid[m3], 1)
    pred3 = feat["sr_mean"].to_numpy() + np.where(np.isfinite(xq), b3[0] * xq + b3[1], 0)
    log(fh, f"in-sample stand-rwy + linear queue: slope={b3[0]:.3f} intercept={b3[1]:.3f} RMSE={rmse(y, pred3):.2f}")

    # AOBT proxy + congestion
    m4 = np.isfinite(aobt_proxy) & np.isfinite(x15)
    b4 = np.polyfit(np.column_stack([aobt_proxy[m4], x15[m4]]), y[m4], 1) if False else None
    # two-feature OLS
    X = np.column_stack([np.ones(m4.sum()), aobt_proxy[m4], x15[m4]])
    coef, *_ = np.linalg.lstsq(X, y[m4], rcond=None)
    pred4 = coef[0] + coef[1] * aobt_proxy + coef[2] * x15
    log(fh, f"OLS AOBT_proxy + dep15: coef={coef} RMSE={rmse(y, pred4):.2f}")

    X2 = np.column_stack(
        [
            np.ones(m4.sum()),
            aobt_proxy[m4],
            x15[m4],
            feat.filter(pl.lit(True))["dep_rwy_prev_15m"].to_numpy().astype(np.float64)[m4],
        ]
    )
    # careful with filter - feat row order same as numpy arrays from feat
    xr_all = feat["dep_rwy_prev_15m"].to_numpy().astype(np.float64)
    xa_all = feat["arr_prev_15m"].to_numpy().astype(np.float64)
    xq_all = feat["queue_at_mvt"].to_numpy().astype(np.float64)
    m5 = m4 & np.isfinite(xr_all) & np.isfinite(xa_all)
    X5 = np.column_stack([np.ones(m5.sum()), aobt_proxy[m5], x15[m5], xr_all[m5], xa_all[m5]])
    coef5, *_ = np.linalg.lstsq(X5, y[m5], rcond=None)
    pred5 = np.full_like(y, np.nan)
    pred5[m5] = X5 @ coef5
    log(fh, f"OLS AOBT + dep15 + rwy15 + arr15: coef={coef5} RMSE={rmse(y, pred5):.2f}")

    m6 = m5 & np.isfinite(xq_all)
    X6 = np.column_stack([np.ones(m6.sum()), aobt_proxy[m6], x15[m6], xr_all[m6], xa_all[m6], xq_all[m6]])
    coef6, *_ = np.linalg.lstsq(X6, y[m6], rcond=None)
    pred6 = np.full_like(y, np.nan)
    pred6[m6] = X6 @ coef6
    log(fh, f"OLS AOBT + dep15 + rwy15 + arr15 + queue: coef={coef6} RMSE={rmse(y, pred6):.2f}")

    # hard cases
    log(fh, "\n" + "=" * 80)
    log(fh, "HARD CASES (taxi > p99 and > 30min)")
    log(fh, "=" * 80)
    p99 = np.nanquantile(y, 0.99)
    log(fh, f"p99={p99:.1f}")
    for label, mask in [
        ("taxi>p99", y > p99),
        ("taxi>30min", y > 1800),
        ("taxi>1h", y > 3600),
        ("taxi<5min", (y > 0) & (y < 300)),
    ]:
        sub = feat.filter(pl.Series(mask))
        log(fh, f"\n--- {label} n={sub.height:,} ({100*sub.height/feat.height:.3f}%) ---")
        log(fh, "airports:", sub.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))
        log(
            fh,
            "mean dep15={:.1f} vs all {:.1f}".format(
                float(sub["dep_prev_15m"].mean() or 0), float(feat["dep_prev_15m"].mean() or 0)
            ),
        )
        log(
            fh,
            "mean rwy15={:.1f} vs all {:.1f}".format(
                float(sub["dep_rwy_prev_15m"].mean() or 0), float(feat["dep_rwy_prev_15m"].mean() or 0)
            ),
        )
        log(
            fh,
            "mean queue={:.1f} vs all {:.1f}".format(
                float(sub["queue_at_mvt"].mean() or 0), float(feat["queue_at_mvt"].mean() or 0)
            ),
        )
        log(
            fh,
            "mean hour UTC:",
            sub.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("h")).group_by("h").agg(pl.len().alias("n")).sort("n", descending=True).head(6),
        )
        log(fh, "wtc:", sub.group_by("WK_TBL_CAT_flt").agg(pl.len().alias("n")).sort("n", descending=True))
        log(fh, "market:", sub.group_by("MARKET_SEGMENT_flt").agg(pl.len().alias("n")).sort("n", descending=True))

    # hour x airport mean for a couple
    log(fh, "\n--- EGLL hour vs taxi and dep15 ---")
    eg = feat.filter(pl.col("ADEP_mvt") == "EGLL").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("h"))
    log(
        fh,
        eg.group_by("h").agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("taxi_mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("taxi_med"),
            pl.col("dep_prev_15m").mean().alias("dep15"),
            pl.col("arr_prev_15m").mean().alias("arr15"),
        ).sort("h"),
    )

    # rolling taxi of PREVIOUS flights cannot be used at test time identically.
    # But we can compute TRAIN-only relationship: lag taxi of previous takeoff at same airport/runway
    log(fh, "\n--- lag-1 previous taxi at airport (TRAIN-ONLY, not available as-is in ranking) ---")
    srt = feat.sort(["ADEP_mvt", "MVT_TIME_UTC_mvt"])
    srt = srt.with_columns(pl.col("TAXITIME_SEC_mvt").shift(1).over("ADEP_mvt").alias("lag1_taxi"))
    xlag = srt["lag1_taxi"].to_numpy().astype(np.float64)
    ylag = srt["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    log(fh, f"corr lag1 previous taxi (same airport): {corr(xlag, ylag):.4f}")
    srt = srt.with_columns(pl.col("TAXITIME_SEC_mvt").shift(1).over(["ADEP_mvt", "RUNWAY_mvt"]).alias("lag1_rwy_taxi"))
    log(fh, f"corr lag1 previous taxi (same rwy): {corr(srt['lag1_rwy_taxi'].to_numpy().astype(float), ylag):.4f}")
    # rolling mean of previous 10 taxis
    srt = srt.with_columns(
        pl.col("TAXITIME_SEC_mvt").shift(1).rolling_mean(window_size=10).over("ADEP_mvt").alias("roll10_taxi")
    )
    log(fh, f"corr roll10 previous taxi (same airport): {corr(srt['roll10_taxi'].to_numpy().astype(float), ylag):.4f}")
    log(fh, "NOTE: these use actual TAXITIME of other DEPs which are BLANK in ranking. Only usable via AOBT-proxy taxi.")

    # AOBT-proxy rolling
    srt = srt.with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("aobt_taxi")
    )
    srt = srt.with_columns(pl.col("aobt_taxi").shift(1).over("ADEP_mvt").alias("lag1_aobt_taxi"))
    srt = srt.with_columns(
        pl.col("aobt_taxi").shift(1).rolling_mean(window_size=10).over("ADEP_mvt").alias("roll10_aobt_taxi")
    )
    log(fh, f"corr lag1 AOBT-proxy taxi: {corr(srt['lag1_aobt_taxi'].to_numpy().astype(float), ylag):.4f}")
    log(fh, f"corr roll10 AOBT-proxy taxi: {corr(srt['roll10_aobt_taxi'].to_numpy().astype(float), ylag):.4f}")

    return feat


def arrival_state(fh, df: pl.DataFrame):
    log(fh, "\n" + "=" * 80)
    log(fh, "ARRIVALS AS SURFACE-STATE (available in ranking including taxi-in)")
    log(fh, "=" * 80)
    arr = airport_of(df.filter(pl.col("PHASE_mvt") == "ARR"))
    log(fh, "ARR taxi-in stats by airport:")
    log(
        fh,
        arr.group_by("airport").agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("med"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        ).sort("airport"),
    )
    # currently taxiing in at a dep's takeoff: ARR with MVT < t < BLOCK
    # skip heavy nested; report that ARR BLOCK is 100% present in ranking ARR
    rank = pl.scan_parquet(DATA / "ranking.parquet")
    log(
        fh,
        "ranking ARR BLOCK nulls",
        rank.filter(pl.col("PHASE_mvt") == "ARR").select(pl.col("BLOCK_TIME_UTC_mvt").is_null().sum()).collect().item(),
    )
    log(
        fh,
        "ranking ARR TAXITIME nulls",
        rank.filter(pl.col("PHASE_mvt") == "ARR").select(pl.col("TAXITIME_SEC_mvt").is_null().sum()).collect().item(),
    )


def main():
    out = OUT / "03_congestion_state.txt"
    with open(out, "w", encoding="utf-8") as fh:
        log(fh, "loading...")
        df = load_all()
        log(fh, "loaded", df.height)
        density(fh, df)
        dep = df.filter(pl.col("PHASE_mvt") == "DEP")
        arr = df.filter(pl.col("PHASE_mvt") == "ARR")
        log(fh, "computing rolling counts...")
        feat = add_rolling_counts(dep, arr)
        log(fh, "computing surface queue proxy (per-airport numpy)...")
        feat = add_surface_proxy(feat)
        analyze_congestion(fh, feat)
        arrival_state(fh, df)
    print("WROTE", out)


if __name__ == "__main__":
    main()
