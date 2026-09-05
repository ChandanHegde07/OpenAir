"""Target distribution, naive predictors, timestamp resolution, airport/stand/runway."""
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


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def load_train():
    return pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])


def print_series_stats(fh, name, s: np.ndarray):
    s = s.astype(np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    qv = np.quantile(s, qs)
    log(fh, f"\n[{name}] n={n:,}")
    log(fh, f"  mean={s.mean():.4f}  median={np.median(s):.4f}  std={s.std(ddof=1):.4f}")
    log(fh, f"  min={s.min():.4f}  max={s.max():.4f}")
    log(fh, f"  skew={((s.mean()-np.median(s))/s.std(ddof=1)):.4f} (mean-median)/std")
    for q, v in zip(qs, qv):
        log(fh, f"  P{int(q*1000)/10:g}={v:.4f}")
    log(fh, f"  neg={(s<0).sum():,}  zero={(s==0).sum():,}  gt30m={(s>1800).sum():,}  gt1h={(s>3600).sum():,}  gt2h={(s>7200).sum():,}")
    # RMSE if predict 0 relative to mean: variance
    # tail SSE share
    mean = s.mean()
    err2 = (s - mean) ** 2
    total = err2.sum()
    for thr in [1800, 2700, 3600, 5400, 7200]:
        mask = s > thr
        share = err2[mask].sum() / total if total else 0
        log(fh, f"  SSE share of values >{thr}s ({mask.sum():,} rows, {100*mask.mean():.3f}%): {100*share:.2f}% of mean-model SSE")
    # histogram
    bins = [-1e9, 0, 120, 300, 480, 600, 720, 900, 1080, 1200, 1500, 1800, 2400, 3600, 7200, 1e12]
    labels = ["<0","0-2m","2-5m","5-8m","8-10m","10-12m","12-15m","15-18m","18-20m","20-25m","25-30m","30-40m","40-60m","1-2h",">2h"]
    idx = np.digitize(s, bins) - 1
    log(fh, "  histogram:")
    for i, lab in enumerate(labels):
        c = (idx == i).sum()
        log(fh, f"    {lab:10s} {c:10,} ({100*c/n:6.2f}%)")


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), int(m.sum())


def mae(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))), int(m.sum())


def target_and_naive(fh):
    log(fh, "=" * 80)
    log(fh, "TARGET DISTRIBUTION (DEP only) + NAIVE PREDICTORS")
    log(fh, "=" * 80)
    train = load_train()
    dep = (
        train.filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            "FLIGHT_ID_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "TAXITIME_SEC_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_TYPE_flt",
            "WK_TBL_CAT_flt",
            "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt",
            "FLIGHT_TYPE_flt",
            "LOBT_flt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_1_flt",
            "ARVT_3_flt",
            "ADES_FILED_flt",
            "ADEP_flt",
            "ADES_flt",
        )
        .collect()
    )
    y = dep["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    print_series_stats(fh, "TAXITIME_SEC DEP", y)

    # timestamp diffs
    def sec(a, b):
        return (dep[a] - dep[b]).dt.total_seconds().to_numpy()

    proxies = {
        "MVT-AOBT3": sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        "MVT-LOBT": sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        "MVT-EOBT": sec("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
        "MVT-IOBT": sec("MVT_TIME_UTC_mvt", "IOBT_flt"),
        "MVT-SCHED": sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "BLOCK-AOBT3": sec("BLOCK_TIME_UTC_mvt", "AOBT_3_flt"),
        "BLOCK-LOBT": sec("BLOCK_TIME_UTC_mvt", "LOBT_flt"),
        "BLOCK-EOBT": sec("BLOCK_TIME_UTC_mvt", "EOBT_1_flt"),
        "BLOCK-SCHED": sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "AOBT3-LOBT": sec("AOBT_3_flt", "LOBT_flt"),
        "AOBT3-EOBT": sec("AOBT_3_flt", "EOBT_1_flt"),
        "planned_dur_ARVT1-EOBT": sec("ARVT_1_flt", "EOBT_1_flt"),
        "actual_dur_ARVT3-AOBT3": sec("ARVT_3_flt", "AOBT_3_flt"),
        "ARVT3-MVT (airborne after takeoff)": sec("ARVT_3_flt", "MVT_TIME_UTC_mvt"),
    }
    log(fh, "\n--- timestamp difference stats (seconds) ---")
    for name, arr in proxies.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            log(fh, f"{name}: no finite")
            continue
        log(
            fh,
            f"{name:34s} n={finite.size:,} mean={finite.mean():.1f} med={np.median(finite):.1f} "
            f"std={finite.std(ddof=1):.1f} p5={np.quantile(finite,0.05):.1f} p95={np.quantile(finite,0.95):.1f} "
            f"min={finite.min():.1f} max={finite.max():.1f} |mean_abs|={np.mean(np.abs(finite)):.1f}",
        )

    log(fh, "\n--- naive predictor RMSE/MAE vs TAXITIME ---")
    mean_y = np.nanmean(y)
    med_y = np.nanmedian(y)
    for name, pred in [
        ("global_mean", np.full_like(y, mean_y)),
        ("global_median", np.full_like(y, med_y)),
        ("MVT-AOBT3", proxies["MVT-AOBT3"]),
        ("MVT-LOBT", proxies["MVT-LOBT"]),
        ("MVT-EOBT", proxies["MVT-EOBT"]),
        ("MVT-IOBT", proxies["MVT-IOBT"]),
        ("MVT-SCHED", proxies["MVT-SCHED"]),
    ]:
        r, n = rmse(y, pred)
        m, _ = mae(y, pred)
        log(fh, f"  {name:20s} RMSE={r:8.2f} MAE={m:8.2f} n={n:,}")

    # clip negative proxies at 0 and at 0-3600
    for name in ["MVT-AOBT3", "MVT-LOBT", "MVT-EOBT"]:
        p = proxies[name].copy()
        p = np.clip(p, 0, 7200)
        r, n = rmse(y, p)
        m, _ = mae(y, p)
        log(fh, f"  {name+'-clip0-2h':20s} RMSE={r:8.2f} MAE={m:8.2f} n={n:,}")

    # group-mean predictors (in-sample, optimistic)
    dep = dep.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    for keys, label in [
        (["ADEP_mvt"], "airport_mean"),
        (["ADEP_mvt", "RUNWAY_mvt"], "airport_rwy_mean"),
        (["ADEP_mvt", "STAND_mvt"], "airport_stand_mean"),
        (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], "stand_rwy_mean"),
        (["ADEP_mvt", "AIRCRAFT_TYPE_mvt"], "airport_actype_mean"),
        (["ADEP_mvt", "WK_TBL_CAT_flt"], "airport_wtc_mean"),
        (["ADEP_mvt", "MARKET_SEGMENT_flt"], "airport_mkt_mean"),
        (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], "airport_op_mean"),
        (["ADEP_mvt", "RUNWAY_mvt", "WK_TBL_CAT_flt"], "rwy_wtc_mean"),
    ]:
        g = dep.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").mean().alias("_p"))
        joined = dep.join(g, on=keys, how="left")
        r, n = rmse(joined["TAXITIME_SEC_mvt"].to_numpy(), joined["_p"].to_numpy())
        m, _ = mae(joined["TAXITIME_SEC_mvt"].to_numpy(), joined["_p"].to_numpy())
        nunq = g.select(pl.len()).item()
        log(fh, f"  {label:20s} RMSE={r:8.2f} MAE={m:8.2f} n={n:,} groups={nunq:,}  [IN-SAMPLE]")

    # hour / dow / month
    dep_t = dep.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("sched_hour"),
    )
    for keys, label in [
        (["ADEP_mvt", "hour"], "airport_hour_mean"),
        (["ADEP_mvt", "dow"], "airport_dow_mean"),
        (["ADEP_mvt", "month"], "airport_month_mean"),
        (["ADEP_mvt", "hour", "dow"], "airport_hour_dow_mean"),
        (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour"], "stand_rwy_hour_mean"),
    ]:
        g = dep_t.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").mean().alias("_p"))
        joined = dep_t.join(g, on=keys, how="left")
        r, n = rmse(joined["TAXITIME_SEC_mvt"].to_numpy(), joined["_p"].to_numpy())
        m, _ = mae(joined["TAXITIME_SEC_mvt"].to_numpy(), joined["_p"].to_numpy())
        nunq = g.select(pl.len()).item()
        log(fh, f"  {label:20s} RMSE={r:8.2f} MAE={m:8.2f} n={n:,} groups={nunq:,}  [IN-SAMPLE]")

    # AOBT proxy + stand-rwy residual? later

    log(fh, "\n--- timestamp resolution (unique second-of vs unique minute) ---")
    for c in [
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "LOBT_flt",
        "IOBT_flt",
        "EOBT_1_flt",
        "AOBT_3_flt",
        "ARVT_1_flt",
        "ARVT_3_flt",
    ]:
        s = dep.filter(pl.col(c).is_not_null()).select(pl.col(c))
        n = s.select(pl.len()).item()
        n_ts = s.select(pl.col(c).n_unique()).item()
        n_min = s.select(pl.col(c).dt.truncate("1m").n_unique()).item()
        n_sec0 = s.select((pl.col(c).dt.second() == 0).sum()).item()
        # sub-minute: not on exact second 0 of a minute... actually second()!=0
        n_submin = s.select((pl.col(c).dt.second() != 0).sum()).item()
        n_subsec = s.select((pl.col(c).dt.microsecond() != 0).sum()).item()
        log(
            fh,
            f"  {c:22s} n={n:,} unique={n_ts:,} unique_min={n_min:,} "
            f"frac_exact_min={n_sec0/n:.3f} frac_submin={n_submin/n:.3f} frac_subsec={n_subsec/n:.3f}",
        )

    return dep, dep_t, y, proxies


def airport_breakdown(fh, dep: pl.DataFrame):
    log(fh, "\n" + "=" * 80)
    log(fh, "AIRPORT BREAKDOWN (DEP)")
    log(fh, "=" * 80)
    rows = []
    for ap in AIRPORTS:
        s = dep.filter(pl.col("ADEP_mvt") == ap)["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
        rows.append(
            (
                ap,
                s.size,
                s.mean(),
                np.median(s),
                s.std(ddof=1),
                s.min(),
                s.max(),
                np.quantile(s, 0.05),
                np.quantile(s, 0.25),
                np.quantile(s, 0.75),
                np.quantile(s, 0.90),
                np.quantile(s, 0.95),
                np.quantile(s, 0.99),
                (s > 1800).mean() * 100,
                (s > 3600).mean() * 100,
            )
        )
    log(
        fh,
        "AP     n         mean    med     std     p5      p25     p75     p90     p95     p99     %>30m  %>1h",
    )
    for r in rows:
        log(
            fh,
            f"{r[0]} {r[1]:8,} {r[2]:7.1f} {r[3]:7.1f} {r[4]:7.1f} {r[7]:7.1f} {r[8]:7.1f} "
            f"{r[9]:7.1f} {r[10]:7.1f} {r[11]:7.1f} {r[12]:7.1f} {r[13]:6.2f} {r[14]:6.3f}",
        )

    log(fh, "\n--- categorical vs target (DEP) ---")
    for col in ["WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]:
        g = (
            dep.group_by(col)
            .agg(
                pl.len().alias("n"),
                pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
                pl.col("TAXITIME_SEC_mvt").median().alias("median"),
                pl.col("TAXITIME_SEC_mvt").std().alias("std"),
                pl.col("TAXITIME_SEC_mvt").quantile(0.95).alias("p95"),
            )
            .sort("n", descending=True)
        )
        log(fh, f"\n{col}:")
        log(fh, g)

    # top aircraft types
    g = (
        dep.group_by("AIRCRAFT_TYPE_mvt")
        .agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        )
        .sort("n", descending=True)
        .head(25)
    )
    log(fh, "\nTop aircraft types:")
    log(fh, g)

    # runways per airport
    log(fh, "\n--- runways per airport ---")
    rwy = (
        dep.group_by(["ADEP_mvt", "RUNWAY_mvt"])
        .agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        )
        .sort(["ADEP_mvt", "n"], descending=[False, True])
    )
    log(fh, rwy)
    log(fh, "unique runways", dep.select(pl.col("RUNWAY_mvt").n_unique()).item())
    log(fh, "unique stands", dep.select(pl.col("STAND_mvt").n_unique()).item())
    log(
        fh,
        "unique stand-rwy pairs",
        dep.select(pl.struct(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).n_unique()).item(),
    )
    log(
        fh,
        "unique stands per airport:",
        dep.group_by("ADEP_mvt").agg(pl.col("STAND_mvt").n_unique().alias("n_stands"), pl.len().alias("n")).sort("ADEP_mvt"),
    )

    # hour effects
    log(fh, "\n--- hour of MVT (UTC) vs taxi ---")
    hg = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        )
        .sort("hour")
    )
    log(fh, hg)

    log(fh, "\n--- weekday (1=Mon) vs taxi ---")
    dg = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"))
        .group_by("dow")
        .agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        )
        .sort("dow")
    )
    log(fh, dg)

    log(fh, "\n--- month vs taxi ---")
    mg = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().alias("mean"),
            pl.col("TAXITIME_SEC_mvt").median().alias("median"),
            pl.col("TAXITIME_SEC_mvt").std().alias("std"),
        )
        .sort("month")
    )
    log(fh, mg)

    # ADEP mvt vs flt mismatch
    log(fh, "\n--- mvt vs flt airport mismatches (DEP) ---")
    log(fh, "ADEP_mvt != ADEP_flt", dep.filter(pl.col("ADEP_mvt") != pl.col("ADEP_flt")).select(pl.len()).item())
    log(fh, "ADES_mvt != ADES_flt", dep.filter(pl.col("ADES_mvt") != pl.col("ADES_flt")).select(pl.len()).item())
    log(
        fh,
        "ADES_flt != ADES_FILED (diversion)",
        dep.filter(
            pl.col("ADES_flt").is_not_null()
            & pl.col("ADES_FILED_flt").is_not_null()
            & (pl.col("ADES_flt") != pl.col("ADES_FILED_flt"))
        )
        .select(pl.len())
        .item(),
    )
    log(fh, "AIRCRAFT_TYPE_mvt != AIRCRAFT_TYPE_flt", dep.filter(pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt")).select(pl.len()).item())


def flight_linking(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "FLIGHT_ID LINKING (ARR/DEP of same NM flight)")
    log(fh, "=" * 80)
    train = load_train()
    ids = train.select("FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt").collect()
    g = (
        ids.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
        .group_by("FLIGHT_ID_mvt")
        .agg(
            pl.len().alias("n"),
            pl.col("PHASE_mvt").n_unique().alias("n_phase"),
            pl.concat_str(pl.col("PHASE_mvt"), separator=",").alias("phases"),
        )
    )
    log(fh, "FLIGHT_ID multiplicity:")
    log(fh, g.group_by("n").agg(pl.len().alias("n_ids")).sort("n"))
    both = g.filter(pl.col("n") == 2)
    log(fh, "exactly 2 rows", both.select(pl.len()).item())
    # among n==2, how many are ARR+DEP
    two = (
        ids.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
        .join(g.filter(pl.col("n") == 2).select("FLIGHT_ID_mvt"), on="FLIGHT_ID_mvt")
        .group_by("FLIGHT_ID_mvt")
        .agg(pl.col("PHASE_mvt").unique().alias("ph"))
        .with_columns(pl.col("ph").list.len().alias("nph"))
    )
    log(fh, "n==2 with 2 distinct phases", two.filter(pl.col("nph") == 2).select(pl.len()).item())
    log(fh, "n==2 with 1 phase", two.filter(pl.col("nph") == 1).select(pl.len()).item())


def ranking_proxy(fh):
    log(fh, "\n" + "=" * 80)
    log(fh, "RANKING: AOBT-based proxy taxi (MVT-AOBT) distribution vs training")
    log(fh, "=" * 80)
    rank = (
        pl.scan_parquet(DATA / "ranking.parquet")
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt")
        .collect()
    )
    proxy = (rank["MVT_TIME_UTC_mvt"] - rank["AOBT_3_flt"]).dt.total_seconds().to_numpy()
    print_series_stats(fh, "RANKING MVT-AOBT3 (DEP)", proxy.astype(np.float64))
    months = rank.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).group_by("month").agg(pl.len())
    log(fh, "ranking DEP months", months)
    log(fh, "ranking DEP airports", rank.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("ADEP_mvt"))


def main():
    out = OUT / "02_target_baselines.txt"
    with open(out, "w", encoding="utf-8") as fh:
        dep, dep_t, y, proxies = target_and_naive(fh)
        airport_breakdown(fh, dep)
        flight_linking(fh)
        ranking_proxy(fh)
        # save dep parquet for later scripts? too big. keep lazy.
    print("WROTE", out)


if __name__ == "__main__":
    main()
