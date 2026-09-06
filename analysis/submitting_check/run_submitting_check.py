"""Diagnostic: apply frozen E16-A to ranking DEP / submitting IDs.

Fits use training_*.parquet only. Ranking is the prediction-time information
set (covariates + contemporaneous ARR). Ranking/submitting never enter OLS,
geometry, hour baselines, or LightGBM.

Ranking DEP has no TAXITIME/BLOCK, so RMSE on ranking cannot be computed.
Jan+Jul 2025 holdout predictions are produced for distribution comparison.

This is not a leaderboard submission.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE if (HERE / "common.py").exists() else HERE.parents[1] / "experiments"
sys.path.insert(0, str(EXP_DIR))

from common import (  # noqa: E402
    AIRPORTS,
    DATA,
    DEP_COLS,
    ROOT,
    add_causal_rolling,
    fill_with_fallback,
    load_dep,
    sec,
    split_by_months,
)
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    attach_hour_baseline,
    fit_residual,
    load_arr_delay,
    qcut_labels,
)
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "submitting_check"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

SUBMIT_PATH = DATA / "submitting.parquet"
RANK_PATH = DATA / "ranking.parquet"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    }
)

# Prediction-time columns. Never BLOCK or TAXITIME of the scored departure.
SCORE_DEP_COLS = [c for c in DEP_COLS if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]
SCORE_ARR_TRAFFIC_COLS = ["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]
SCORE_ARR_DELAY_COLS = ["ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def savefig(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"  figure {path.name}")
    return path


def savecsv(name: str, df: pd.DataFrame) -> Path:
    path = TAB / name
    df.to_csv(path, index=False)
    log(f"  table {path.name}")
    return path


def json_conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def encode_dep_clocks(df: pl.DataFrame) -> pl.DataFrame:
    """Same clocks as common.load_dep, without reading BLOCK/TAXITIME as features."""
    df = df.with_columns(
        pl.col("ADEP_mvt").alias("airport"),
        sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_aobt"),
        sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_eobt"),
        sec("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_iobt"),
        sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_lobt"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_sched"),
        sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt_eobt"),
        sec("AOBT_3_flt", "IOBT_flt").alias("aobt_iobt"),
        sec("AOBT_3_flt", "LOBT_flt").alias("aobt_lobt"),
        sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt_sched"),
        sec("EOBT_1_flt", "IOBT_flt").alias("eobt_iobt"),
        sec("EOBT_1_flt", "LOBT_flt").alias("eobt_lobt"),
        sec("IOBT_flt", "LOBT_flt").alias("iobt_lobt"),
        pl.col("AOBT_3_flt").is_null().alias("unmatched"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("sched_hour"),
        pl.col("AIRCRAFT_TYPE_mvt").str.slice(0, 3).alias("ac_family"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"),
    )
    clocks = df.select(["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"])
    arr = []
    for c in clocks.columns:
        arr.append(clocks[c].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[ns]").astype(np.float64))
    stack = np.stack(arr, axis=1) / 1e9
    with np.errstate(all="ignore"):
        clock_std = np.nanstd(stack, axis=1, ddof=0)
        clock_range = np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)
        n_clocks = np.sum(np.isfinite(stack), axis=1)
    clock_std[~np.isfinite(clock_std)] = np.nan
    clock_range[~np.isfinite(clock_range)] = np.nan
    return df.with_columns(
        pl.Series("clock_std", clock_std),
        pl.Series("clock_range", clock_range),
        pl.Series("n_clocks", n_clocks.astype(np.int32)),
        pl.col("mvt_aobt").abs().alias("abs_mvt_aobt"),
        pl.col("aobt_eobt").abs().alias("abs_aobt_eobt"),
        pl.col("aobt_iobt").abs().alias("abs_aobt_iobt"),
        pl.col("aobt_lobt").abs().alias("abs_aobt_lobt"),
    ).sort(["airport", "MVT_TIME_UTC_mvt"])


def featurize(dep: pl.DataFrame, arr_traffic: pl.DataFrame, arr_delay: pl.DataFrame) -> pl.DataFrame:
    dep = add_causal_rolling(dep)
    if "type_null" not in dep.columns:
        dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    dep = add_traffic(dep, arr_traffic)
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, arr_delay)
    return dep


def verify_submitting() -> dict:
    lf = pl.scan_parquet(SUBMIT_PATH)
    schema = lf.collect_schema()
    names = list(schema.names())
    n = lf.select(pl.len()).collect().item()
    n_unique = lf.select(pl.col("MVT_ID_mvt").n_unique()).collect().item()
    n_taxi_null = lf.select(pl.col("TAXITIME_SEC_mvt").null_count()).collect().item()
    n_taxi_ok = lf.select(pl.col("TAXITIME_SEC_mvt").is_not_null().sum()).collect().item()
    info = {
        "path": str(SUBMIT_PATH),
        "n": int(n),
        "n_unique_mvt_id": int(n_unique),
        "columns": names,
        "schema": {k: str(v) for k, v in schema.items()},
        "taxitime_null": int(n_taxi_null),
        "taxitime_non_null": int(n_taxi_ok),
        "schema_ok": names == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
        "unique_ok": n == n_unique,
        "target_all_null": n_taxi_ok == 0,
        "n_ok": n == 344841,
    }
    log(
        f"submitting n={n:,} unique={n_unique:,} cols={names} "
        f"taxi_null={n_taxi_null:,} schema_ok={info['schema_ok']}"
    )
    if not (info["schema_ok"] and info["unique_ok"] and info["target_all_null"] and info["n_ok"]):
        raise SystemExit(f"submitting.parquet failed verification: {info}")
    return info


def load_ranking_score() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Prediction-time frames from ranking.parquet. No BLOCK/TAXITIME of DEP."""
    if not RANK_PATH.exists():
        raise SystemExit(f"ranking.parquet not found at {RANK_PATH}")
    raw = pl.scan_parquet(RANK_PATH)
    n_rank = raw.select(pl.len()).collect().item()
    dep = (
        raw.filter(pl.col("PHASE_mvt") == "DEP")
        .select(SCORE_DEP_COLS)
        .collect()
    )
    arr_tr = (
        raw.filter(pl.col("PHASE_mvt") == "ARR")
        .select(SCORE_ARR_TRAFFIC_COLS)
        .with_columns(pl.col("ADES_mvt").alias("airport"))
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    arr_d = (
        raw.filter(pl.col("PHASE_mvt") == "ARR")
        .select(SCORE_ARR_DELAY_COLS)
        .with_columns(
            pl.col("ADES_mvt").alias("airport"),
            sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("arr_sched_delay"),
        )
        .select("airport", "MVT_TIME_UTC_mvt", "arr_sched_delay")
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    dep = encode_dep_clocks(dep).with_columns(pl.lit(None).cast(pl.Float64).alias("y"))
    meta = {
        "ranking_rows": int(n_rank),
        "ranking_dep": int(dep.height),
        "ranking_arr": int(arr_tr.height),
        "unmatched": int(dep["unmatched"].sum()),
        "type_null": int(dep["type_null"].sum()),
        "lirf_unmatched": int(dep.filter((pl.col("airport") == "LIRF") & pl.col("unmatched")).height),
        "tmin": str(dep["MVT_TIME_UTC_mvt"].min()),
        "tmax": str(dep["MVT_TIME_UTC_mvt"].max()),
        "months": dep.group_by("month").len().sort("month").to_dicts(),
    }
    log(
        f"ranking DEP={meta['ranking_dep']:,} ARR={meta['ranking_arr']:,} "
        f"unmatched={meta['unmatched']:,} LIRF unmatched={meta['lirf_unmatched']:,} "
        f"{meta['tmin']} .. {meta['tmax']}"
    )
    return dep, arr_tr, arr_d, meta


def pred_summary(pred: np.ndarray, name: str) -> dict:
    p = np.asarray(pred, dtype=np.float64)
    p = p[np.isfinite(p)]
    qs = np.quantile(p, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    bins = {
        "lt_5m": float(np.mean(p < 300)),
        "lt_10m": float(np.mean(p < 600)),
        "10_20m": float(np.mean((p >= 600) & (p < 1200))),
        "20_30m": float(np.mean((p >= 1200) & (p < 1800))),
        "30_60m": float(np.mean((p >= 1800) & (p < 3600))),
        "gt_60m": float(np.mean(p >= 3600)),
        "n_lt_5m": int((p < 300).sum()),
        "n_lt_10m": int((p < 600).sum()),
        "n_10_20m": int(((p >= 600) & (p < 1200)).sum()),
        "n_20_30m": int(((p >= 1200) & (p < 1800)).sum()),
        "n_30_60m": int(((p >= 1800) & (p < 3600)).sum()),
        "n_gt_60m": int((p >= 3600).sum()),
        "n_negative": int((p < 0).sum()),
        "n_gt_4h": int((p > 14400).sum()),
        "n_gt_24h": int((p > 86400).sum()),
    }
    return {
        "name": name,
        "n": int(p.size),
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "median": float(np.median(p)),
        "std": float(p.std(ddof=1)),
        "p1": float(qs[0]),
        "p5": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        **bins,
    }


def airport_pred_table(airport: np.ndarray, pred: np.ndarray, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for a in AIRPORTS:
        m = airport == a
        if m.sum() == 0:
            continue
        p = pred[m]
        rec = {
            "airport": a,
            "n": int(m.sum()),
            "mean": float(np.nanmean(p)),
            "median": float(np.nanmedian(p)),
            "std": float(np.nanstd(p, ddof=1)) if m.sum() > 1 else 0.0,
            "p5": float(np.nanquantile(p, 0.05)),
            "p95": float(np.nanquantile(p, 0.95)),
            "frac_gt30": float(np.mean(p > 1800)),
            "frac_gt60": float(np.mean(p > 3600)),
            "n_neg": int(np.sum(p < 0)),
        }
        if extra:
            rec.update({k: extra[k][m].mean() if hasattr(extra[k], "mean") else extra[k] for k in []})
        rows.append(rec)
    return pd.DataFrame(rows)


def dis_q_table(dis: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    lab = qcut_labels(dis, 8, "Q")
    pdf = pd.DataFrame({"q": lab, "pred": pred, "dis": dis})
    rows = []
    for q, sub in pdf.groupby("q", dropna=False):
        rows.append(
            {
                "dis_q": q,
                "n": len(sub),
                "mean_dis": float(np.nanmean(sub["dis"])),
                "mean_pred": float(np.nanmean(sub["pred"])),
                "median_pred": float(np.nanmedian(sub["pred"])),
                "frac_gt30": float(np.mean(sub["pred"] > 1800)),
                "frac_gt60": float(np.mean(sub["pred"] > 3600)),
            }
        )
    return pd.DataFrame(rows).sort_values("dis_q")


def fmt_s(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.2f}"


def fmt_pct(x) -> str:
    return f"{100.0 * float(x):.2f}%"


def write_report(payload: dict) -> Path:
    sub = payload["submitting_verify"]
    sm = payload["rank_summary"]
    vm = payload["val_summary"]
    path = REP / "submitting_check_report.md"
    lines = []
    a = lines.append
    a("# Submitting / ranking diagnostic — frozen E16-A")
    a("")
    a("**Not a leaderboard submission.** Ranking DEP has no taxi labels; RMSE on ranking cannot be computed.")
    a("")
    a("## Verify submitting.parquet")
    a("")
    a(f"- rows: **{sub['n']:,}** (expected 344,841)")
    a(f"- `MVT_ID_mvt` unique: **{sub['unique_ok']}** ({sub['n_unique_mvt_id']:,})")
    a(f"- schema: `{sub['columns']}` — **{'exact' if sub['schema_ok'] else 'MISMATCH'}**")
    a(f"- `TAXITIME_SEC_mvt` all null: **{sub['target_all_null']}** ({sub['taxitime_null']:,} null / {sub['taxitime_non_null']:,} non-null)")
    a("")
    a("`submitting.parquet` is a 2-column template. Covariates for those IDs are the ranking DEP rows")
    a("(Jan+Jul **2026**). Ranking is used only as the prediction-time information set. No ranking")
    a("row entered OLS, `geo_mean`, hour baselines, or LightGBM.")
    a("")
    a("## Model")
    a("")
    a("```")
    a("P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)   # fit on training only")
    a("y_hat = P_cal + LGB_L2_residual(E12 matrix + E16-A disruption cols)")
    a("if unmatched and airport==LIRF: y_hat = MVT - SCHED")
    a("```")
    a("")
    a("Submitting/ranking scores use a model fit on **all 12 training months** (2025).")
    a("Jan+Jul 2025 comparison predictions use the E16-A holdout fit (train excludes Jan+Jul),")
    a(f"overall RMSE **{fmt_s(payload['val_rmse_overall'])}** (journal 376.04).")
    a("")
    a("## Ranking population")
    a("")
    rm = payload["rank_meta"]
    a(f"- DEP rows scored: {rm['ranking_dep']:,}")
    a(f"- ARR context rows: {rm['ranking_arr']:,}")
    a(f"- time: {rm['tmin']} → {rm['tmax']}")
    a(f"- unmatched: {rm['unmatched']:,} ({100*rm['unmatched']/rm['ranking_dep']:.2f}%)")
    a(f"- LIRF unmatched fallback candidates: {rm['lirf_unmatched']:,}")
    a("")
    a("Months are January and July 2026 only — the same calendar months as the primary 2025 holdout.")
    a("")
    a("## Current OpenAir model predictions on submitting.parquet")
    a("")
    a(
        f"**n={sm['n']:,}  mean={sm['mean']:.1f}s  median={sm['median']:.1f}s  "
        f"std={sm['std']:.1f}s  min={sm['min']:.1f}  max={sm['max']:.1f}  "
        f"fallback={payload['n_fallback']:,} ({100*payload['n_fallback']/sm['n']:.2f}%)**"
    )
    a("")
    a("| | Ranking 2026 (scored) | Jan+Jul 2025 val preds |")
    a("|---|---:|---:|")
    for k, lab in [
        ("n", "N"),
        ("min", "min"),
        ("p1", "P1"),
        ("p5", "P5"),
        ("p25", "P25"),
        ("p50", "P50 / median"),
        ("p75", "P75"),
        ("p95", "P95"),
        ("p99", "P99"),
        ("max", "max"),
        ("mean", "mean"),
        ("std", "std"),
    ]:
        a(f"| {lab} | {fmt_s(sm[k]) if k != 'n' else f'{sm[k]:,}'} | {fmt_s(vm[k]) if k != 'n' else f'{vm[k]:,}'} |")
    a("")
    a("| Bin | Ranking n | Ranking % | Jan+Jul 2025 val % |")
    a("|---|---:|---:|---:|")
    for lab, key in [
        ("<5 min", "lt_5m"),
        ("<10 min", "lt_10m"),
        ("10–20 min", "10_20m"),
        ("20–30 min", "20_30m"),
        ("30–60 min", "30_60m"),
        (">60 min", "gt_60m"),
    ]:
        nk = "n_" + key if not key.startswith("lt") else ("n_lt_5m" if key == "lt_5m" else "n_lt_10m")
        if key == "10_20m":
            nk = "n_10_20m"
        elif key == "20_30m":
            nk = "n_20_30m"
        elif key == "30_60m":
            nk = "n_30_60m"
        elif key == "gt_60m":
            nk = "n_gt_60m"
        a(f"| {lab} | {sm[nk]:,} | {fmt_pct(sm[key])} | {fmt_pct(vm[key])} |")
    a("")
    a(f"LIRF unmatched fallback used on **{payload['n_fallback']:,}** rows ({fmt_pct(payload['n_fallback']/sm['n'])}).")
    a(f"Jan+Jul 2025 val fallback: {payload['n_fallback_val']:,} ({fmt_pct(payload['n_fallback_val']/vm['n'])}).")
    a("")
    a("## Sanity")
    a("")
    a(f"- negative predictions: ranking {sm['n_negative']:,} / val {vm['n_negative']:,}")
    a(f"- predictions >4 h: ranking {sm['n_gt_4h']:,} / val {vm['n_gt_4h']:,}")
    a(f"- predictions >24 h: ranking {sm['n_gt_24h']:,} / val {vm['n_gt_24h']:,}")
    a("")
    a(payload["sanity_narrative"])
    a("")
    a("## Airport means")
    a("")
    a("| Airport | Rank n | Rank mean | Rank median | Rank >30% | Val n | Val mean | Val median | Val >30% | Δ mean |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    apt = payload["airport_compare"]
    for _, r in apt.iterrows():
        a(
            f"| {r['airport']} | {int(r['n_rank']):,} | {r['mean_rank']:.1f} | {r['median_rank']:.1f} | "
            f"{100*r['frac_gt30_rank']:.2f}% | {int(r['n_val']):,} | {r['mean_val']:.1f} | "
            f"{r['median_val']:.1f} | {100*r['frac_gt30_val']:.2f}% | {r['d_mean']:+.1f} |"
        )
    a("")
    a("## Disruption-state quantiles (ranking predictions)")
    a("")
    a("| Q | N | Mean dis | Mean pred | Median pred | >30% | >60% |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in payload["dis_q_rank"].iterrows():
        a(
            f"| {r['dis_q']} | {int(r['n']):,} | {r['mean_dis']:.0f} | {r['mean_pred']:.1f} | "
            f"{r['median_pred']:.1f} | {100*r['frac_gt30']:.2f}% | {100*r['frac_gt60']:.2f}% |"
        )
    a("")
    a("## Verdict")
    a("")
    a(payload["verdict"])
    a("")
    a("## Artifacts")
    a("")
    a("- `analysis/submitting_check/tables/submitting_predictions.parquet`")
    a("- `analysis/submitting_check/tables/submitting_predictions.csv` (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`)")
    a("- `analysis/submitting_check/tables/janjul_val_predictions.parquet`")
    a("- `analysis/submitting_check/figures/`")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  report {path}")
    return path


def main():
    t0 = datetime.now(timezone.utc)
    log("=== submitting/ranking diagnostic (E16-A, training fits only) ===")
    sub_info = verify_submitting()

    log("load ranking prediction-time frames (no DEP BLOCK/TAXITIME)...")
    score_dep, score_arr_t, score_arr_d, rank_meta = load_ranking_score()
    sub_ids = pl.read_parquet(SUBMIT_PATH, columns=["MVT_ID_mvt"])
    n_join = sub_ids.join(score_dep.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="inner").height
    log(f"  submitting IDs found in ranking DEP: {n_join:,}/{sub_info['n']:,}")
    if n_join != sub_info["n"]:
        raise SystemExit("submitting IDs do not fully match ranking DEP")

    log("featurize training_*.parquet (fits live here)...")
    train_raw = load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    train_arr = load_arr()
    train_arr_d = load_arr_delay()
    train_feat = featurize(train_raw, train_arr, train_arr_d)
    log(f"  training DEP featurized {train_feat.height:,}")

    log("featurize ranking DEP (causal within 2026 ranking stream only)...")
    score_feat = featurize(score_dep, score_arr_t, score_arr_d)
    log(f"  ranking DEP featurized {score_feat.height:,}")

    log("fit E16-A on ALL training months, score ranking...")
    tabs_all = geometry_tables(train_feat)
    tr_all = attach_geometry(train_feat, tabs_all, 30)
    sc = attach_geometry(score_feat, tabs_all, 30)
    tr_all = attach_hour_baseline(tr_all, tr_all)
    sc = attach_hour_baseline(tr_all, sc)
    p_cal_s, pred_s, _ = fit_residual(tr_all, sc, extra=MODEL_DISRUPT_COLS)
    log(f"  ranking preds finite={np.isfinite(pred_s).sum():,} mean={float(np.nanmean(pred_s)):.1f}")

    um_s = sc["unmatched"].to_numpy()
    ap_s = sc["airport"].to_numpy()
    ms_s = sc["mvt_sched"].to_numpy().astype(float)
    fb_s = um_s & (ap_s == "LIRF") & np.isfinite(ms_s)
    tn_s = sc["type_null"].to_numpy().astype(bool)

    log("Jan+Jul 2025 holdout E16-A (distribution comparison + RMSE check)...")
    tr0, va0 = split_by_months(train_feat, [1, 7])
    tabs_h = geometry_tables(tr0)
    tr_h = attach_geometry(tr0, tabs_h, 30)
    va_h = attach_geometry(va0, tabs_h, 30)
    tr_h = attach_hour_baseline(tr_h, tr_h)
    va_h = attach_hour_baseline(tr_h, va_h)
    _, pred_v, _ = fit_residual(tr_h, va_h, extra=MODEL_DISRUPT_COLS)
    y_v = va_h["y"].to_numpy().astype(np.float64)
    ok = np.isfinite(y_v) & np.isfinite(pred_v)
    val_rmse = float(np.sqrt(np.mean((y_v[ok] - pred_v[ok]) ** 2)))
    log(f"  Jan+Jul overall RMSE={val_rmse:.2f} (journal 376.04)")
    um_v = va_h["unmatched"].to_numpy()
    ap_v = va_h["airport"].to_numpy()
    ms_v = va_h["mvt_sched"].to_numpy().astype(float)
    fb_v = um_v & (ap_v == "LIRF") & np.isfinite(ms_v)

    sm = pred_summary(pred_s, "ranking_2026")
    vm = pred_summary(pred_v, "janjul_2025_val")
    log(f"  rank mean={sm['mean']:.1f} med={sm['median']:.1f} p95={sm['p95']:.1f} >30%={100*sm['30_60m']+100*sm['gt_60m']:.2f}")
    log(f"  val  mean={vm['mean']:.1f} med={vm['median']:.1f} p95={vm['p95']:.1f}")

    apt_s = airport_pred_table(ap_s, pred_s)
    apt_v = airport_pred_table(ap_v, pred_v)
    apt = apt_s.merge(apt_v, on="airport", suffixes=("_rank", "_val"))
    apt["d_mean"] = apt["mean_rank"] - apt["mean_val"]
    savecsv("airport_prediction_compare.csv", apt)
    savecsv("prediction_summary.csv", pd.DataFrame([sm, vm]))

    dis_s = sc["dis_state_30m"].to_numpy().astype(float)
    dis_v = va_h["dis_state_30m"].to_numpy().astype(float)
    dq_s = dis_q_table(dis_s, pred_s)
    dq_v = dis_q_table(dis_v, pred_v)
    savecsv("dis_state_quantile_ranking.csv", dq_s)
    savecsv("dis_state_quantile_janjul_val.csv", dq_v)

    # align submitting order
    pred_pl = sc.select(
        "MVT_ID_mvt", "airport", "month", "mvt_sched", "aobt_eobt", "mvt_aobt"
    ).with_columns(
        pl.Series("TAXITIME_SEC_mvt", pred_s),
        pl.Series("p_cal", p_cal_s),
        pl.Series("fallback_lirf_unmatched", fb_s.astype(np.int8)),
        pl.Series("unmatched", um_s.astype(np.int8)),
        pl.Series("type_null", tn_s.astype(np.int8)),
        pl.Series("dis_state_30m", dis_s),
    )
    out = sub_ids.join(pred_pl, on="MVT_ID_mvt", how="left")
    if int(out["TAXITIME_SEC_mvt"].null_count()) != 0:
        raise SystemExit("missing predictions after join to submitting IDs")
    out.write_parquet(TAB / "submitting_predictions.parquet")
    out.select("MVT_ID_mvt", "TAXITIME_SEC_mvt").write_csv(TAB / "submitting_predictions.csv")
    log(f"  wrote {TAB / 'submitting_predictions.parquet'}")

    va_h.select("MVT_ID_mvt", "airport", "month", "y", "unmatched").with_columns(
        pl.Series("pred", pred_v),
        pl.Series("fallback_lirf_unmatched", fb_v.astype(np.int8)),
        pl.Series("dis_state_30m", dis_v),
    ).write_parquet(TAB / "janjul_val_predictions.parquet")

    # figures
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bins = np.linspace(0, 3600, 73)
    ax.hist(pred_v[np.isfinite(pred_v)], bins=bins, density=True, alpha=0.5, label="Jan+Jul 2025 val pred", color="#4C72B0")
    ax.hist(pred_s[np.isfinite(pred_s)], bins=bins, density=True, alpha=0.5, label="Ranking 2026 pred", color="#DD8452")
    ax.set_xlabel("predicted taxi (s)")
    ax.set_ylabel("density")
    ax.set_title("E16-A prediction distribution")
    ax.legend()
    savefig("pred_hist_overlay.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labs = ["<5m", "<10m", "10–20m", "20–30m", "30–60m", ">60m"]
    keys = ["lt_5m", "lt_10m", "10_20m", "20_30m", "30_60m", "gt_60m"]
    x = np.arange(len(labs))
    ax.bar(x - 0.18, [100 * vm[k] for k in keys], 0.36, label="Jan+Jul 2025 val", color="#4C72B0")
    ax.bar(x + 0.18, [100 * sm[k] for k in keys], 0.36, label="Ranking 2026", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("% of predictions")
    ax.set_title("Predicted taxi bins")
    ax.legend()
    savefig("pred_bin_share.png")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(apt))
    ax.bar(x - 0.18, apt["mean_val"], 0.36, label="Jan+Jul 2025 val", color="#4C72B0")
    ax.bar(x + 0.18, apt["mean_rank"], 0.36, label="Ranking 2026", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(apt["airport"], rotation=0)
    ax.set_ylabel("mean prediction (s)")
    ax.set_title("Mean prediction by airport")
    ax.legend()
    savefig("pred_airport_mean.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    dq_plot = dq_s[dq_s["dis_q"].astype(str).str.startswith("Q")]
    ax.bar(dq_plot["dis_q"], dq_plot["mean_pred"], color="#DD8452")
    ax.set_ylabel("mean prediction (s)")
    ax.set_title("Ranking 2026 mean pred by dis_state_30m quantile")
    savefig("pred_by_dis_q_ranking.png")

    # sanity narrative
    d_mean = sm["mean"] - vm["mean"]
    d_med = sm["median"] - vm["median"]
    d_gt30 = (sm["30_60m"] + sm["gt_60m"]) - (vm["30_60m"] + vm["gt_60m"])
    max_airport_shift = float(apt["d_mean"].abs().max())
    worst_ap = str(apt.loc[apt["d_mean"].abs().idxmax(), "airport"])
    looks_ok = (
        sm["n_negative"] == 0
        and sm["n_gt_24h"] == 0
        and abs(d_mean) < 120
        and abs(d_gt30) < 0.05
        and sm["gt_60m"] < 0.02
    )
    sanity = (
        f"Ranking predictions are {d_mean:+.1f} s mean / {d_med:+.1f} s median vs Jan+Jul 2025 val preds. "
        f"Share predicted ≥30 min differs by {100*d_gt30:+.2f} pp. "
        f"Largest airport mean shift: {worst_ap} {max_airport_shift:.1f} s. "
        f"Negatives={sm['n_negative']}, >24h={sm['n_gt_24h']}, >4h={sm['n_gt_4h']}."
    )
    if looks_ok:
        verdict = (
            "The ranking/submitting prediction distribution looks **reasonable** relative to Jan+Jul 2025 "
            "validation predictions: same order of magnitude, no negative or 24 h bombs in the mass, "
            f"fallback rate similar ({100*int(fb_s.sum())/sm['n']:.2f}% vs {100*int(fb_v.sum())/vm['n']:.2f}%). "
            "RMSE on ranking is unknown (labels blank). Do not treat this as a leaderboard score. "
            "The model was not changed."
        )
    else:
        verdict = (
            "Distribution checks flagged something unusual (negatives, extreme tails, or a large shift "
            "vs Jan+Jul 2025 val preds). See tables. The model was not changed."
        )

    payload = {
        "submitting_verify": sub_info,
        "rank_meta": rank_meta,
        "rank_summary": sm,
        "val_summary": vm,
        "val_rmse_overall": val_rmse,
        "n_fallback": int(fb_s.sum()),
        "n_fallback_val": int(fb_v.sum()),
        "n_type_null": int(tn_s.sum()),
        "airport_compare": apt,
        "dis_q_rank": dq_s,
        "sanity_narrative": sanity,
        "verdict": verdict,
        "looks_reasonable": looks_ok,
    }
    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "submitting_verify": sub_info,
        "rank_meta": rank_meta,
        "rank_summary": sm,
        "val_summary": vm,
        "val_rmse_overall": val_rmse,
        "n_fallback": int(fb_s.sum()),
        "n_fallback_val": int(fb_v.sum()),
        "looks_reasonable": looks_ok,
        "verdict": verdict,
        "sanity_narrative": sanity,
    }
    (TAB / "findings.json").write_text(json.dumps(findings, indent=2, default=json_conv), encoding="utf-8")
    write_report(payload)
    shutil.copy2(Path(__file__).resolve(), OUT / "run_submitting_check.py")
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log(f"done in {elapsed/60:.1f} min")
    log(
        f"Current OpenAir model predictions on submitting.parquet: "
        f"n={sm['n']:,} mean={sm['mean']:.1f}s median={sm['median']:.1f}s "
        f"P5={sm['p5']:.0f} P95={sm['p95']:.0f} fallback={int(fb_s.sum()):,} "
        f"reasonable={looks_ok}"
    )


if __name__ == "__main__":
    main()
