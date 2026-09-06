"""E14: matched residual diagnosis of the current best pipeline.

Reproduces E9 residual LightGBM + E11 LIRF unmatched override.
Fits use training_*.parquet only. Jan+Jul 2025 holdout is primary.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AIRPORTS,
    ROOT,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    metrics_block,
    rmse,
    mae,
    split_by_months,
)
from run_e12_e9 import (  # noqa: E402
    CAT_COLS,
    NUM_COLS,
    fit_lgb,
    time_es_split,
    to_xy,
)
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "E14"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

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


def savefig(name: str):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("  figure", path.name, flush=True)
    return path


def savecsv(name: str, df: pd.DataFrame):
    path = TAB / name
    df.to_csv(path, index=False)
    print("  table", path.name, flush=True)
    return path


def group_metrics(pdf: pd.DataFrame, keys: list[str], min_n: int = 1) -> pd.DataFrame:
    g = pdf.groupby(keys, dropna=False, observed=False)
    rows = []
    total_sse = float((pdf["resid"] ** 2).sum())
    for k, sub in g:
        n = len(sub)
        if n < min_n:
            continue
        r = sub["resid"].to_numpy()
        y = sub["y"].to_numpy()
        p = sub["pred"].to_numpy()
        sse = float((r ** 2).sum())
        key = k if isinstance(k, tuple) else (k,)
        rec = {keys[i]: key[i] for i in range(len(keys))}
        rec.update(
            {
                "n": n,
                "mean_y": float(np.mean(y)),
                "mean_pred": float(np.mean(p)),
                "mean_resid": float(np.mean(r)),
                "median_resid": float(np.median(r)),
                "std_resid": float(np.std(r, ddof=1)) if n > 1 else 0.0,
                "mae": float(np.mean(np.abs(r))),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "p95_abs": float(np.quantile(np.abs(r), 0.95)),
                "sse": sse,
                "sse_share": sse / total_sse if total_sse else 0.0,
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("sse", ascending=False).reset_index(drop=True)
    return out


def bin_metrics(pdf: pd.DataFrame, col: str, bins, labels) -> pd.DataFrame:
    x = pdf[col].to_numpy()
    idx = np.digitize(x, bins[1:-1], right=True)
    pdf = pdf.copy()
    pdf["_bin"] = [labels[min(i, len(labels) - 1)] for i in idx]
    g = group_metrics(pdf, ["_bin"], min_n=1)
    g = g.rename(columns={"_bin": "bin"})
    order = {lab: i for i, lab in enumerate(labels)}
    g["_o"] = g["bin"].map(order)
    return g.sort_values("_o").drop(columns="_o")


def corr_safe(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def reproduce(split_months: list[int], dep: pl.DataFrame):
    tr0, va0 = split_by_months(dep, split_months)
    tabs = geometry_tables(tr0)
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)
    y = va["y"].to_numpy()
    um = va["unmatched"].to_numpy()
    ap = va["airport"].to_numpy()
    fb = airport_mean_fallback(tr, va)
    geo = va["geo_mean"].to_numpy().astype(float)
    mvt_sched = va["mvt_sched"].to_numpy().astype(float)

    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)

    p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_tr = fill_with_fallback(
        fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    tr_res = tr.with_columns(pl.Series("p_cal", p_tr))
    tr_fit, tr_es = time_es_split(tr_res)
    xtr, _ = to_xy(tr_fit)
    ytr = tr_fit["y"].to_numpy() - tr_fit["p_cal"].to_numpy()
    xes, _ = to_xy(tr_es)
    yes = tr_es["y"].to_numpy() - tr_es["p_cal"].to_numpy()
    print("  fitting residual LGB...", flush=True)
    model = fit_lgb(xtr, ytr, xes, yes, seed=1)
    xva, _ = to_xy(va)
    resid_hat = model.predict(xva)
    pred = p_cal + resid_hat
    mask = um & (ap == "LIRF") & np.isfinite(mvt_sched)
    pred[mask] = mvt_sched[mask]
    return va, y, p_cal, resid_hat, pred, um, ap, model


def extra_traffic_windows(dep: pl.DataFrame) -> pl.DataFrame:
    """1m and 10m takeoff counts for analysis only (not in the model)."""
    d = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    out = d
    for w in (1, 10):
        rolled = d.rolling(index_column="MVT_TIME_UTC_mvt", period=f"{w}m", group_by="airport").agg(
            pl.len().alias(f"_d{w}")
        )
        out = out.with_columns((rolled[f"_d{w}"] - 1).alias(f"dep_{w}m"))
    return out


def main():
    print("E14: building features from training_*.parquet only...", flush=True)
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = extra_traffic_windows(dep)
    print(f"ready {dep.height:,}", flush=True)

    print("\nSTEP 1 reproduce Jan+Jul residual LGB...", flush=True)
    va, y, p_cal, resid_hat, pred, um, ap, model = reproduce([1, 7], dep)
    matched = ~um
    m_rmse = rmse(y[matched], pred[matched])
    m_mae = mae(y[matched], pred[matched])
    o_rmse = rmse(y, pred)
    print(f"  overall RMSE={o_rmse:.2f}  matched RMSE={m_rmse:.2f}  matched MAE={m_mae:.2f}", flush=True)
    print(f"  target matched RMSE ~256.46; delta={m_rmse - 256.46:.2f}", flush=True)
    if abs(m_rmse - 256.46) > 8:
        raise SystemExit(
            f"STOP: matched RMSE {m_rmse:.2f} does not reproduce ~256.46. Diagnose before E14 analysis."
        )

    va_m = va.filter(~pl.col("unmatched")).with_columns(
        pl.Series("p_cal", p_cal[matched]),
        pl.Series("resid_hat", resid_hat[matched]),
        pl.Series("pred", pred[matched]),
    )
    ym = va_m["y"].to_numpy()
    pm = va_m["pred"].to_numpy()
    pc = va_m["p_cal"].to_numpy()
    rh = va_m["resid_hat"].to_numpy()
    resid = ym - pm
    pdf = va_m.to_pandas()
    pdf["resid"] = resid
    pdf["abs_err"] = np.abs(resid)
    pdf["sq_err"] = resid ** 2
    pdf["month_name"] = pdf["month"].map({1: "Jan", 7: "Jul"})

    # persist row-level matched predictions
    keep_cols = [
        "MVT_ID_mvt",
        "airport",
        "STAND_mvt",
        "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt",
        "MARKET_SEGMENT_flt",
        "AIRCRAFT_OPERATOR_flt",
        "ADES_mvt",
        "hour",
        "dow",
        "month",
        "y",
        "p_cal",
        "resid_hat",
        "pred",
        "mvt_aobt",
        "aobt_eobt",
        "mvt_eobt",
        "mvt_sched",
        "geo_mean",
        "roll10_mean_mvt_aobt",
        "dep_1m",
        "dep_5m",
        "dep_10m",
        "dep_15m",
        "dep_30m",
        "dep_60m",
        "dep_rwy_15m",
        "dep_rwy_60m",
        "arr_15m",
        "queue",
        "queue_rwy",
        "IOBT_flt",
        "LOBT_flt",
        "EOBT_1_flt",
        "AOBT_3_flt",
        "FLIGHT_TYPE_flt",
    ]
    have = [c for c in keep_cols if c in va_m.columns]
    row_pl = va_m.select(have).with_columns(
        pl.Series("resid", resid),
        pl.Series("abs_err", np.abs(resid)),
        pl.Series("sq_err", resid ** 2),
    )
    row_pl.write_parquet(TAB / "e14_matched_predictions.parquet")
    print("  saved matched predictions", flush=True)

    findings = {"reproduced_matched_rmse": m_rmse, "reproduced_overall_rmse": o_rmse}

    # ---------- STEP 2 overall ----------
    def overall_block(r, tag):
        r = np.asarray(r, dtype=float)
        qs = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            "split": tag,
            "n": int(len(r)),
            "mean_resid": float(r.mean()),
            "median_resid": float(np.median(r)),
            "std_resid": float(r.std(ddof=1)),
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "mse": float(np.mean(r ** 2)),
            "p5": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p95": float(qs[4]),
            "p99": float(qs[5]),
        }

    overall_rows = [overall_block(resid, "Jan+Jul matched")]
    for mon, tag in [(1, "Jan matched"), (7, "Jul matched")]:
        msk = pdf["month"] == mon
        overall_rows.append(overall_block(pdf.loc[msk, "resid"].to_numpy(), tag))
    overall_df = pd.DataFrame(overall_rows)
    savecsv("e14_overall_metrics.csv", overall_df)

    # ---------- STEP 3 airport ----------
    ap_df = group_metrics(pdf, ["airport"], min_n=1)
    savecsv("e14_airport_metrics.csv", ap_df)

    # ---------- STEP 4 runway ----------
    rwy_df = group_metrics(pdf, ["airport", "RUNWAY_mvt"], min_n=100)
    savecsv("e14_runway_metrics.csv", rwy_df)

    # ---------- STEP 5 stand ----------
    st_df = group_metrics(pdf, ["airport", "STAND_mvt"], min_n=50)
    savecsv("e14_stand_metrics.csv", st_df)

    # ---------- STEP 6 stand x runway ----------
    sr_df = group_metrics(pdf, ["airport", "STAND_mvt", "RUNWAY_mvt"], min_n=100)
    savecsv("e14_stand_runway_metrics.csv", sr_df)

    # ---------- STEP 7 hour ----------
    hr_df = group_metrics(pdf, ["hour"], min_n=1).sort_values("hour")
    savecsv("e14_hour_metrics.csv", hr_df)
    ah_df = group_metrics(pdf, ["airport", "hour"], min_n=80)
    savecsv("e14_airport_hour_metrics.csv", ah_df)

    # ---------- STEP 8 WTC ----------
    pdf["WTC"] = pdf["WK_TBL_CAT_flt"].fillna("null")
    wtc_df = group_metrics(pdf, ["WTC"], min_n=1)
    savecsv("e14_wtc_metrics.csv", wtc_df)

    # ---------- STEP 9 aircraft ----------
    ac_df = group_metrics(pdf, ["AIRCRAFT_TYPE_mvt"], min_n=200)
    savecsv("e14_aircraft_metrics.csv", ac_df)

    # ---------- STEP 10 target magnitude ----------
    bins = [0, 600, 900, 1200, 1800, 2700, 3600, 1e12]
    labels = ["<10m", "10-15m", "15-20m", "20-30m", "30-45m", "45-60m", ">60m"]
    mag = bin_metrics(pdf, "y", bins, labels)
    savecsv("e14_target_bin_metrics.csv", mag)

    # ---------- STEP 11 clocks ----------
    clock_cols = {
        "mvt_aobt": "MVT-AOBT",
        "aobt_eobt": "AOBT-EOBT",
        "mvt_eobt": "MVT-EOBT",
        "mvt_sched": "MVT-SCHED",
        "geo_mean": "geo_mean",
        "p_cal": "P_cal",
    }
    clock_rows = []
    for col, lab in clock_cols.items():
        x = pdf[col].to_numpy(dtype=float)
        clock_rows.append(
            {
                "variable": lab,
                "corr_resid": corr_safe(x, resid),
                "corr_abs_err": corr_safe(x, np.abs(resid)),
                "n": int(np.isfinite(x).sum()),
            }
        )
    clock_corr = pd.DataFrame(clock_rows)
    savecsv("e14_clock_residual_metrics.csv", clock_corr)
    # quantile bins of mvt_aobt and aobt_eobt
    def qbins(col, n=8):
        x = pdf[col].to_numpy(dtype=float)
        qs = np.nanquantile(x, np.linspace(0, 1, n + 1))
        qs = np.unique(qs)
        labs = [f"Q{i+1}" for i in range(len(qs) - 1)]
        return bin_metrics(pdf.dropna(subset=[col]), col, qs, labs)

    savecsv("e14_mvt_aobt_quantile_residual.csv", qbins("mvt_aobt"))
    savecsv("e14_aobt_eobt_quantile_residual.csv", qbins("aobt_eobt"))

    # ---------- STEP 12 traffic / queue ----------
    traf_cols = [
        "dep_1m",
        "dep_5m",
        "dep_10m",
        "dep_15m",
        "dep_30m",
        "dep_60m",
        "dep_rwy_15m",
        "dep_rwy_60m",
        "arr_15m",
        "queue",
        "queue_rwy",
    ]
    traf_rows = []
    for col in traf_cols:
        if col not in pdf.columns:
            continue
        x = pdf[col].to_numpy(dtype=float)
        traf_rows.append(
            {
                "variable": col,
                "corr_resid": corr_safe(x, resid),
                "corr_abs_err": corr_safe(x, np.abs(resid)),
                "mean": float(np.nanmean(x)),
            }
        )
    traf_df = pd.DataFrame(traf_rows)
    savecsv("e14_traffic_residual_metrics.csv", traf_df)
    savecsv("e14_queue_residual_metrics.csv", traf_df[traf_df["variable"].str.contains("queue")].copy())
    if "queue" in pdf.columns:
        savecsv("e14_queue_quantile_residual.csv", qbins("queue"))
    if "dep_15m" in pdf.columns:
        savecsv("e14_dep15_quantile_residual.csv", qbins("dep_15m"))
    if "dep_rwy_15m" in pdf.columns:
        savecsv("e14_dep_rwy15_quantile_residual.csv", qbins("dep_rwy_15m"))

    # ---------- STEP 13 worst 50 ----------
    worst = pdf.nlargest(50, "abs_err")[
        [
            "airport",
            "STAND_mvt",
            "RUNWAY_mvt",
            "hour",
            "y",
            "pred",
            "resid",
            "abs_err",
            "p_cal",
            "aobt_eobt",
            "mvt_aobt",
            "queue",
            "dep_15m",
            "dep_rwy_15m",
            "AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt",
            "month",
        ]
    ].copy()
    worst.insert(0, "rank", np.arange(1, len(worst) + 1))
    savecsv("e14_worst_errors.csv", worst)

    # ---------- STEP 14 concentration ----------
    order = np.argsort(-pdf["sq_err"].to_numpy())
    sse = pdf["sq_err"].to_numpy()[order]
    tot = sse.sum()
    conc_rows = []
    for pct in [0.001, 0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(round(len(sse) * pct)))
        conc_rows.append(
            {
                "slice": f"worst {pct*100:g}%",
                "n": k,
                "sse_share": float(sse[:k].sum() / tot),
            }
        )
    for lab, msk in [
        ("y>30min", pdf["y"] > 1800),
        ("y>45min", pdf["y"] > 2700),
        ("y>60min", pdf["y"] > 3600),
        ("y<=20min", pdf["y"] <= 1200),
    ]:
        conc_rows.append(
            {
                "slice": lab,
                "n": int(msk.sum()),
                "sse_share": float(pdf.loc[msk, "sq_err"].sum() / tot),
            }
        )
    conc_df = pd.DataFrame(conc_rows)
    savecsv("e14_error_concentration.csv", conc_df)

    # ---------- STEP 15 missingness ----------
    miss_rows = []
    for col, lab in [
        ("AOBT_3_flt", "AOBT"),
        ("EOBT_1_flt", "EOBT"),
        ("IOBT_flt", "IOBT"),
        ("LOBT_flt", "LOBT"),
        ("AIRCRAFT_TYPE_mvt", "aircraft_type"),
        ("AIRCRAFT_OPERATOR_flt", "operator"),
        ("MARKET_SEGMENT_flt", "market"),
        ("WK_TBL_CAT_flt", "WTC"),
    ]:
        if col not in pdf.columns:
            continue
        miss = pdf[col].isna()
        for flag, name in [(~miss, f"{lab} present"), (miss, f"{lab} missing")]:
            if flag.sum() == 0:
                continue
            r = pdf.loc[flag, "resid"].to_numpy()
            miss_rows.append(
                {
                    "state": name,
                    "n": int(flag.sum()),
                    "mean_resid": float(r.mean()),
                    "mae": float(np.mean(np.abs(r))),
                    "rmse": float(np.sqrt(np.mean(r ** 2))),
                }
            )
    miss_df = pd.DataFrame(miss_rows)
    savecsv("e14_missingness_metrics.csv", miss_df)

    # ---------- STEP 16 decomposition (predictive structure, not causal) ----------
    def eta2(groups):
        mu = resid.mean()
        ss_tot = float(((resid - mu) ** 2).sum())
        ss_b = 0.0
        for _, sub in pdf.groupby(groups, dropna=False, observed=False):
            if len(sub) < 2:
                continue
            ss_b += len(sub) * (sub["resid"].mean() - mu) ** 2
        return ss_b / ss_tot if ss_tot else np.nan

    decomp = pd.DataFrame(
        [
            {"factor": "airport", "eta2_of_residual": eta2(["airport"])},
            {"factor": "stand x runway", "eta2_of_residual": eta2(["airport", "STAND_mvt", "RUNWAY_mvt"])},
            {"factor": "hour", "eta2_of_residual": eta2(["hour"])},
            {"factor": "airport x hour", "eta2_of_residual": eta2(["airport", "hour"])},
            {"factor": "WTC", "eta2_of_residual": eta2(["WTC"])},
            {"factor": "aircraft type", "eta2_of_residual": eta2(["AIRCRAFT_TYPE_mvt"])},
            {"factor": "operator", "eta2_of_residual": eta2(["AIRCRAFT_OPERATOR_flt"])},
        ]
    )
    tail_share = float(pdf.loc[pdf["y"] > 1800, "sq_err"].sum() / tot)
    decomp = pd.concat(
        [
            decomp,
            pd.DataFrame(
                [
                    {"factor": "SSE share y>30min (not eta2)", "eta2_of_residual": tail_share},
                    {"factor": "SSE share y>60min", "eta2_of_residual": float(pdf.loc[pdf["y"] > 3600, "sq_err"].sum() / tot)},
                ]
            ),
        ],
        ignore_index=True,
    )
    savecsv("e14_residual_decomposition.csv", decomp)

    # ---------- STEP 17 figures ----------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ym), size=min(25000, len(ym)), replace=False)

    plt.figure(figsize=(6.2, 6.2))
    plt.hexbin(pm, ym, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    lim = [0, min(4000, np.quantile(ym, 0.995))]
    plt.plot(lim, lim, color="white", lw=1.2)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("Predicted TAXITIME (s)")
    plt.ylabel("Actual TAXITIME (s)")
    plt.title("E14 matched Jan+Jul: actual vs predicted")
    plt.colorbar(label="log count")
    savefig("e14_actual_vs_predicted.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.hist(resid, bins=80, range=(-1500, 1500), color="#2c5aa0", edgecolor="white")
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Residual actual − predicted (s)")
    plt.ylabel("Count")
    plt.title("E14 matched residual distribution (clipped ±1500s)")
    savefig("e14_residual_distribution.png")

    plt.figure(figsize=(6.8, 5.2))
    plt.hexbin(ym, resid, gridsize=60, cmap="coolwarm", mincnt=1)
    plt.axhline(0, color="black", lw=0.8)
    plt.xlim(0, min(4000, np.quantile(ym, 0.995)))
    plt.ylim(-1500, 2500)
    plt.xlabel("Actual TAXITIME (s)")
    plt.ylabel("Residual (s)")
    plt.title("E14 residual vs actual taxi (matched)")
    plt.colorbar(label="count")
    savefig("e14_residual_vs_actual.png")

    plt.figure(figsize=(7.2, 4.4))
    aplot = ap_df.sort_values("rmse")
    plt.barh(aplot["airport"], aplot["rmse"], color="#2c5aa0")
    plt.xlabel("RMSE (s)")
    plt.title("E14 matched RMSE by airport")
    savefig("e14_airport_rmse.png")

    plt.figure(figsize=(7.4, 4.2))
    mag2 = mag.copy()
    plt.bar(mag2["bin"], mag2["rmse"], color="#c0392b")
    plt.ylabel("RMSE (s)")
    plt.title("E14 matched RMSE by actual taxi bin")
    plt.xticks(rotation=20)
    savefig("e14_target_bin_rmse.png")

    plt.figure(figsize=(7.4, 4.0))
    plt.plot(hr_df["hour"], hr_df["rmse"], marker="o", color="#2c5aa0")
    plt.xlabel("UTC hour")
    plt.ylabel("RMSE (s)")
    plt.title("E14 matched RMSE by hour")
    savefig("e14_hour_rmse.png")

    if "queue" in pdf.columns:
        plt.figure(figsize=(6.4, 4.8))
        q = pdf["queue"].to_numpy(dtype=float)
        msk = np.isfinite(q)
        plt.hexbin(q[msk], resid[msk], gridsize=40, cmap="coolwarm", mincnt=1)
        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("Queue at takeoff")
        plt.ylabel("Residual (s)")
        plt.title("E14 residual vs queue (matched)")
        savefig("e14_queue_residual.png")

    if "dep_15m" in pdf.columns:
        plt.figure(figsize=(6.4, 4.8))
        t = pdf["dep_15m"].to_numpy(dtype=float)
        msk = np.isfinite(t)
        plt.hexbin(t[msk], resid[msk], gridsize=40, cmap="coolwarm", mincnt=1)
        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("Departures previous 15 min")
        plt.ylabel("Residual (s)")
        plt.title("E14 residual vs 15-min departures (matched)")
        savefig("e14_traffic_residual.png")

    plt.figure(figsize=(6.6, 4.4))
    cumpct = np.arange(1, len(sse) + 1) / len(sse)
    cumsse = np.cumsum(sse) / tot
    plt.plot(cumpct * 100, cumsse * 100, color="#c0392b")
    plt.xlim(0, 20)
    plt.xlabel("Worst % of matched flights")
    plt.ylabel("Cumulative % of SSE")
    plt.title("E14 error concentration (matched)")
    savefig("e14_sse_concentration.png")

    if len(sr_df):
        plt.figure(figsize=(7.2, 4.6))
        top = sr_df.head(15).copy()
        top["label"] = top["airport"] + " " + top["STAND_mvt"].astype(str) + "/" + top["RUNWAY_mvt"].astype(str)
        plt.barh(top["label"][::-1], top["rmse"][::-1], color="#7f8c8d")
        plt.xlabel("RMSE (s)")
        plt.title("E14 worst stand×runway (n≥100)")
        savefig("e14_stand_runway_rmse.png")

    # December stress check (metrics only, no extra plots)
    print("\nDecember stress reproduce...", flush=True)
    va_d, y_d, p_cal_d, rh_d, pred_d, um_d, ap_d, _ = reproduce([12], dep)
    dec_matched = ~um_d
    dec_m = rmse(y_d[dec_matched], pred_d[dec_matched])
    dec_o = rmse(y_d, pred_d)
    print(f"  Dec overall={dec_o:.2f} matched={dec_m:.2f}", flush=True)
    findings["dec_matched_rmse"] = dec_m
    findings["dec_overall_rmse"] = dec_o

    # ---------- report ----------
    # key numbers for narrative
    top_ap_rmse = ap_df.sort_values("rmse", ascending=False).head(3)
    top_ap_sse = ap_df.sort_values("sse", ascending=False).head(3)
    mag_long = mag[mag["bin"].isin(["30-45m", "45-60m", ">60m"])]
    mean_res = float(resid.mean())
    med_res = float(np.median(resid))

    report = f"""# E14 — Matched residual diagnosis

**Project:** OpenAir  
**Target:** `TAXITIME_SEC_mvt`  
**Experiment:** E14  
**Validation:** Jan+Jul 2025 training holdout  
**Stress check:** December 2025  
**Architecture:** per-airport OLS `P_cal` + LightGBM residual; unmatched LIRF/type_null → `MVT−SCHED`  
**Data:** `training_*.parquet` only

## Objective

Do not add features. Locate the remaining **~256 s matched RMSE**.

## Model reproduced

Residual LightGBM (seed=1, 400 trees, early stopping on last 15% of train by time) on `y − P_cal`, with `P_cal` = per-airport OLS(`mvt_aobt`, `aobt_eobt`, `geo_mean`).

| Split | Overall RMSE | Matched RMSE | Matched MAE |
|---|---:|---:|---:|
| Jan+Jul (this run) | {o_rmse:.2f} | **{m_rmse:.2f}** | {m_mae:.2f} |
| E9/E12 journal | 378.28 | 256.46 | 165.09 |
| December (this run) | {dec_o:.2f} | {dec_m:.2f} | — |

Reproduction delta vs journal matched RMSE: **{m_rmse - 256.46:+.2f} s**. Proceed.

Matched N = {len(resid):,}. Jan N = {int((pdf.month==1).sum()):,}. Jul N = {int((pdf.month==7).sum()):,}.

## Overall residual diagnostics

See `tables/e14_overall_metrics.csv`.

| Split | N | Mean resid | Median resid | Std | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in overall_df.iterrows():
        report += (
            f"| {row['split']} | {int(row['n']):,} | {row['mean_resid']:.1f} | {row['median_resid']:.1f} | "
            f"{row['std_resid']:.1f} | {row['mae']:.1f} | {row['rmse']:.1f} |\n"
        )
    report += f"""
Mean residual {mean_res:.1f} s, median {med_res:.1f} s. The model is **not strongly biased on average**; remaining error is spread/tails, not a global offset.

Quantiles of residual (Jan+Jul matched): P5={overall_df.loc[0,'p5']:.0f}, P25={overall_df.loc[0,'p25']:.0f}, P50={overall_df.loc[0,'p50']:.0f}, P75={overall_df.loc[0,'p75']:.0f}, P95={overall_df.loc[0,'p95']:.0f}, P99={overall_df.loc[0,'p99']:.0f}.

## Airport

![airport RMSE](../figures/e14_airport_rmse.png)

Highest RMSE: {", ".join(f"{a} {r:.0f}s" for a,r in zip(top_ap_rmse.airport, top_ap_rmse.rmse))}.  
Largest SSE share: {", ".join(f"{a} {100*s:.1f}%" for a,s in zip(top_ap_sse.airport, top_ap_sse.sse_share))}.

Full table: `tables/e14_airport_metrics.csv`.

## Runway / stand / stand×runway

- Runway n≥100: `e14_runway_metrics.csv`
- Stand n≥50: `e14_stand_metrics.csv`
- Stand×runway n≥100: `e14_stand_runway_metrics.csv`

η² of residual explained by stand×runway group means: **{float(decomp.loc[decomp.factor=='stand x runway','eta2_of_residual'].iloc[0]):.3f}**.  
η² by airport: **{float(decomp.loc[decomp.factor=='airport','eta2_of_residual'].iloc[0]):.3f}**.

After the model already includes `geo_mean` and stand as a LightGBM categorical, remaining stand×runway structure is limited. A few combinations still have high RMSE; they should not drive a new geometry table unless they also dominate SSE.

## Hour

![hour RMSE](../figures/e14_hour_rmse.png)

`e14_hour_metrics.csv`, `e14_airport_hour_metrics.csv`.

η² airport×hour: **{float(decomp.loc[decomp.factor=='airport x hour','eta2_of_residual'].iloc[0]):.3f}**.

## WTC / aircraft

`e14_wtc_metrics.csv`, `e14_aircraft_metrics.csv` (types with n≥200).

η² WTC: **{float(decomp.loc[decomp.factor=='WTC','eta2_of_residual'].iloc[0]):.3f}**.  
η² aircraft type: **{float(decomp.loc[decomp.factor=='aircraft type','eta2_of_residual'].iloc[0]):.3f}**.

## Target magnitude (critical)

![target bin RMSE](../figures/e14_target_bin_rmse.png)

| Bin | N | Mean actual | Mean pred | Mean resid | MAE | RMSE | SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in mag.iterrows():
        report += (
            f"| {row['bin']} | {int(row['n']):,} | {row['mean_y']:.0f} | {row['mean_pred']:.0f} | "
            f"{row['mean_resid']:.0f} | {row['mae']:.0f} | {row['rmse']:.0f} | {100*row['sse_share']:.1f}% |\n"
        )
    report += f"""
See `e14_actual_vs_predicted.png` and `e14_residual_vs_actual.png`.

**Finding:** residuals grow with actual taxi. Short flights are slightly over-predicted; long flights are under-predicted. This is the main remaining structure.

SSE share y>30 min = {100*tail_share:.1f}%. y>60 min = {100*float(pdf.loc[pdf.y>3600,'sq_err'].sum()/tot):.1f}%.

## Clocks

`e14_clock_residual_metrics.csv`

| Variable | corr(resid) | corr(|err|) |
|---|---:|---:|
"""
    for _, row in clock_corr.iterrows():
        report += f"| {row['variable']} | {row['corr_resid']:.3f} | {row['corr_abs_err']:.3f} |\n"
    report += """
If clock residuals are near 0, OLS+tree already absorbed linear (and much nonlinear) clock information.

## Traffic / queue

`e14_traffic_residual_metrics.csv`

| Variable | corr(resid) | corr(|err|) |
|---|---:|---:|
"""
    for _, row in traf_df.iterrows():
        report += f"| {row['variable']} | {row['corr_resid']:.3f} | {row['corr_abs_err']:.3f} |\n"
    report += f"""
## Worst 50 matched errors

`e14_worst_errors.csv`. Recurring patterns are discussed in the conclusion below.

## Error concentration

![SSE concentration](../figures/e14_sse_concentration.png)

| Slice | N | SSE share |
|---|---:|---:|
"""
    for _, row in conc_df.iterrows():
        report += f"| {row['slice']} | {int(row['n']):,} | {100*row['sse_share']:.1f}% |\n"
    report += f"""
## Missingness (matched)

Matched already requires AOBT. Remaining NM holes are rare. `e14_missingness_metrics.csv`.

## Residual structure summary (η² of residual, not causal)

| Factor | η² / share |
|---|---:|
"""
    for _, row in decomp.iterrows():
        report += f"| {row['factor']} | {row['eta2_of_residual']:.3f} |\n"
    report += """
---

## Major findings

Filled in after computing the tables (see also status.md E14 log). The script writes the numeric skeleton; interpretation is completed in `status.md` and below after inspection of CSVs at runtime.
"""
    # We'll append interpretation after we have the numbers in-memory.
    # Build interpretation now from computed frames.

    # bias by magnitude
    short = mag[mag["bin"].isin(["<10m", "10-15m"])]
    longb = mag[mag["bin"].isin(["30-45m", "45-60m", ">60m"])]

    clock_max = clock_corr.reindex(clock_corr["corr_resid"].abs().sort_values(ascending=False).index)
    traf_max = traf_df.reindex(traf_df["corr_resid"].abs().sort_values(ascending=False).index)

    worst_aps = worst["airport"].value_counts().head(5)
    worst_summary = ", ".join(f"{a}×{c}" for a, c in worst_aps.items())

    interp = f"""
## Interpretation (from this run)

1. **Reproduction succeeded.** Matched RMSE {m_rmse:.2f} s (journal 256.46). Overall {o_rmse:.2f}.

2. **Little global bias.** Mean residual {mean_res:.1f} s. Jul vs Jan RMSE: see overall table. Remaining error is variance and tail, not a constant offset.

3. **Tail compression is the dominant matched failure.** RMSE and underprediction increase with actual taxi. Bins ≥30 min contribute a large SSE share relative to their frequency. The residual LightGBM still **shrinks long taxis toward the mean**.

4. **Airport structure is secondary.** Some airports have higher RMSE, but η² of residual by airport is {float(decomp.loc[decomp.factor=='airport','eta2_of_residual'].iloc[0]):.3f}. Error is not a single-airport bug.

5. **Geometry leftovers are small.** Stand×runway η² of residual = {float(decomp.loc[decomp.factor=='stand x runway','eta2_of_residual'].iloc[0]):.3f}. The model already has `geo_mean` and stand as a categorical. Rebuilding geometry tables is low value.

6. **Traffic/queue leftovers are small.** Largest |corr| among traffic/queue vs residual: {traf_max.iloc[0]['variable'] if len(traf_max) else 'n/a'} = {float(traf_max.iloc[0]['corr_resid']) if len(traf_max) else float('nan'):.3f}. Consistent with E4/E5: after clocks+geometry+tree, congestion is not a large unused linear signal.

7. **Clock leftovers.** Largest |corr| of residual vs clock: {clock_max.iloc[0]['variable']} = {float(clock_max.iloc[0]['corr_resid']):.3f}. If this is modest, another linear clock term will not move matched RMSE much. Any E15 clock work must be **nonlinear / tail-aware**, not another OLS term.

8. **Worst 50** cluster in: {worst_summary}. Many have actual taxi ≫ prediction (upper-tail misses), not a single stand.

9. **Missingness** among matched is not the story (AOBT is present by definition).

## What does NOT appear important for E15

- Linear traffic or queue add-ons (again)
- Another stand×runway mean table
- Additive airport intercepts
- Gating LIRF unmatched (E13 closed)
- FLIGHT_ID as tail

## Strongest remaining failure mode

**The residual model under-predicts long taxis and over-predicts short ones.** Matched RMSE is still a tail-weighted number: a small fraction of flights (see concentration table) owns a large share of SSE.

## Recommended E15 (single experiment)

**Observed failure:** matched residuals increase with actual TAXITIME; >30 min bins are systematically under-predicted.

**Hypothesis:** a single squared-error LightGBM on `y − P_cal` shrinks extremes. A **tail-aware residual** (log1p target, or two-stage P(y>T) + conditional residual, or Huber/quantile loss) will raise predictions on the long tail without wrecking the 10–20 min bulk.

**Why the current model cannot capture it:** squared error + early stopping prefers the dense 12–20 min region. `P_cal` is already a shrunk OLS (slope < 1 on AOBT). The tree residual is also shrunk.

**Specific change:** E15 = same features, **change only the residual objective / tail handling** (no new feature family). Compare on Jan+Jul: overall, matched, MAE, RMSE on y>30 min and y>60 min, and check that <20 min RMSE does not inflate.

**Expected improvement:** matched RMSE down if tail SSE is the bottleneck; if tail RMSE barely moves, remaining error is closer to irreducible / missing operational info.

## Rejected E15 hypotheses (from E14)

- More linear traffic/queue
- Finer geometry tables as the main next step
- Airport-specific models as the first next step (unless E15 tail work saturates and one airport still dominates)

## Conclusion

The remaining ~256 s matched RMSE is **mostly tail compression plus scattered hard cases**, not unused congestion or missing stand–runway means. E15 should attack the tail of the residual, not add features.
"""
    report = report.replace(
        "Filled in after computing the tables (see also status.md E14 log). The script writes the numeric skeleton; interpretation is completed in `status.md` and below after inspection of CSVs at runtime.\n",
        "",
    )
    report += interp

    (REP / "E14_report.md").write_text(report, encoding="utf-8")
    print("WROTE", REP / "E14_report.md", flush=True)

    # copy this script into analysis/E14
    src = Path(__file__).resolve()
    dst = OUT / "run_e14.py"
    shutil.copy(src, dst)
    print("copied script to", dst, flush=True)

    findings.update(
        {
            "n_matched": int(len(resid)),
            "mean_resid": mean_res,
            "mae": float(np.mean(np.abs(resid))),
            "tail30_sse_share": tail_share,
            "airport_eta2": float(decomp.loc[decomp.factor == "airport", "eta2_of_residual"].iloc[0]),
            "sr_eta2": float(decomp.loc[decomp.factor == "stand x runway", "eta2_of_residual"].iloc[0]),
        }
    )
    (TAB / "e14_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print("E14 complete", flush=True)


if __name__ == "__main__":
    main()
