"""E18: non-LIRF unmatched specialist.

Hypothesis: after the LIRF MVT−SCHED override, remaining unmatched RMSE (~2240)
is a different DGP (EHAM shorter, LTFM/EGLL/LFPG heavier tails) currently scored
with matched geo_mean + a residual tree trained on 99% matched rows. E12 showed
that tree slightly *hurts* unmatched vs linear geo_mean.

Variants (training_*.parquet only; Jan+Jul primary; December stress):
  E18-0     exact E16-A reproduction (matched path frozen for A/B splices)
  E18-A-geo non-LIRF unmatched → P_cal (geo_mean); drop residual
  E18-A-mean  non-LIRF unmatched → train airport-unmatched mean
  E18-A-med   non-LIRF unmatched → train airport-unmatched median
  E18-H     hygiene refit: geo_mean on matched-only; drop LIRF-override rows
            from residual training (matched path may move)
  E18-B     E18-0 matched + LIRF override; non-LIRF unmatched = unmatched-mean
            + small L2 residual tree trained only on that slice

No ranking/submitting in any fit. No loss-function change.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    AIRPORTS,
    ROOT,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    mae,
    metrics_block,
    rmse,
    save_result,
    split_by_months,
)
from run_e12_e9 import CAT_COLS, NUM_COLS as BASE_NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    apply_override,
    attach_hour_baseline,
    fit_residual as fit_residual_e16a,
    load_arr_delay,
    override_mask,
    prepare_split as prepare_split_e16a,
    score_pred,
    to_xy,
)

OUT = ROOT / "analysis" / "E18"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
RES = HERE / "results" / "E18"
for d in (FIG, TAB, REP, RES, FIG / "dec"):
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

SEED = 1
BASELINE_OVERALL = 376.04
BASELINE_MATCHED = 253.82
BASELINE_UNMATCHED = 2235.88

U_NUM = [
    "mvt_sched",
    "geo_mean",
    "roll10_mean_mvt_aobt",
    "hour",
    "dow",
    "month",
    "dis_state_30m",
    "dis_frac20_30m",
    "dep_15m",
    "dep_rwy_15m",
    "queue",
]
U_CAT = [
    "airport",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "ADES_mvt",
    "flt_prefix",
]
VARIANTS = ["E18-0", "E18-A-geo", "E18-A-mean", "E18-A-med", "E18-H", "E18-B"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def savefig(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"  figure {path.name}")
    return path


def non_lirf_unmatched(um, ap) -> np.ndarray:
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    return um & (ap != "LIRF")


def lirf_unmatched(um, ap) -> np.ndarray:
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    return um & (ap == "LIRF")


def splice(base, replacement, mask) -> np.ndarray:
    out = np.asarray(base, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool) & np.isfinite(replacement)
    out[m] = np.asarray(replacement, dtype=np.float64)[m]
    return out


def add_prefix(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("FLIGHT_mvt").fill_null("NA").str.slice(0, 3).fill_null("NA").alias("flt_prefix")
    )


def _airport_unmatched_means(train: pl.DataFrame) -> dict[str, float]:
    out = {}
    for a in AIRPORTS:
        s = train.filter((pl.col("airport") == a) & pl.col("unmatched"))["y"].mean()
        out[a] = float(s) if s is not None and np.isfinite(s) else float("nan")
    return out


def airport_unmatched_stat(train: pl.DataFrame, val: pl.DataFrame, how: str) -> np.ndarray:
    """Train-split unmatched (any airport) mean/median by airport, mapped to val rows."""
    u = train.filter(pl.col("unmatched"))
    if how == "mean":
        g = u.group_by("airport").agg(pl.col("y").mean().alias("_u"))
        glob = float(u["y"].mean()) if u.height else float(train["y"].mean())
    elif how == "median":
        g = u.group_by("airport").agg(pl.col("y").median().alias("_u"))
        glob = float(u["y"].median()) if u.height else float(train["y"].median())
    else:
        raise ValueError(how)
    j = val.join(g, on="airport", how="left")
    p = j["_u"].to_numpy().astype(np.float64)
    return np.where(np.isfinite(p), p, glob)


def prepare_split(dep: pl.DataFrame, months: list[int], matched_geo: bool = False) -> dict:
    """Same joins as E16-A; optional matched-only geometry tables. No row-order change."""
    tr0, va0 = split_by_months(dep, months)
    src = tr0.filter(~pl.col("unmatched")) if matched_geo else tr0
    tabs = geometry_tables(src)
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)
    tr = attach_hour_baseline(tr, tr)
    va = attach_hour_baseline(tr, va)
    tr = add_prefix(tr)
    va = add_prefix(va)
    return {"tr": tr, "va": va}


def align_pred(src_va: pl.DataFrame, src_pred: np.ndarray, dst_va: pl.DataFrame) -> np.ndarray:
    """Map predictions onto dst_va row order via MVT_ID_mvt."""
    src = src_va.select("MVT_ID_mvt").with_columns(pl.Series("_p", np.asarray(src_pred, dtype=np.float64)))
    j = dst_va.select("MVT_ID_mvt").join(src, on="MVT_ID_mvt", how="left")
    p = j["_p"].to_numpy().astype(np.float64)
    if int(np.isnan(p).sum()) != 0:
        raise RuntimeError("align_pred: missing MVT_IDs")
    return p


def fit_residual(
    tr: pl.DataFrame,
    va: pl.DataFrame,
    extra: list[str],
    drop_override_from_train: bool = False,
):
    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    geo = va["geo_mean"].to_numpy().astype(float)
    fb = airport_mean_fallback(tr, va)
    p_cal = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)

    p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_tr = fill_with_fallback(
        fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    ov_tr = override_mask(
        tr["unmatched"].to_numpy(),
        tr["airport"].to_numpy(),
        tr["mvt_sched"].to_numpy().astype(float),
    )
    tr_res = tr.with_columns(pl.Series("p_cal", p_tr), pl.Series("_ov", ov_tr))
    if drop_override_from_train:
        n_drop = int(ov_tr.sum())
        tr_res = tr_res.filter(~pl.col("_ov"))
        log(f"    hygiene: dropped {n_drop:,} LIRF-override rows from residual train")
    tr_fit, tr_es = time_es_split(tr_res)
    xtr, _ = to_xy(tr_fit, extra)
    ytr = tr_fit["y"].to_numpy() - tr_fit["p_cal"].to_numpy()
    xes, _ = to_xy(tr_es, extra)
    yes = tr_es["y"].to_numpy() - tr_es["p_cal"].to_numpy()
    model = fit_lgb(xtr, ytr, xes, yes, seed=SEED)
    xva, _ = to_xy(va, extra)
    resid_hat = np.asarray(model.predict(xva), dtype=np.float64)
    pred = apply_override(
        p_cal + resid_hat,
        override_mask(
            va["unmatched"].to_numpy(),
            va["airport"].to_numpy(),
            va["mvt_sched"].to_numpy().astype(float),
        ),
        va["mvt_sched"].to_numpy().astype(float),
    )
    return p_cal, pred, model


def fit_unmatched_specialist(tr: pl.DataFrame, va: pl.DataFrame, p_u_tr: np.ndarray, p_u_va: np.ndarray):
    """L2 residual on non-LIRF unmatched only, around airport-unmatched mean."""
    tr_um = tr["unmatched"].to_numpy()
    tr_ap = tr["airport"].to_numpy()
    va_um = va["unmatched"].to_numpy()
    va_ap = va["airport"].to_numpy()
    tr_m = non_lirf_unmatched(tr_um, tr_ap)
    va_m = non_lirf_unmatched(va_um, va_ap)

    tr_u = tr.with_columns(pl.Series("p_u", p_u_tr), pl.Series("_keep", tr_m)).filter(pl.col("_keep"))
    if tr_u.height < 200:
        log(f"    specialist: only {tr_u.height} train rows; falling back to unmatched mean")
        return p_u_va.copy(), None, {"n_train": int(tr_u.height), "trees": 0}

    tr_fit, tr_es = time_es_split(tr_u, frac=0.15)
    cols = U_NUM + U_CAT + ["y", "p_u"]
    pdf_fit = tr_fit.select(cols).to_pandas()
    pdf_es = tr_es.select(cols).to_pandas()
    pdf_va = va.select(U_NUM + U_CAT).to_pandas()
    for c in U_CAT:
        cats = pd.Index(pdf_fit[c].astype("string").fillna("NA").unique()).union(
            pdf_es[c].astype("string").fillna("NA").unique()
        ).union(pdf_va[c].astype("string").fillna("NA").unique())
        pdf_fit[c] = pd.Categorical(pdf_fit[c].astype("string").fillna("NA"), categories=cats)
        pdf_es[c] = pd.Categorical(pdf_es[c].astype("string").fillna("NA"), categories=cats)
        pdf_va[c] = pd.Categorical(pdf_va[c].astype("string").fillna("NA"), categories=cats)
    xtr, xes, xva = pdf_fit[U_NUM + U_CAT], pdf_es[U_NUM + U_CAT], pdf_va[U_NUM + U_CAT]
    ytr = pdf_fit["y"].to_numpy(dtype=np.float64) - pdf_fit["p_u"].to_numpy(dtype=np.float64)
    yes = pdf_es["y"].to_numpy(dtype=np.float64) - pdf_es["p_u"].to_numpy(dtype=np.float64)

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xes, yes)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=U_CAT,
    )
    resid = np.asarray(model.predict(xva), dtype=np.float64)
    pred = p_u_va + resid
    trees = int(model.best_iteration_ or model.n_estimators)
    imp = sorted(
        zip(U_NUM + U_CAT, model.feature_importances_.tolist()),
        key=lambda t: -t[1],
    )
    log(f"    specialist n_train={tr_u.height:,} trees={trees} top={imp[:8]}")
    return pred, model, {"n_train": int(tr_u.height), "n_es": int(tr_es.height), "trees": trees, "importance": imp[:20]}


def slice_metrics(y, p, um, ap) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    m = metrics_block(y, p, um, ap)
    ok = np.isfinite(y) & np.isfinite(p)
    nlu = ok & non_lirf_unmatched(um, ap)
    lu = ok & lirf_unmatched(um, ap)
    m["rmse_non_lirf_unmatched"] = rmse(y[nlu], p[nlu]) if nlu.any() else float("nan")
    m["mae_non_lirf_unmatched"] = mae(y[nlu], p[nlu]) if nlu.any() else float("nan")
    m["n_non_lirf_unmatched"] = int(nlu.sum())
    m["rmse_lirf_unmatched"] = rmse(y[lu], p[lu]) if lu.any() else float("nan")
    m["n_lirf_unmatched"] = int(lu.sum())
    m["sse_overall"] = float(np.sum((y[ok] - p[ok]) ** 2)) if ok.any() else 0.0
    m["sse_unmatched"] = float(np.sum((y[ok & um] - p[ok & um]) ** 2)) if (ok & um).any() else 0.0
    m["sse_non_lirf_unmatched"] = float(np.sum((y[nlu] - p[nlu]) ** 2)) if nlu.any() else 0.0
    sc = score_pred(y, p, um)
    m["rmse_overall"] = sc["rmse_overall"]
    m["rmse_matched"] = sc["rmse_matched"]
    m["mae_matched"] = sc["mae_matched"]
    m["rmse_unmatched"] = sc["rmse_unmatched"]
    return m


def airport_unmatched_table(y, preds: dict, um, ap) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    rows = []
    for a in AIRPORTS:
        sel = um & (ap == a) & np.isfinite(y)
        n = int(sel.sum())
        if n == 0:
            continue
        row = {
            "airport": a,
            "n": n,
            "mean_y": float(np.mean(y[sel])),
            "med_y": float(np.median(y[sel])),
            "lirf_override": a == "LIRF",
        }
        for name, p in preds.items():
            pp = np.asarray(p, dtype=np.float64)
            row[f"rmse_{name}"] = rmse(y[sel], pp[sel])
            row[f"mae_{name}"] = mae(y[sel], pp[sel])
            row[f"mean_pred_{name}"] = float(np.nanmean(pp[sel]))
        rows.append(row)
    # all unmatched + non-LIRF unmatched
    for lab, sel0 in [
        ("ALL_UNMATCHED", um),
        ("NON_LIRF_UNMATCHED", non_lirf_unmatched(um, ap)),
        ("LIRF_UNMATCHED", lirf_unmatched(um, ap)),
    ]:
        sel = sel0 & np.isfinite(y)
        row = {
            "airport": lab,
            "n": int(sel.sum()),
            "mean_y": float(np.mean(y[sel])) if sel.any() else float("nan"),
            "med_y": float(np.median(y[sel])) if sel.any() else float("nan"),
            "lirf_override": lab == "LIRF_UNMATCHED",
        }
        for name, p in preds.items():
            pp = np.asarray(p, dtype=np.float64)
            row[f"rmse_{name}"] = rmse(y[sel], pp[sel]) if sel.any() else float("nan")
            row[f"mae_{name}"] = mae(y[sel], pp[sel]) if sel.any() else float("nan")
            row[f"mean_pred_{name}"] = float(np.nanmean(pp[sel])) if sel.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def make_plots(figdir: Path, label: str, y, preds: dict, um, ap, metrics: dict):
    figdir.mkdir(parents=True, exist_ok=True)
    names = [k for k in VARIANTS if k in preds]
    colors = {
        "E18-0": "#4c72b0",
        "E18-A-geo": "#dd8452",
        "E18-A-mean": "#55a868",
        "E18-A-med": "#8172b3",
        "E18-H": "#c44e52",
        "E18-B": "#937860",
    }

    # 01 overall / matched / unmatched / non-LIRF unmatched
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    keys = ["rmse_overall", "rmse_matched", "rmse_unmatched", "rmse_non_lirf_unmatched"]
    labs = ["overall", "matched", "unmatched", "non-LIRF unmatched"]
    x = np.arange(len(labs))
    w = 0.13
    for i, name in enumerate(names):
        vals = [metrics[name][k] for k in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=name, color=colors.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(f"{label} — RMSE by slice")
    ax.legend(fontsize=8, ncol=3)
    savefig(figdir / "01_rmse_comparison.png")

    # 02 unmatched RMSE by airport (E18-0 vs A-mean vs B)
    show = [k for k in ("E18-0", "E18-A-mean", "E18-B") if k in preds]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(AIRPORTS))
    w = 0.25
    for i, name in enumerate(show):
        vals = [metrics[name].get(f"rmse_{a}", np.nan) for a in AIRPORTS]
        # per-airport overall RMSE includes matched; we want unmatched-only from the table later.
        # Use a dedicated computation:
        vals = []
        for a in AIRPORTS:
            sel = um & (ap == a)
            vals.append(rmse(y[sel], preds[name][sel]) if sel.any() else np.nan)
        ax.bar(x + (i - (len(show) - 1) / 2) * w, vals, w, label=name, color=colors.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels(AIRPORTS, rotation=30)
    ax.set_ylabel("Unmatched RMSE (s)")
    ax.set_title(f"{label} — Unmatched RMSE by airport")
    ax.legend()
    savefig(figdir / "02_unmatched_rmse_by_airport.png")

    # 03 non-LIRF unmatched actual vs predicted
    nlu = non_lirf_unmatched(um, ap) & np.isfinite(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rng = np.random.default_rng(0)
    idx = np.where(nlu)[0]
    sample = rng.choice(idx, size=min(8000, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)
    for ax, key, title in [
        (axes[0], "E18-0", "E18-0 (E16-A geo+tree)"),
        (axes[1], "E18-B" if "E18-B" in preds else "E18-A-mean", "specialist" if "E18-B" in preds else "A-mean"),
    ]:
        if key not in preds or len(sample) == 0:
            ax.set_visible(False)
            continue
        ax.scatter(preds[key][sample], y[sample], s=6, alpha=0.25, color="#2a6f97")
        lim = [0, 4000]
        ax.plot(lim, lim, "r--", lw=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Predicted (s)")
        ax.set_ylabel("Actual (s)")
        ax.set_title(f"{label} — {title}\nnon-LIRF unmatched")
    savefig(figdir / "03_nlu_actual_vs_pred.png")

    # 04 EHAM unmatched residual
    eham = um & (ap == "EHAM") & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(-1500, 1500, 50)
    for name in show:
        r = y[eham] - preds[name][eham]
        ax.hist(r, bins=bins, alpha=0.45, density=True, label=f"{name} μ={r.mean():.0f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual actual − pred (s)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — EHAM unmatched residual")
    ax.legend(fontsize=8)
    savefig(figdir / "04_eham_unmatched_residual.png")

    # 05 mean pred vs mean y by airport (unmatched)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(AIRPORTS))
    mean_y = [float(np.mean(y[um & (ap == a)])) if (um & (ap == a)).any() else np.nan for a in AIRPORTS]
    ax.plot(x, mean_y, "k-o", ms=5, label="mean actual")
    for name in show:
        mean_p = [
            float(np.mean(preds[name][um & (ap == a)])) if (um & (ap == a)).any() else np.nan for a in AIRPORTS
        ]
        ax.plot(x, mean_p, "-s", ms=4, label=name, color=colors.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels(AIRPORTS, rotation=30)
    ax.set_ylabel("Seconds")
    ax.set_title(f"{label} — Unmatched mean actual vs predicted")
    ax.legend()
    savefig(figdir / "05_unmatched_mean_by_airport.png")


def fmt_row(name, m) -> str:
    return (
        f"  {name:12s} overall={m['rmse']:.2f}  matched={m['rmse_matched']:.2f}  "
        f"unmatched={m['rmse_unmatched']:.2f}  nlu={m['rmse_non_lirf_unmatched']:.2f}  "
        f"lirf_u={m['rmse_lirf_unmatched']:.2f}"
    )


def write_report(payload: dict) -> Path:
    j = payload["janjul"]
    d = payload["dec"]
    jm, dm = j["metrics"], d["metrics"]
    decision = payload["decision"]
    lines = []
    a = lines.append
    a("# E18 — Non-LIRF unmatched specialist")
    a("")
    a("**Project:** OpenAir")
    a("**Target:** `TAXITIME_SEC_mvt`")
    a("**Experiment:** E18")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress check:** December 2025")
    a("**Data:** 12 `training_*.parquet` files only. No ranking/submission.")
    a("")
    a("Script: `experiments/run_e18_unmatched_specialist.py` (copied to `analysis/E18/`).")
    a("")
    a("---")
    a("")
    a("## Objective")
    a("")
    a("After the LIRF unmatched `MVT−SCHED` override, unmatched RMSE stays ~2240.")
    a("E12 showed residual LightGBM slightly *hurts* that slice vs E3+override")
    a("(2241 vs 2228). Non-LIRF unmatched (~25% of holdout SSE) are scored with")
    a("matched `geo_mean` plus a tree trained on 99% matched rows.")
    a("")
    a("**Hypothesis:** a dedicated non-LIRF unmatched head — even a train airport")
    a("unmatched mean — moves **overall** RMSE, because EHAM unmatched are")
    a("systematically shorter than matched and LTFM/EGLL/LFPG unmatched have")
    a("heavier tails.")
    a("")
    a("Matched path and LIRF override stay frozen for A/B. Hygiene (H) is the")
    a("one variant allowed to refit the residual tree.")
    a("")
    a("---")
    a("")
    a("## Frozen baseline")
    a("")
    a("```")
    a("P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)")
    a("y_hat = P_cal + LGB_L2_residual(E12 matrix + E16-A disruption cols)")
    a("if unmatched and airport==LIRF: y_hat = MVT - SCHED")
    a("```")
    a("")
    a(
        f"Reproduction: Jan+Jul overall **{jm['E18-0']['rmse']:.2f}**, matched "
        f"**{jm['E18-0']['rmse_matched']:.2f}** (E16-A journal 376.04 / 253.82)."
    )
    a("")
    a("---")
    a("")
    a("## Variants")
    a("")
    a("| ID | What changes | Matched path |")
    a("|---|---|---|")
    a("| E18-0 | E16-A reproduction | frozen |")
    a("| E18-A-geo | non-LIRF unmatched → `P_cal` (drop residual) | frozen |")
    a("| E18-A-mean | non-LIRF unmatched → train airport-unmatched mean | frozen |")
    a("| E18-A-med | non-LIRF unmatched → train airport-unmatched median | frozen |")
    a("| E18-H | `geo_mean` fit on matched-only; LIRF-override rows dropped from residual train | refit |")
    a("| E18-B | non-LIRF unmatched → unmatched-mean + small L2 residual tree on that slice only | frozen |")
    a("")
    a("Specialist features (ranking-safe, present without AOBT): `mvt_sched`,")
    a("`geo_mean`, roll10 `MVT−AOBT`, hour/dow/month, `dis_state_30m`,")
    a("`dis_frac20_30m`, dep/queue counts, airport/stand/runway/type/ADES/prefix.")
    a("")
    a("---")
    a("")
    a("## Jan+Jul 2025 (primary)")
    a("")
    a("| Variant | Overall | Matched | Unmatched | Non-LIRF unmatched | LIRF unmatched |")
    a("|---|---:|---:|---:|---:|---:|")
    for name in VARIANTS:
        m = jm[name]
        a(
            f"| {name} | {m['rmse']:.2f} | {m['rmse_matched']:.2f} | "
            f"{m['rmse_unmatched']:.2f} | {m['rmse_non_lirf_unmatched']:.2f} | "
            f"{m['rmse_lirf_unmatched']:.2f} |"
        )
    a("")
    a("Deltas vs E18-0 (negative = better):")
    a("")
    a("| Variant | Δ overall | Δ matched | Δ unmatched | Δ non-LIRF unmatched |")
    a("|---|---:|---:|---:|---:|")
    b0 = jm["E18-0"]
    for name in VARIANTS:
        if name == "E18-0":
            continue
        m = jm[name]
        a(
            f"| {name} | {m['rmse']-b0['rmse']:+.2f} | {m['rmse_matched']-b0['rmse_matched']:+.2f} | "
            f"{m['rmse_unmatched']-b0['rmse_unmatched']:+.2f} | "
            f"{m['rmse_non_lirf_unmatched']-b0['rmse_non_lirf_unmatched']:+.2f} |"
        )
    a("")
    a("n unmatched = "
      f"{j['n_unmatched']:,} (LIRF {j['n_lirf_unmatched']:,}, "
      f"non-LIRF {j['n_non_lirf_unmatched']:,}).")
    a("")
    a("---")
    a("")
    a("## December 2025 (stress)")
    a("")
    a("| Variant | Overall | Matched | Unmatched | Non-LIRF unmatched | LIRF unmatched |")
    a("|---|---:|---:|---:|---:|---:|")
    for name in VARIANTS:
        m = dm[name]
        a(
            f"| {name} | {m['rmse']:.2f} | {m['rmse_matched']:.2f} | "
            f"{m['rmse_unmatched']:.2f} | {m['rmse_non_lirf_unmatched']:.2f} | "
            f"{m['rmse_lirf_unmatched']:.2f} |"
        )
    a("")
    a("Deltas vs E18-0:")
    a("")
    a("| Variant | Δ overall | Δ matched | Δ unmatched | Δ non-LIRF unmatched |")
    a("|---|---:|---:|---:|---:|")
    b0d = dm["E18-0"]
    for name in VARIANTS:
        if name == "E18-0":
            continue
        m = dm[name]
        a(
            f"| {name} | {m['rmse']-b0d['rmse']:+.2f} | {m['rmse_matched']-b0d['rmse_matched']:+.2f} | "
            f"{m['rmse_unmatched']-b0d['rmse_unmatched']:+.2f} | "
            f"{m['rmse_non_lirf_unmatched']-b0d['rmse_non_lirf_unmatched']:+.2f} |"
        )
    a("")
    a("---")
    a("")
    a("## Unmatched by airport (Jan+Jul)")
    a("")
    tbl = pd.read_csv(TAB / "e18_unmatched_by_airport_janjul.csv")
    ap_tbl = tbl[~tbl["airport"].str.contains("UNMATCHED")]
    a("| Airport | n | mean y | RMSE E18-0 | RMSE A-mean | RMSE B | Δ A-mean | Δ B |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ap_tbl.iterrows():
        a(
            f"| {r['airport']} | {int(r['n'])} | {r['mean_y']:.0f} | "
            f"{r['rmse_E18-0']:.1f} | {r['rmse_E18-A-mean']:.1f} | {r['rmse_E18-B']:.1f} | "
            f"{r['rmse_E18-A-mean']-r['rmse_E18-0']:+.1f} | {r['rmse_E18-B']-r['rmse_E18-0']:+.1f} |"
        )
    a("")
    spec = j.get("specialist", {})
    if spec:
        a("### E18-B specialist")
        a("")
        a(f"- train non-LIRF unmatched n = {spec.get('n_train', 'NA')}")
        a(f"- trees = {spec.get('trees', 'NA')}")
        imp = spec.get("importance") or []
        if imp:
            top = ", ".join(f"`{f}` ({g})" for f, g in imp[:10])
            a(f"- top gain: {top}")
        a("")
    a("---")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {decision['verdict']}")
    a("")
    a(decision["reason"])
    a("")
    a("Keep rules:")
    a("")
    a("- Accept a splice (A/B) only if Jan+Jul **overall** RMSE falls and December")
    a("  overall does not rise. Matched RMSE must stay within 0.05 s of E18-0")
    a("  (frozen path).")
    a("- Accept H only if unmatched improves without a matched regression on both splits.")
    a("- Do not apply the LIRF `MVT−SCHED` rule to non-LIRF unmatched (EHAM is shorter).")
    a("")
    a("Artifacts: `analysis/E18/`.")
    a("")
    path = REP / "E18_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    return path


def decide(payload: dict) -> dict:
    jm = payload["janjul"]["metrics"]
    dm = payload["dec"]["metrics"]
    b0j, b0d = jm["E18-0"], dm["E18-0"]

    def ok_frozen(m):
        return abs(m["rmse_matched"] - b0j["rmse_matched"]) < 0.05

    winners = []
    notes = []
    for name in ("E18-A-geo", "E18-A-mean", "E18-A-med", "E18-B"):
        mj, md = jm[name], dm[name]
        dj = mj["rmse"] - b0j["rmse"]
        dd = md["rmse"] - b0d["rmse"]
        frozen = ok_frozen(mj) and abs(md["rmse_matched"] - b0d["rmse_matched"]) < 0.05
        if not frozen:
            notes.append(f"{name}: matched moved (not a pure splice)")
            continue
        if dj < -0.5 and dd <= 0.5:
            winners.append((name, dj, dd))
            notes.append(f"{name}: KEEP candidate (Jan+Jul {dj:+.2f}, Dec {dd:+.2f})")
        elif dj < 0 and dd < 0:
            notes.append(f"{name}: small consistent gain (Jan+Jul {dj:+.2f}, Dec {dd:+.2f})")
        else:
            notes.append(f"{name}: reject (Jan+Jul {dj:+.2f}, Dec {dd:+.2f})")

    hj, hd = jm["E18-H"], dm["E18-H"]
    h_j = hj["rmse"] - b0j["rmse"]
    h_d = hd["rmse"] - b0d["rmse"]
    h_mj = hj["rmse_matched"] - b0j["rmse_matched"]
    if h_j < -0.5 and h_d <= 0.5 and h_mj <= 1.0:
        winners.append(("E18-H", h_j, h_d))
        notes.append(f"E18-H: KEEP candidate (overall {h_j:+.2f}/{h_d:+.2f}, matched {h_mj:+.2f})")
    else:
        notes.append(f"E18-H: reject or weak (overall {h_j:+.2f}/{h_d:+.2f}, matched {h_mj:+.2f})")

    if winners:
        best = min(winners, key=lambda t: t[1])
        # require Dec not worse
        if best[2] > 1.0:
            verdict = "INCONCLUSIVE"
        elif best[1] < -2.0:
            verdict = "ACCEPTED"
        else:
            verdict = "INCONCLUSIVE"
        keep = best[0]
    else:
        verdict = "REJECTED"
        keep = None

    reason = (
        f"Jan+Jul E18-0={b0j['rmse']:.2f} / unmatched={b0j['rmse_unmatched']:.2f} / "
        f"nlu={b0j['rmse_non_lirf_unmatched']:.2f}. "
        + ("Best splice+hygiene: " + ", ".join(f"{n} {dj:+.2f}/{dd:+.2f}" for n, dj, dd in winners) + ". "
           if winners else "No variant beat E18-0 on overall RMSE with December confirmation. ")
        + " ".join(notes)
    )
    return {"verdict": verdict, "keep": keep, "winners": winners, "notes": notes, "reason": reason}


def run_split(dep: pl.DataFrame, months: list[int], split_name: str) -> dict:
    label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
    log(f"===== {split_name} {label} =====")
    pack0 = prepare_split_e16a(dep, months)
    tr, va = add_prefix(pack0["tr"]), add_prefix(pack0["va"])
    y = va["y"].to_numpy().astype(np.float64)
    um = va["unmatched"].to_numpy()
    ap = va["airport"].to_numpy()
    nlu = non_lirf_unmatched(um, ap)
    lu = lirf_unmatched(um, ap)
    log(f"  val n={len(y):,} unmatched={int(um.sum()):,} LIRF_u={int(lu.sum()):,} non-LIRF_u={int(nlu.sum()):,}")

    log("  fitting E18-0 (E16-A residual, exact helper)...")
    p_cal, pred0, _ = fit_residual_e16a(tr, va, extra=list(MODEL_DISRUPT_COLS))
    m0 = slice_metrics(y, pred0, um, ap)
    log(fmt_row("E18-0", m0))
    if split_name == "janjul" and abs(m0["rmse"] - BASELINE_OVERALL) > 3.0:
        raise SystemExit(f"E16-A failed to reproduce: overall {m0['rmse']:.2f} vs {BASELINE_OVERALL}")

    pred_geo = splice(pred0, p_cal, nlu)
    u_mean = airport_unmatched_stat(tr, va, "mean")
    u_med = airport_unmatched_stat(tr, va, "median")
    pred_mean = splice(pred0, u_mean, nlu)
    pred_med = splice(pred0, u_med, nlu)

    log("  fitting E18-H (matched geo + drop LIRF-override from residual train)...")
    pack_h = prepare_split(dep, months, matched_geo=True)
    _, pred_h_raw, _ = fit_residual(
        pack_h["tr"], pack_h["va"], extra=list(MODEL_DISRUPT_COLS), drop_override_from_train=True
    )
    pred_h = align_pred(pack_h["va"], pred_h_raw, va)

    log("  fitting E18-B unmatched specialist...")
    spec_pred, _, spec_meta = fit_unmatched_specialist(tr, va, airport_unmatched_stat(tr, tr, "mean"), u_mean)
    pred_b = splice(pred0, spec_pred, nlu)

    preds = {
        "E18-0": pred0,
        "E18-A-geo": pred_geo,
        "E18-A-mean": pred_mean,
        "E18-A-med": pred_med,
        "E18-H": pred_h,
        "E18-B": pred_b,
    }
    metrics = {k: slice_metrics(y, p, um, ap) for k, p in preds.items()}
    for name in VARIANTS:
        log(fmt_row(name, metrics[name]))
        if name not in ("E18-0", "E18-H"):
            dmatch = metrics[name]["rmse_matched"] - m0["rmse_matched"]
            if abs(dmatch) > 0.05:
                log(f"    WARN {name} matched moved {dmatch:+.4f} (expected frozen)")

    tbl = airport_unmatched_table(y, preds, um, ap)
    tbl.to_csv(TAB / f"e18_unmatched_by_airport_{split_name}.csv", index=False)
    log(f"  table e18_unmatched_by_airport_{split_name}.csv")

    figdir = FIG if split_name == "janjul" else FIG / "dec"
    make_plots(figdir, label, y, preds, um, ap, metrics)

    return {
        "metrics": metrics,
        "n_val": int(len(y)),
        "n_unmatched": int(um.sum()),
        "n_lirf_unmatched": int(lu.sum()),
        "n_non_lirf_unmatched": int(nlu.sum()),
        "specialist": spec_meta,
        "train_unmatched_mean_by_airport": _airport_unmatched_means(tr),
    }


def main():
    t0 = datetime.now(timezone.utc)
    log("E18: load training_*.parquet only...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    log("  disruption state...")
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    arr_d = load_arr_delay()
    dep = add_arrival_delay_state(dep, arr_d)
    log(f"ready {dep.height:,} rows")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "janjul": run_split(dep, [1, 7], "janjul"),
        "dec": run_split(dep, [12], "dec"),
    }
    payload["decision"] = decide(payload)
    log(f"VERDICT {payload['decision']['verdict']}")
    log(payload["decision"]["reason"])

    write_report(payload)
    save_result("E18", payload)
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(HERE / "run_e18_unmatched_specialist.py", OUT / "run_e18.py")
    log(f"done in {(datetime.now(timezone.utc) - t0).total_seconds()/60:.1f} min")


if __name__ == "__main__":
    main()
