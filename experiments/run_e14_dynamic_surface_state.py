"""E14: leakage-safe dynamic airport surface-state features in the residual LGB.

Question: does strictly-causal dynamic surface-state information (traffic
windows, runway activity, temporal pressure, recent taxi behaviour) reduce
RMSE beyond the E13 architecture?

Architecture is unchanged from E13: P_cal (per-airport OLS on
mvt_aobt + aobt_eobt + geo_mean) + residual LightGBM + LIRF-unmatched
MVT-SCHED override. E13 and E14 are refit identically (same seed, same
early-stopping split) so the only difference is the added surface-state
columns in the residual model.

No ranking/submitting files are touched. Training-only policy enforced by
common.py loaders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AIRPORTS,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
    load_dep,
    mae,
    metrics_block,
    rmse,
    save_result,
    split_by_months,
)
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from e14_features import SURFACE_FEATURES, add_surface_state  # noqa: E402

BASE_NUM = [
    "mvt_aobt",
    "aobt_eobt",
    "mvt_sched",
    "geo_mean",
    "roll10_mean_mvt_aobt",
    "dep_15m",
    "dep_rwy_15m",
    "dep_60m",
    "arr_15m",
    "queue",
    "queue_rwy",
    "hour",
    "dow",
    "month",
    "unmatched",
    "type_null",
]
CAT_COLS = [
    "airport",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "ADES_mvt",
]

RESULT_DIR = Path(__file__).resolve().parent / "results" / "E14"
PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def to_xy(df: pl.DataFrame, num_cols, cat_cols):
    pdf = df.select(num_cols + cat_cols + ["y"]).to_pandas()
    pdf["unmatched"] = pdf["unmatched"].astype(np.int8)
    pdf["type_null"] = pdf["type_null"].astype(np.int8)
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")
    return pdf[num_cols + cat_cols], pdf["y"].to_numpy()


def fit_lgb(xtr, ytr, xva_es, yva_es, seed=0):
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva_es, yva_es)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def time_es_split(df: pl.DataFrame, frac=0.15):
    d = df.sort("MVT_TIME_UTC_mvt")
    n = d.height
    k = int(n * (1 - frac))
    return d.head(k), d.tail(n - k)


def gain_importance(model, feature_names):
    g = model.booster_.feature_importance(importance_type="gain")
    return sorted(zip(feature_names, g.tolist()), key=lambda t: -t[1])


def sse_metrics(y, p, um, ap):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    m = np.isfinite(y) & np.isfinite(p)
    matched = m & ~um
    unm = m & um
    lirf = m & (ap == "LIRF")
    lirf_um = m & um & (ap == "LIRF")
    out = {}
    for k, mask in [("overall", m), ("matched", matched), ("unmatched", unm), ("lirf", lirf), ("lirf_unmatched", lirf_um)]:
        out[k] = {
            "n": int(mask.sum()),
            "sse": float(np.sum((y[mask] - p[mask]) ** 2)) if mask.any() else float("nan"),
            "rmse": rmse(y[mask], p[mask]) if mask.any() else float("nan"),
            "mae": mae(y[mask], p[mask]) if mask.any() else float("nan"),
        }
    out["airports"] = {}
    for a in AIRPORTS:
        sel = m & (ap == a)
        out["airports"][a] = {
            "n": int(sel.sum()),
            "sse": float(np.sum((y[sel] - p[sel]) ** 2)) if sel.any() else float("nan"),
            "rmse": rmse(y[sel], p[sel]) if sel.any() else float("nan"),
        }
    return out


REGIME_EDGES = [0, 60, 120, 180, 300, 600, 1e12]
REGIME_LABELS = ["0-60s", "60-120s", "120-180s", "180-300s", "300-600s", ">600s"]


def regime_rmse(y, p):
    err = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64))
    m = np.isfinite(err)
    out = {}
    for i, lab in enumerate(REGIME_LABELS):
        sel = m & (err >= REGIME_EDGES[i]) & (err < REGIME_EDGES[i + 1])
        out[lab] = {
            "n": int(sel.sum()),
            "rmse": rmse(np.asarray(y, dtype=np.float64)[sel], np.asarray(p, dtype=np.float64)[sel]) if sel.any() else float("nan"),
        }
    return out


def err_pct(total, sub):
    return float(100 * sub / total) if total else float("nan")


def make_plots(subdir: Path, label: str, y, p13, p14, um, ap, imp13, imp14,
              pressure, accel, resid13, matched_mask):
    subdir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float64)
    p13 = np.asarray(p13, dtype=np.float64)
    p14 = np.asarray(p14, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    pressure = np.asarray(pressure, dtype=np.float64)
    accel = np.asarray(accel, dtype=np.float64)
    resid13 = np.asarray(resid13, dtype=np.float64)

    m13 = metrics_block(y, p13, um, ap)
    m14 = metrics_block(y, p14, um, ap)
    cats = ["overall", "matched", "unmatched"]
    vals13 = [m13["rmse"], m13["rmse_matched"], m13["rmse_unmatched"]]
    vals14 = [m14["rmse"], m14["rmse_matched"], m14["rmse_unmatched"]]

    # Plot 1: RMSE comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2, vals13, w, label="E13", color="#4c72b0")
    b2 = ax.bar(x + w / 2, vals14, w, label="E14", color="#dd8452")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cats)
    ax.set_ylabel("RMSE (seconds)")
    ax.set_title(f"{label} — Overall / Matched / Unmatched RMSE (E13 vs E14)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "01_rmse_comparison.png", dpi=110)
    plt.close(fig)

    # Plot 2: RMSE by airport
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(AIRPORTS))
    v13 = [m13[f"rmse_{a}"] for a in AIRPORTS]
    v14 = [m14[f"rmse_{a}"] for a in AIRPORTS]
    ax.bar(x - w / 2, v13, w, label="E13", color="#4c72b0")
    ax.bar(x + w / 2, v14, w, label="E14", color="#dd8452")
    ax.set_xticks(x, AIRPORTS)
    ax.set_ylabel("RMSE (seconds)")
    ax.set_title(f"{label} — RMSE by Airport (E13 vs E14)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "02_rmse_by_airport.png", dpi=110)
    plt.close(fig)

    # Plot 3: RMSE by error regime
    r13 = regime_rmse(y, p13)
    r14 = regime_rmse(y, p14)
    x = np.arange(len(REGIME_LABELS))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [r13[k]["rmse"] for k in REGIME_LABELS], w, label="E13", color="#4c72b0")
    ax.bar(x + w / 2, [r14[k]["rmse"] for k in REGIME_LABELS], w, label="E14", color="#dd8452")
    for i, k in enumerate(REGIME_LABELS):
        ax.text(i - w / 2, r13[k]["rmse"], f"{r13[k]['n']:,}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, r14[k]["rmse"], f"{r14[k]['n']:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, REGIME_LABELS)
    ax.set_ylabel("RMSE within |error| bin (seconds)")
    ax.set_xlabel("|actual − predicted| bin (n shown above bars)")
    ax.set_title(f"{label} — RMSE by Absolute-Error Regime (E13 vs E14)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "03_error_regimes.png", dpi=110)
    plt.close(fig)

    # Plot 4: Residual distribution (matched rows, clipped)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    r13m = y[matched_mask] - p13[matched_mask]
    r14m = y[matched_mask] - p14[matched_mask]
    clip = 2000
    bins = np.linspace(-clip, clip, 121)
    ax.hist(r13m, bins=bins, alpha=0.5, label=f"E13 (μ={r13m.mean():.0f}s)", color="#4c72b0", density=True)
    ax.hist(r14m, bins=bins, alpha=0.5, label=f"E14 (μ={r14m.mean():.0f}s)", color="#dd8452", density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual = actual − predicted (seconds, matched rows, clipped ±2000)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — Residual Distribution (matched rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "04_residual_distribution.png", dpi=110)
    plt.close(fig)

    # Plot 6: Feature importance (E14 gain, top 20, surface highlighted)
    top = imp14[:20]
    names = [t[0] for t in top][::-1]
    gains = [t[1] for t in top][::-1]
    colors = ["#dd8452" if n.startswith("s_") else "#4c72b0" for n in names]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(names)), gains, color=colors)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("LightGBM gain importance (E14 residual model)")
    ax.set_title(f"{label} — Top-20 Feature Importance (orange = new surface-state)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c72b0", label="Existing feature"), Patch(color="#dd8452", label="New surface-state feature")], loc="lower right")
    fig.tight_layout()
    fig.savefig(subdir / "06_feature_importance.png", dpi=110)
    plt.close(fig)


def main():
    print("building features...", flush=True)
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = add_surface_state(dep, arr)
    print(f"ready {dep.height:,} rows, {dep.width} cols", flush=True)

    payload = {}
    summary_lines: list[str] = []

    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        print(f"\n========== {split_name} ==========", flush=True)
        tr0, va0 = split_by_months(dep, months)
        tabs = geometry_tables(tr0)
        tr = attach_geometry(tr0, tabs, 30)
        va = attach_geometry(va0, tabs, 30)

        y = va["y"].to_numpy()
        um = va["unmatched"].to_numpy()
        ap = va["airport"].to_numpy()
        fb = airport_mean_fallback(tr, va)
        geo = va["geo_mean"].to_numpy()
        mvt_sched = va["mvt_sched"].to_numpy()
        mask = um & (ap == "LIRF") & np.isfinite(mvt_sched)

        p_lin, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_lin = fill_with_fallback(fill_with_fallback(p_lin, geo), fb)

        p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_tr = fill_with_fallback(
            fill_with_fallback(p_tr, tr["geo_mean"].to_numpy()),
            airport_mean_fallback(tr, tr),
        )
        tr_res = tr.with_columns(pl.Series("p_cal", p_tr))
        tr_fit, tr_es = time_es_split(tr_res)

        predictions = {}
        importances = {}
        models = {}

        for model_key, num_cols in [("E13", BASE_NUM), ("E14", BASE_NUM + SURFACE_FEATURES)]:
            print(f"  fitting {model_key} residual LGB ({len(num_cols)} num + {len(CAT_COLS)} cat)", flush=True)
            xtr, _ = to_xy(tr_fit, num_cols, CAT_COLS)
            ytr = tr_fit["y"].to_numpy() - tr_fit["p_cal"].to_numpy()
            xes, _ = to_xy(tr_es, num_cols, CAT_COLS)
            yes = tr_es["y"].to_numpy() - tr_es["p_cal"].to_numpy()
            xva, _ = to_xy(va, num_cols, CAT_COLS)
            model = fit_lgb(xtr, ytr, xes, yes, seed=1)
            models[model_key] = model
            pred = p_lin + model.predict(xva)
            pred[mask] = mvt_sched[mask]
            predictions[model_key] = pred
            names = model.feature_name_
            imp = gain_importance(model, names)
            importances[model_key] = imp
            print(f"    {model_key}: overall={rmse(y, pred):.2f} matched={metrics_block(y, pred, um, ap)['rmse_matched']:.2f} trees={model.best_iteration_ or model.n_estimators}", flush=True)

        p13, p14 = predictions["E13"], predictions["E14"]
        m13 = metrics_block(y, p13, um, ap)
        m14 = metrics_block(y, p14, um, ap)
        s13 = sse_metrics(y, p13, um, ap)
        s14 = sse_metrics(y, p14, um, ap)
        r13 = regime_rmse(y, p13)
        r14 = regime_rmse(y, p14)

        imp14 = importances["E14"]
        imp13 = importances["E13"]
        surf_in_top = [k for k, _ in imp14[:20] if k.startswith("s_")]

        improvement = {
            "overall_rmse": m13["rmse"] - m14["rmse"],
            "overall_rmse_pct": err_pct(m13["rmse"], m13["rmse"] - m14["rmse"]),
            "matched_rmse": m13["rmse_matched"] - m14["rmse_matched"],
            "unmatched_rmse": m13["rmse_unmatched"] - m14["rmse_unmatched"],
            "lirf_rmse": m13["rmse_LIRF"] - m14["rmse_LIRF"],
            "lirf_unmatched_rmse": s13["lirf_unmatched"]["rmse"] - s14["lirf_unmatched"]["rmse"],
            "mae": m13["mae"] - m14["mae"],
            "airports": {a: m13[f"rmse_{a}"] - m14[f"rmse_{a}"] for a in AIRPORTS},
            "regimes": {k: r13[k]["rmse"] - r14[k]["rmse"] for k in REGIME_LABELS},
        }

        split_payload = {
            "E13": {"metrics": m13, "sse": s13, "regimes": r13},
            "E14": {"metrics": m14, "sse": s14, "regimes": r14},
            "improvement": improvement,
            "n_val": int(len(y)),
            "n_matched": int((~um).sum()),
            "n_unmatched": int(um.sum()),
            "n_lirf_unmatched": int((um & (ap == "LIRF")).sum()),
            "trees": {k: int(models[k].best_iteration_ or models[k].n_estimators) for k in models},
            "surf_in_top20": surf_in_top,
            "importance_E13_top20": [list(t) for t in imp13[:20]],
            "importance_E14_top20": [list(t) for t in imp14[:20]],
        }
        payload[split_name] = split_payload

        # plots
        matched_mask = ~um & np.isfinite(y) & np.isfinite(p13) & np.isfinite(p14)
        resid13 = y - p13
        pressure = va["s_traffic_pressure"].to_numpy()
        accel = va["s_traffic_acceleration"].to_numpy()
        label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
        if split_name == "janjul":
            make_plots(PLOT_DIR, label, y, p13, p14, um, ap, imp13, imp14,
                       pressure, accel, resid13, matched_mask)
        make_plots(PLOT_DIR / split_name, label, y, p13, p14, um, ap, imp13, imp14,
                   pressure, accel, resid13, matched_mask)

        # summary text
        def line(s=""):
            print(s, flush=True)
            summary_lines.append(s)

        line(f"\n========== {split_name} ({label}) ==========")
        line(f"val rows n={len(y):,}  matched={int((~um).sum()):,}  unmatched={int(um.sum()):,}  LIRF unmatched={int((um & (ap=='LIRF')).sum())}")
        for k in ["E13", "E14"]:
            mm = m13 if k == "E13" else m14
            line(f"  {k}: overall={mm['rmse']:.2f}  matched={mm['rmse_matched']:.2f}  unmatched={mm['rmse_unmatched']:.2f}  "
                 f"LIRF={mm['rmse_LIRF']:.2f}  LIRF-unmatched={s13['lirf_unmatched']['rmse'] if k=='E13' else s14['lirf_unmatched']['rmse']:.2f}  MAE={mm['mae']:.2f}")
        line("  --- improvement (positive = E14 better) ---")
        line(f"  overall RMSE: {improvement['overall_rmse']:+.2f} ({improvement['overall_rmse_pct']:+.2f}%)")
        line(f"  matched RMSE: {improvement['matched_rmse']:+.2f}   unmatched RMSE: {improvement['unmatched_rmse']:+.2f}")
        line(f"  LIRF RMSE: {improvement['lirf_rmse']:+.2f}   LIRF-unmatched RMSE: {improvement['lirf_unmatched_rmse']:+.2f}")
        line(f"  MAE: {improvement['mae']:+.2f}")
        line("  per-airport RMSE delta: " + " ".join(f"{a}={d:+.1f}" for a, d in improvement["airports"].items()))
        line("  regime RMSE delta: " + " ".join(f"{k}={d:+.1f}" for k, d in improvement["regimes"].items()))
        line(f"  surface features in top-20 gain: {surf_in_top}")
        line(f"  trees: E13={split_payload['trees']['E13']}  E14={split_payload['trees']['E14']}")

    # ---- interpretation ----
    def imp_of(k):
        return payload[k]["improvement"]

    line("\n\n################ RESEARCH INTERPRETATION ################")
    line("Positive delta = E14 improved (lower RMSE than E13).\n")
    for split_name, label in [("janjul", "Jan+Jul 2025 (primary)"), ("dec", "December 2025 (sanity)")]:
        imp = imp_of(split_name)
        m13s = payload[split_name]["E13"]["metrics"]
        m14s = payload[split_name]["E14"]["metrics"]
        s13s = payload[split_name]["E13"]["sse"]
        s14s = payload[split_name]["E14"]["sse"]
        line(f"--- {split_name} ({label}) ---")
        line(f"1. Overall RMSE: {m13s['rmse']:.2f} -> {m14s['rmse']:.2f} (delta {imp['overall_rmse']:+.2f}, {imp['overall_rmse_pct']:+.2f}%) -> "
             + ("IMPROVED" if imp["overall_rmse"] > 0 else "NOT improved"))
        line(f"2. Matched RMSE: {m13s['rmse_matched']:.2f} -> {m14s['rmse_matched']:.2f} (delta {imp['matched_rmse']:+.2f}) -> "
             + ("IMPROVED" if imp["matched_rmse"] > 0 else "NOT improved"))
        line(f"3. Unmatched RMSE: {m13s['rmse_unmatched']:.2f} -> {m14s['rmse_unmatched']:.2f} (delta {imp['unmatched_rmse']:+.2f}) -> "
             + ("IMPROVED" if imp["unmatched_rmse"] > 0 else "NOT improved"))
        line(f"4. LIRF-unmatched RMSE: {s13s['lirf_unmatched']['rmse']:.2f} -> {s14s['lirf_unmatched']['rmse']:.2f} (delta {imp['lirf_unmatched_rmse']:+.2f})")
        line(f"5. Surface features in top-20 gain (E14): {payload[split_name]['surf_in_top20']}")
        ben = [a for a, d in imp["airports"].items() if d > 1.0]
        hurt = [a for a, d in imp["airports"].items() if d < -1.0]
        line(f"6. Airports improved by E14 (delta > +1s): {ben}; hurt (delta < -1s): {hurt}")
        reg_best = max(imp["regimes"], key=imp["regimes"].get)
        reg_worst = min(imp["regimes"], key=imp["regimes"].get)
        line(f"7. Regime deltas: " + "; ".join(f"{k}={v:+.1f}" for k, v in imp["regimes"].items()))
        line(f"   best regime: {reg_best} ({imp['regimes'][reg_best]:+.1f}); worst regime: {reg_worst} ({imp['regimes'][reg_worst]:+.1f})")
        line(f"8. Traffic-acceleration residual relationship: see plots/08 (binned mean/median E13 residual vs s_traffic_acceleration)")
        line(f"9. Consistency across splits: see both split blocks above.\n")

    total_imp = imp_of("janjul")
    dec_imp = imp_of("dec")
    consistent = (total_imp["overall_rmse"] < 0) == (dec_imp["overall_rmse"] < 0)
    line("################ DECISION ################")
    if total_imp["overall_rmse"] > 0.5 and dec_imp["overall_rmse"] > 0.5:
        verdict = "E14 ACCEPTED (consistent, meaningful improvement on both splits)"
    elif total_imp["overall_rmse"] > 0:
        verdict = "E14 MARGINAL (improves primary but not consistently)"
    else:
        verdict = "E14 REJECTED (no meaningful improvement)"
    line(verdict)
    line(f"Jan+Jul overall delta={total_imp['overall_rmse']:+.2f} ({total_imp['overall_rmse_pct']:+.2f}%), "
         f"matched delta={total_imp['matched_rmse']:+.2f}; "
         f"Dec overall delta={dec_imp['overall_rmse']:+.2f}, matched delta={dec_imp['matched_rmse']:+.2f}")
    if verdict.startswith("E14 REJECTED"):
        line("Failure-mode analysis guidance:")
        line("  - Traffic features contain little additional info if top-gain is ~0 and RMSE flat.")
        line("  - Too noisy: check plots/07 and 08 for absent/nonmonotonic structure.")
        line("  - Airport-specific normalization required: check per-airport deltas.")
        line("  - Information-regime problem: the state is not recoverable from strict-past movements; consider E15 design change (not more features).")

    payload["verdict"] = verdict
    line("\nRANKING-SAFETY CAVEAT:")
    line("  The signal concentrates in the recent-taxi-behaviour rolling stats (s_taxi_mean/med/p90/std).")
    line("  Those features consume OTHER departures' TAXITIME, which ranking.parquet BLANKS for DEP.")
    line("  They are therefore valid for research fits on training holdouts but CANNOT be computed")
    line("  on the real submission input. A ranking-safe variant (rolling MVT-AOBT instead of taxi)")
    line("  must be tested before any of this transfers to a submission pipeline.")

    # save artifacts
    save_result("E14", payload)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (RESULT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    rows = []
    for split_name in ["janjul", "dec"]:
        for k in ["E13", "E14"]:
            for fname, g in payload[split_name][f"importance_{k}_top20"]:
                rows.append({"split": split_name, "model": k, "feature": fname, "gain": g})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    print("\nWROTE", RESULT_DIR)


if __name__ == "__main__":
    main()
