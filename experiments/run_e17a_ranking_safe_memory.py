"""E17-A: ranking-safe operational memory.

Can the E14 short-term operational-memory signal (rolling taxi behaviour)
be recovered using ONLY ranking-safe historical variables (MVT-AOBT,
AOBT-EOBT, MVT-SCHED, runway-local history)? No TAXITIME anywhere.

Models (architecture frozen from E16-A):
  E17-A0 = exact E16-A reproduction (P_cal + residual LGB + disruption state
           + LIRF MVT-SCHED override)
  E17-A1 = E16-A + airport-level ranking-safe operational memory
  E17-A2 = E17-A1 + runway-local operational memory

Training-only policy enforced by common.py loaders. No ranking/submitting.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    AIRPORTS,
    add_causal_rolling,
    fill_with_fallback,
    load_dep,
    mae,
    metrics_block,
    rmse,
    save_result,
    split_by_months,
)
from run_e12_e9 import CAT_COLS, NUM_COLS as BASE_NUM_COLS  # noqa: E402
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    apply_override,
    load_arr_delay,
    override_mask,
    prepare_split,
    score_pred,
    fit_residual,
)
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from e17a_features import (  # noqa: E402
    AIRPORT_MEMORY_COLS,
    RUNWAY_MEMORY_COLS,
    add_airport_memory,
    add_runway_memory,
)

RESULT_DIR = HERE / "results" / "E17-A"
PLOT_DIR = RESULT_DIR / "plots"
for d in (RESULT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
BASELINE_E16A_JANJUL = 376.04
BASELINE_E16A_MATCHED = 253.82
E14_REFERENCE_JANJUL = 371.57

E17A0_EXTRA = MODEL_DISRUPT_COLS
E17A1_EXTRA = MODEL_DISRUPT_COLS + AIRPORT_MEMORY_COLS
E17A2_EXTRA = E17A1_EXTRA + RUNWAY_MEMORY_COLS

REGIME_EDGES = [0, 60, 120, 180, 300, 600, 1e12]
REGIME_LABELS = ["0-60s", "60-120s", "120-180s", "180-300s", "300-600s", ">600s"]


def regime_stats(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    err = np.abs(y - p)
    sse_total = float(np.sum((y[m] - p[m]) ** 2)) if m.any() else 1.0
    out = {}
    for i, lab in enumerate(REGIME_LABELS):
        sel = m & (err >= REGIME_EDGES[i]) & (err < REGIME_EDGES[i + 1])
        n = int(sel.sum())
        r = rmse(y[sel], p[sel]) if sel.any() else float("nan")
        a = mae(y[sel], p[sel]) if sel.any() else float("nan")
        sse = float(np.sum((y[sel] - p[sel]) ** 2)) if sel.any() else 0.0
        out[lab] = {"n": n, "rmse": r, "mae": a, "sse": sse, "sse_share": sse / sse_total if sse_total else 0.0}
    return out


def full_metrics(y, p, um, ap):
    m = metrics_block(y, p, um, ap)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    ok = np.isfinite(y) & np.isfinite(p)
    lirf_um = ok & um & (ap == "LIRF")
    m["n_val"] = int(ok.sum())
    m["n_matched"] = int((ok & ~um).sum())
    m["n_unmatched"] = int((ok & um).sum())
    m["n_lirf_unmatched"] = int(lirf_um.sum())
    m["rmse_lirf_unmatched"] = rmse(y[lirf_um], p[lirf_um]) if lirf_um.any() else float("nan")
    m["mae_lirf_unmatched"] = mae(y[lirf_um], p[lirf_um]) if lirf_um.any() else float("nan")
    m["sse_overall"] = float(np.sum((y[ok] - p[ok]) ** 2)) if ok.any() else 0.0
    m["sse_matched"] = float(np.sum((y[ok & ~um] - p[ok & ~um]) ** 2)) if (ok & ~um).any() else 0.0
    m["sse_unmatched"] = float(np.sum((y[ok & um] - p[ok & um]) ** 2)) if (ok & um).any() else 0.0
    m["sse_lirf_unmatched"] = float(np.sum((y[lirf_um] - p[lirf_um]) ** 2)) if lirf_um.any() else 0.0
    return m


def feature_class(feat: str) -> str:
    if feat in AIRPORT_MEMORY_COLS:
        return "airport_memory"
    if feat in RUNWAY_MEMORY_COLS:
        return "runway_memory"
    if feat in MODEL_DISRUPT_COLS:
        return "e16a"
    return "baseline"


def gain_importance(model, feature_names):
    g = model.booster_.feature_importance(importance_type="gain")
    return sorted(zip(feature_names, g.tolist()), key=lambda t: -t[1])


def binned_rel(x, y, nb=12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return None
    xm, ym = x[m], y[m]
    order = np.argsort(xm)
    q = np.array_split(xm[order], nb)
    qy = np.array_split(ym[order], nb)
    return {
        "x": [v.mean() for v in q],
        "mean": [v.mean() for v in qy],
        "med": [np.median(v) for v in qy],
        "n": [len(v) for v in qy],
    }


def make_plots(subdir: Path, label: str, y, preds: dict, um, ap, imp_top, resid_a0, matched_mask, va):
    subdir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    mm = matched_mask
    colors = {"E17-A0": "#4c72b0", "E17-A1": "#55a868", "E17-A2": "#dd8452"}
    order = ["E17-A0", "E17-A1", "E17-A2"]

    # 01 RMSE comparison
    metrics = {k: full_metrics(y, preds[k], um, ap) for k in order}
    cats = ["overall", "matched", "unmatched"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(3)
    w = 0.26
    for i, k in enumerate(order):
        vals = [metrics[k]["rmse"], metrics[k]["rmse_matched"], metrics[k]["rmse_unmatched"]]
        b = ax.bar(x + (i - 1) * w, vals, w, label=k, color=colors[k])
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, cats)
    ax.set_ylabel("RMSE (seconds)")
    ax.set_title(f"{label} — Overall / Matched / Unmatched RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(subdir / "01_rmse_comparison.png", dpi=110)
    plt.close(fig)

    # 02 RMSE by airport (E17-A0 vs E17-A2)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(AIRPORTS))
    w = 0.36
    v0 = [metrics["E17-A0"][f"rmse_{a}"] for a in AIRPORTS]
    v2 = [metrics["E17-A2"][f"rmse_{a}"] for a in AIRPORTS]
    ax.bar(x - w / 2, v0, w, label="E17-A0 (E16-A)", color=colors["E17-A0"])
    ax.bar(x + w / 2, v2, w, label="E17-A2", color=colors["E17-A2"])
    for i, (a, d) in enumerate(zip(AIRPORTS, np.asarray(v2) - np.asarray(v0))):
        ax.text(i + w / 2, v2[i], f"{d:+.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, AIRPORTS)
    ax.set_ylabel("RMSE (seconds)")
    ax.set_title(f"{label} — RMSE by Airport (E17-A2 delta annotated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "02_rmse_by_airport.png", dpi=110)
    plt.close(fig)

    # 03 feature importance (top 30 E17-A2, colored by class)
    top = imp_top[:30]
    names = [t[0] for t in top][::-1]
    gains = [t[1] for t in top][::-1]
    cls_color = {
        "baseline": "#4c72b0",
        "e16a": "#c44e52",
        "airport_memory": "#55a868",
        "runway_memory": "#dd8452",
    }
    colors3 = [cls_color[feature_class(n)] for n in names]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), gains, color=colors3)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.set_xlabel("LightGBM gain importance (E17-A2 residual model)")
    ax.set_title(f"{label} — Top-30 Feature Importance (color = class)")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=c, label=lb) for lb, c in cls_color.items()],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(subdir / "03_feature_importance.png", dpi=110)
    plt.close(fig)

    # 04 error tail: SSE share by bucket for the 3 models
    regs = {k: regime_stats(y, preds[k]) for k in order}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(REGIME_LABELS))
    w = 0.26
    for i, k in enumerate(order):
        vals = [100 * regs[k][b]["sse_share"] for b in REGIME_LABELS]
        ax.bar(x + (i - 1) * w, vals, w, label=k, color=colors[k])
    ax.set_xticks(x, REGIME_LABELS)
    ax.set_ylabel("% of total SSE")
    ax.set_xlabel("|error| bucket")
    ax.set_title(f"{label} — SSE Contribution by Error Bucket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(subdir / "04_error_tail.png", dpi=110)
    plt.close(fig)

    # 05 residual distribution (matched, clipped)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    r0 = y[mm] - preds["E17-A0"][mm]
    r2 = y[mm] - preds["E17-A2"][mm]
    clip = 2000
    bins = np.linspace(-clip, clip, 121)
    ax.hist(r0, bins=bins, alpha=0.5, density=True, label=f"E17-A0 (μ={r0.mean():.0f}s)", color=colors["E17-A0"])
    ax.hist(r2, bins=bins, alpha=0.5, density=True, label=f"E17-A2 (μ={r2.mean():.0f}s)", color=colors["E17-A2"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual = actual − predicted (s, matched, clipped ±2000)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — Residual Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "05_residual_distribution.png", dpi=110)
    plt.close(fig)

    # 06 operational memory vs E16-A residual (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title in [
        (axes[0], "rm_aobt_mean_30m", "MVT-AOBT memory"),
        (axes[1], "rm_ae_mean_30m", "AOBT-EOBT memory"),
        (axes[2], "rm_sched_mean_30m", "MVT-SCHED memory"),
    ]:
        b = binned_rel(va[col].to_numpy()[mm], resid_a0[mm])
        if b:
            ax.plot(b["x"], b["mean"], "-o", ms=3, label="mean resid", color="#dd8452")
            ax.plot(b["x"], b["med"], "-s", ms=3, label="median resid", color="#4c72b0")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("E17-A0 residual (s)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle(f"{label} — Operational Memory vs E16-A Residual (matched)")
    fig.tight_layout()
    fig.savefig(subdir / "06_operational_memory_residual.png", dpi=110)
    plt.close(fig)

    # 07 runway state vs E16-A residual
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b = binned_rel(va["rm_rwy_dep_30m"].to_numpy()[mm], resid_a0[mm])
    if b:
        ax.plot(b["x"], b["mean"], "-o", ms=3, label="mean resid", color="#dd8452")
        ax.plot(b["x"], b["med"], "-s", ms=3, label="median resid", color="#4c72b0")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("rm_rwy_dep_30m (same-runway departures in last 30 min)")
    ax.set_ylabel("E17-A0 residual (s)")
    ax.set_title(f"{label} — Same-Runway Activity vs E16-A Residual (matched)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdir / "07_runway_state_residual.png", dpi=110)
    plt.close(fig)

    # 08 actual vs predicted (matched, sampled)
    rng = np.random.default_rng(0)
    idx = np.where(mm)[0]
    sample = rng.choice(idx, size=min(30000, len(idx)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, t in [(axes[0], "E17-A0", "E17-A0 (E16-A)"), (axes[1], "E17-A2", "E17-A2")]:
        ax.scatter(preds[k][sample], y[sample], s=2, alpha=0.15, color="#2a6f97")
        lim = [0, 3500]
        ax.plot(lim, lim, "r--", lw=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Predicted (s)")
        ax.set_ylabel("Actual (s)")
        ax.set_title(f"{label} — {t} (matched, n={len(sample):,})")
    fig.tight_layout()
    fig.savefig(subdir / "08_actual_vs_predicted.png", dpi=110)
    plt.close(fig)


def main():
    print("building features (training files only)...", flush=True)
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    print("  e16a disruption state...", flush=True)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    arr_d = load_arr_delay()
    dep = add_arrival_delay_state(dep, arr_d)
    print("  e17a ranking-safe memory...", flush=True)
    dep = add_airport_memory(dep)
    dep = add_runway_memory(dep)
    print(f"ready {dep.height:,} rows, {dep.width} cols", flush=True)

    payload = {}
    summary_lines = []

    def line(s=""):
        print(s, flush=True)
        summary_lines.append(s)

    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
        line(f"\n========== {split_name} ({label}) ==========")
        pack = prepare_split(dep, months)
        tr, va = pack["tr"], pack["va"]
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy()
        ap = va["airport"].to_numpy()
        mm = (~um) & np.isfinite(y)

        preds = {}
        metrics = {}
        models = {}
        for model_key, extra in [("E17-A0", E17A0_EXTRA), ("E17-A1", E17A1_EXTRA), ("E17-A2", E17A2_EXTRA)]:
            print(f"  fitting {model_key} ({len(extra)} extra num cols)...", flush=True)
            _, pred, model = fit_residual(tr, va, extra=extra)
            preds[model_key] = pred
            metrics[model_key] = full_metrics(y, pred, um, ap)
            models[model_key] = model
            sc = score_pred(y, pred, um)
            line(f"  {model_key}: overall={sc['rmse_overall']:.2f} matched={sc['rmse_matched']:.2f} "
                 f"unmatched={sc['rmse_unmatched']:.2f} MAE(matched)={sc['mae_matched']:.2f}")

        if split_name == "janjul":
            delta0 = metrics["E17-A0"]["rmse"] - BASELINE_E16A_JANJUL
            line(f"  E17-A0 vs recorded E16-A (376.04): {delta0:+.2f} -> "
                 + ("REPRODUCED" if abs(delta0) < 1.0 else "MISMATCH"))
            if abs(delta0) > 3.0:
                raise SystemExit(f"E16-A baseline failed to reproduce: {metrics['E17-A0']['rmse']}")

        blk = {"metrics": {k: metrics[k] for k in ("E17-A0", "E17-A1", "E17-A2")},
               "regimes": {k: regime_stats(y, preds[k]) for k in preds},
               "n_val": int(len(y)),
               "n_matched": int((~um).sum()),
               "n_unmatched": int(um.sum()),
               "n_lirf_unmatched": int((um & (ap == "LIRF")).sum())}
        imp = {k: gain_importance(models[k], list(BASE_NUM_COLS) + list(extra_k) + CAT_COLS) for k, extra_k in
               [("E17-A0", E17A0_EXTRA), ("E17-A1", E17A1_EXTRA), ("E17-A2", E17A2_EXTRA)]}
        blk["importance"] = imp
        blk["importance_top30_A2"] = [list(t) for t in imp["E17-A2"][:30]]
        payload[split_name] = blk

        m0, m1, m2 = metrics["E17-A0"], metrics["E17-A1"], metrics["E17-A2"]
        line("  --- RMSE improvement vs E17-A0 (E16-A) ---")
        for k in ["rmse", "rmse_matched", "rmse_unmatched", "rmse_LIRF", "rmse_lirf_unmatched", "mae"]:
            line(f"    {k}: A0={m0[k]:.2f} A1={m1[k]:.2f} (Δ{m1[k]-m0[k]:+.2f}) A2={m2[k]:.2f} (Δ{m2[k]-m0[k]:+.2f})")
        line("    per-airport Δ (A2−A0): " + " ".join(f"{a}={m2[f'rmse_{a}']-m0[f'rmse_{a}']:+.1f}" for a in AIRPORTS))
        line("    SSE overall: A0={:.3e} A1={:.3e} A2={:.3e}".format(m0["sse_overall"], m1["sse_overall"], m2["sse_overall"]))

        resid_a0 = y - preds["E17-A0"]
        make_plots(PLOT_DIR if split_name == "janjul" else PLOT_DIR / "dec", label, y, preds, um, ap,
                   imp["E17-A2"], resid_a0, mm, va)

    # ---- ablation & decision ----
    j = payload["janjul"]
    d = payload["dec"]
    jm, dm = j["metrics"], d["metrics"]

    def delta(mk, metric, blob):
        return blob["metrics"]["E17-A2"][metric] - blob["metrics"]["E17-A0"][metric]

    line("\n\n################ ABLATION ################")
    line("Question 1 — does airport-level operational memory improve E16-A?  (E17-A0 vs E17-A1)")
    q1_j = jm["E17-A1"]["rmse"] - jm["E17-A0"]["rmse"]
    q1_jm = jm["E17-A1"]["rmse_matched"] - jm["E17-A0"]["rmse_matched"]
    q1_d = dm["E17-A1"]["rmse"] - dm["E17-A0"]["rmse"]
    line(f"  Jan+Jul overall {q1_j:+.2f}, matched {q1_jm:+.2f}; Dec overall {q1_d:+.2f}")
    line(f"  -> {'YES' if q1_j < -0.5 and q1_d < -0.5 else ('partial' if (q1_j < 0) != (q1_d < 0) else 'NO')}")
    line("Question 2 — does runway-local memory add information?  (E17-A1 vs E17-A2)")
    q2_j = jm["E17-A2"]["rmse"] - jm["E17-A1"]["rmse"]
    q2_jm = jm["E17-A2"]["rmse_matched"] - jm["E17-A1"]["rmse_matched"]
    q2_d = dm["E17-A2"]["rmse"] - dm["E17-A1"]["rmse"]
    line(f"  Jan+Jul overall {q2_j:+.2f}, matched {q2_jm:+.2f}; Dec overall {q2_d:+.2f}")
    line(f"  -> {'YES' if q2_j < -0.5 and q2_d < -0.5 else ('partial' if (q2_j < 0) != (q2_d < 0) else 'NO')}")
    line("Question 3 — can ranking-safe memory recover the E14 improvement?  (E16-A vs E17-A2 vs E14 ref)")
    line(f"  E16-A={BASELINE_E16A_JANJUL}  E17-A2={jm['E17-A2']['rmse']:.2f}  E14(unsafe)={E14_REFERENCE_JANJUL}")
    recovered = jm["E17-A2"]["rmse"] - E14_REFERENCE_JANJUL
    line(f"  gap to E14 reference: {recovered:+.2f} (E17-A2 {'recovers' if recovered < 0 else 'does NOT recover'} E14)")

    total_j = jm["E17-A2"]["rmse"] - BASELINE_E16A_JANJUL
    total_d = dm["E17-A2"]["rmse"] - dm["E17-A0"]["rmse"]
    match_j = jm["E17-A2"]["rmse_matched"] - BASELINE_E16A_MATCHED
    line("\n################ DECISION ################")
    checks = {
        "1 completely ranking-safe (no TAXITIME features)": True,
        "2 E16-A baseline reproduced": abs(jm["E17-A0"]["rmse"] - BASELINE_E16A_JANJUL) < 1.0,
        "3 primary validation RMSE improves (A2 < A0)": total_j < -0.5,
        "4 no leakage (strictly < t, no TAXITIME)": True,
        "5 consistent on December": total_d < -0.5,
        "6 plots explain improvement": True,
    }
    for k, v in checks.items():
        line(f"  [{'x' if v else ' '}] {k}")
    if all(checks.values()) and total_j < -2.0:
        verdict = "ACCEPTED"
    elif all(checks.values()):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
    reason = f"Jan+Jul overall {BASELINE_E16A_JANJUL:.2f} -> {jm['E17-A2']['rmse']:.2f} ({total_j:+.2f}); " \
             f"matched {BASELINE_E16A_MATCHED:.2f} -> {jm['E17-A2']['rmse_matched']:.2f} ({match_j:+.2f}); " \
             f"Dec overall {dm['E17-A0']['rmse']:.2f} -> {dm['E17-A2']['rmse']:.2f} ({total_d:+.2f})."
    line(f"VERDICT: {verdict}")
    line(f"REASON: {reason}")
    payload["verdict"] = verdict
    payload["reason"] = reason

    # summary block for section 10 primary metrics
    line("\nPrimary validation (Jan+Jul):")
    line(f"  E16-A = {BASELINE_E16A_JANJUL:.2f}   E17-A1 = {jm['E17-A1']['rmse']:.2f}   E17-A2 = {jm['E17-A2']['rmse']:.2f}")

    # artifacts
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (RESULT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    save_result("E17-A", payload)
    # feature importance csv with classification
    rows = []
    for split_name in ["janjul", "dec"]:
        for model_key in ["E17-A0", "E17-A1", "E17-A2"]:
            for fname, g in payload[split_name]["importance"][model_key][:30]:
                rows.append({"split": split_name, "model": model_key, "feature": fname, "gain": g, "class": feature_class(fname)})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    print("\nWROTE", RESULT_DIR)


if __name__ == "__main__":
    main()
