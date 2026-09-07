"""E19: RMSE tail reduction via causal local queue-state features.

Baseline = E18-H (matched-only geo_mean + LIRF-override rows dropped from
residual train + LIRF MVT-SCHED override). Identical LightGBM L2, seed and
early-stopping as E18-H. Variants add ranking-safe, strictly-causal (< t)
local-queue feature families (A..F). No loss change, no HP change.

Primary metric: Jan+Jul overall RMSE (E18-H 372.36). No ranking/submitting.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from common import (  # noqa: E402
    AIRPORTS,
    add_causal_rolling,
    load_dep,
    mae,
    rmse,
    save_result,
)
from run_e16a import MODEL_DISRUPT_COLS, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS  # noqa: E402
from run_e18_unmatched_specialist import fit_residual, prepare_split  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from e19_features import (  # noqa: E402
    A_COLS,
    B_COLS,
    C_COLS,
    D_COLS,
    E_COLS,
    F_COLS,
    add_interactions,
    add_neighbor,
    add_pressure,
    add_runway_queue,
    attach_shock,
)

RES = HERE / "results" / "E19"
PLOTS = RES / "plots"
for d in (RES, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
BASELINE_OVERALL = 372.36
BASELINE_MATCHED = 250.98
BASELINE_DEC = 238.01

BASE_EXTRA = MODEL_DISRUPT_COLS
VARIANTS = {
    "E19-0": BASE_EXTRA,                          # E18-H reproduction
    "E19-A": BASE_EXTRA + A_COLS,                  # neighbour state
    "E19-B": BASE_EXTRA + B_COLS,                  # same-runway queue
    "E19-C": BASE_EXTRA + C_COLS,                  # delay shock
    "E19-D": BASE_EXTRA + D_COLS,                  # queue/arrival pressure
    "E19-E": BASE_EXTRA + A_COLS + B_COLS + C_COLS + D_COLS + E_COLS,  # full state
    "E19-F": BASE_EXTRA + A_COLS + B_COLS + C_COLS + D_COLS + E_COLS + F_COLS,  # + tail-aware
}
FAMILY_OF = {}
for fam, cols in [("A", A_COLS), ("B", B_COLS), ("C", C_COLS), ("D", D_COLS),
                  ("E", E_COLS), ("F", F_COLS)]:
    for c in cols:
        FAMILY_OF[c] = f"E19-{fam}"
for c in MODEL_DISRUPT_COLS:
    FAMILY_OF[c] = "e16a"
FAMILY_OF.setdefault("", "baseline")

ORDER = ["E19-0", "E19-A", "E19-B", "E19-C", "E19-D", "E19-E", "E19-F"]
COLORS = {"E19-0": "#4c72b0", "E19-A": "#2a9d8f", "E19-B": "#e76f51", "E19-C": "#c44e52",
          "E19-D": "#9370db", "E19-E": "#55a868", "E19-F": "#dd8452"}

TAIL_ABS = [180, 240, 300, 360]
TOP_PCT = [1, 2, 5, 10]


def tail_metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    r = y - p
    sse_all = float(np.sum(r**2))
    out = {"n": int(m.sum()), "sse": sse_all}
    for th in TAIL_ABS:
        for sign, mask in [("abs", np.abs(r) > th), ("pos", r > th)]:
            k = f"{sign}_{th}"
            if mask.any():
                out[k] = {
                    "n": int(mask.sum()),
                    "rmse": rmse(y[mask], p[mask]),
                    "mae": mae(y[mask], p[mask]),
                    "sse": float(np.sum(r[mask] ** 2)),
                    "sse_share": float(np.sum(r[mask] ** 2)) / sse_all if sse_all else 0.0,
                }
            else:
                out[k] = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "sse": 0.0, "sse_share": 0.0}
    order = np.argsort(-np.abs(r))
    cs = np.cumsum(r[order] ** 2)
    for pct in TOP_PCT:
        k = int(np.ceil(pct / 100.0 * m.sum()))
        out[f"top{pct}pct_sse_share"] = cs[k - 1] / sse_all if sse_all else 0.0
    return out


def slice_rmse_mae(y, p, mask):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    return {"n": int(m.sum()), "rmse": rmse(y[m], p[m]), "mae": mae(y[m], p[m])}


def binned_rel(x, y, nb=12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 500:
        return None
    xm, ym = x[m], y[m]
    order = np.argsort(xm)
    q = np.array_split(xm[order], nb)
    qy = np.array_split(ym[order], nb)
    return {"x": [v.mean() for v in q], "mean": [v.mean() for v in qy], "med": [np.median(v) for v in qy]}


def make_plots(label: str, y, preds, um, ap, resid0, va, tail_by_ap, tail_by_rwy):
    y = np.asarray(y, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    mm = (~um) & np.isfinite(y)
    p0, pf = preds["E19-0"], preds["E19-F"]
    r0m, rfm = y[mm] - p0[mm], y[mm] - pf[mm]

    # residual distribution (matched)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(-2000, 2000, 121)
    ax.hist(r0m, bins=bins, alpha=0.5, density=True, label=f"E18-H (μ={r0m.mean():.0f}s)", color=COLORS["E19-0"])
    ax.hist(rfm, bins=bins, alpha=0.5, density=True, label=f"E19-F (μ={rfm.mean():.0f}s)", color=COLORS["E19-F"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual (s, matched, clipped ±2000)"); ax.set_ylabel("Density")
    ax.set_title(f"{label} — Residual distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "residual_distribution.png", dpi=110); plt.close(fig)

    # tail SSE (top-x% and thresholds)
    tm0, tmf = tail_metrics(y, p0), tail_metrics(y, pf)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(TOP_PCT))
    v0 = [tm0[f"top{p}pct_sse_share"] * 100 for p in TOP_PCT]
    vf = [tmf[f"top{p}pct_sse_share"] * 100 for p in TOP_PCT]
    w = 0.36
    ax.bar(x - w / 2, v0, w, label="E18-H", color=COLORS["E19-0"])
    ax.bar(x + w / 2, vf, w, label="E19-F", color=COLORS["E19-F"])
    for i in range(len(TOP_PCT)):
        ax.text(i - w / 2, v0[i], f"{v0[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, vf[i], f"{vf[i]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"top {p}%" for p in TOP_PCT])
    ax.set_ylabel("% of total SSE"); ax.set_title(f"{label} — SSE share of largest errors")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "tail_sse.png", dpi=110); plt.close(fig)

    # queue state vs residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, t in [
        (axes[0], "e19_qp", "queue pressure × delay shock (e19_qp)"),
        (axes[1], "e19_shock_p90_5", "neighbour P90 delay shock"),
    ]:
        b = binned_rel(va[col].to_numpy()[mm], r0m)
        if b:
            ax.plot(b["x"], b["mean"], "-o", ms=3, label="mean E18-H residual", color=COLORS["E19-F"])
            ax.plot(b["x"], b["med"], "-s", ms=3, label="median residual", color=COLORS["E19-0"])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel(col); ax.set_ylabel("E18-H residual (s)"); ax.set_title(t); ax.legend(fontsize=7)
    fig.suptitle(f"{label} — Queue-state vs residual (matched)"); fig.tight_layout()
    fig.savefig(PLOTS / "queue_state_vs_residual.png", dpi=110); plt.close(fig)

    # tail by airport
    if tail_by_ap is not None and len(tail_by_ap):
        fig, ax = plt.subplots(figsize=(10, 4.4))
        df = tail_by_ap.sort_values("sse0", ascending=False)
        x = np.arange(len(df))
        v0 = (df["sse0"] / df["sse0"].sum() * 100).to_numpy()
        vf = (df["sseF"] / df["sseF"].sum() * 100).to_numpy()
        w = 0.36
        ax.bar(x - w / 2, v0, w, label="E18-H", color=COLORS["E19-0"])
        ax.bar(x + w / 2, vf, w, label="E19-F", color=COLORS["E19-F"])
        ax.set_xticks(x, df["airport"], rotation=45, ha="right")
        ax.set_ylabel("% of tail SSE (matched >300 s)"); ax.set_title(f"{label} — Tail SSE by airport (matched >300 s)")
        ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "tail_by_airport.png", dpi=110); plt.close(fig)

    # tail by runway (top 15 runways by tail SSE)
    if tail_by_rwy is not None and len(tail_by_rwy):
        fig, ax = plt.subplots(figsize=(10, 4.4))
        df = tail_by_rwy.sort_values("sse0", ascending=False).head(15)
        x = np.arange(len(df))
        v0 = (df["sse0"] / df["sse0"].sum() * 100).to_numpy()
        vf = (df["sseF"] / df["sseF"].sum() * 100).to_numpy()
        w = 0.36
        ax.bar(x - w / 2, v0, w, label="E18-H", color=COLORS["E19-0"])
        ax.bar(x + w / 2, vf, w, label="E19-F", color=COLORS["E19-F"])
        ax.set_xticks(x, [f"{a}" for a in df["runway"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("% of tail SSE (matched >300 s)"); ax.set_title(f"{label} — Tail SSE by runway (top 15)")
        ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "tail_by_runway.png", dpi=110); plt.close(fig)


def json_conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    raise TypeError(type(o))


def main():
    t0 = datetime.now(timezone.utc)
    print("building base features...", flush=True)
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    from run_e16a import add_arrival_delay_state, add_push_disruption, add_rolling_quantiles
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    arr_d = load_arr_delay()
    dep = add_arrival_delay_state(dep, arr_d)

    print("attaching E19 local queue-state features...", flush=True)
    dep = add_neighbor(dep)
    dep = add_runway_queue(dep)
    dep = add_pressure(dep, arr_d)
    print(f"ready {dep.height:,} rows, {dep.width} cols", flush=True)

    payload = {}

    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
        print(f"\n========== {split_name} ==========", flush=True)
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        tr = attach_shock(tr, tr)
        va = attach_shock(tr, va)
        tr = add_interactions(tr)
        va = add_interactions(va)

        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy()
        ap = va["airport"].to_numpy()
        mm = (~um) & np.isfinite(y)

        preds, models, metrics = {}, {}, {}
        for vk, extra in VARIANTS.items():
            print(f"  fitting {vk} ({len(extra)} extra cols)...", flush=True)
            _, pred, model = fit_residual(tr, va, extra=extra, drop_override_from_train=True)
            preds[vk] = pred
            models[vk] = model
            tm = tail_metrics(y, pred)
            lirf = (um & (ap == "LIRF"))
            nlu = um & (ap != "LIRF")
            metrics[vk] = {
                "overall": tm and rmse(y, pred),
                "n_val": tm["n"], "sse": tm["sse"],
                "rmse": rmse(y, pred), "mae": mae(y, pred),
                "matched": slice_rmse_mae(y, pred, ~um),
                "unmatched": slice_rmse_mae(y, pred, um),
                "lirf_unmatched": slice_rmse_mae(y, pred, lirf),
                "non_lirf_unmatched": slice_rmse_mae(y, pred, nlu),
                "tail": tail_metrics(y, pred),
            }
            r0 = y[mm] - preds["E19-0"][mm]
            r = y[mm] - pred[mm]
            print(f"    {vk}: overall={rmse(y, pred):.2f} matched={metrics[vk]['matched']['rmse']:.2f} "
                  f">300s SSE-share={metrics[vk]['tail']['abs_300']['sse_share']:.3f}", flush=True)

        # per-airport + tail per airport/runway (E19-0 vs E19-F)
        tail_mask = mm & (np.abs(y - preds["E19-F"]) > 300)
        rows_ap, rows_rwy = [], []
        for a in AIRPORTS:
            sel = mm & (ap == a)
            if not sel.any():
                continue
            rows_ap.append({
                "airport": a,
                "rmse0": rmse(y[sel], preds["E19-0"][sel]),
                "rmseF": rmse(y[sel], preds["E19-F"][sel]),
                "rmseE": rmse(y[sel], preds["E19-E"][sel]),
                "tail0": rmse(y[sel & tail_mask], preds["E19-0"][sel & tail_mask]) if (sel & tail_mask).any() else np.nan,
                "tailF": rmse(y[sel & tail_mask], preds["E19-F"][sel & tail_mask]) if (sel & tail_mask).any() else np.nan,
                "sse0": float(np.sum((y[sel] - preds["E19-0"][sel]) ** 2)),
                "sseF": float(np.sum((y[sel] - preds["E19-F"][sel]) ** 2)),
            })
        rw = va["RUNWAY_mvt"].fill_null("NA").to_numpy()
        sel_all = mm
        for r in sorted(set(rw[sel_all])):
            sel = sel_all & (rw == r)
            if sel.sum() < 200:
                continue
            rows_rwy.append({
                "runway": r,
                "n": int(sel.sum()),
                "rmse0": rmse(y[sel], preds["E19-0"][sel]),
                "rmseF": rmse(y[sel], preds["E19-F"][sel]),
                "sse0": float(np.sum((y[sel] - preds["E19-0"][sel]) ** 2)),
                "sseF": float(np.sum((y[sel] - preds["E19-F"][sel]) ** 2)),
            })
        per_ap = pd.DataFrame(rows_ap)
        per_rwy = pd.DataFrame(rows_rwy)

        importance = {}
        for vk in ORDER:
            names = list(model_feature_names(models[vk], VARIANTS[vk]))
            imp = sorted(
                zip(names, models[vk].booster_.feature_importance(importance_type="gain").tolist()),
                key=lambda t: -t[1],
            )
            importance[vk] = imp

        if split_name == "janjul":
            delta0 = metrics["E19-0"]["rmse"] - BASELINE_OVERALL
            if abs(delta0) > 1.0:
                raise SystemExit(f"E18-H baseline failed to reproduce: {metrics['E19-0']['rmse']}")
            print(f"  E19-0 baseline check: {metrics['E19-0']['rmse']:.2f} (Δ {delta0:+.2f} vs 372.36)", flush=True)

        blk = {"metrics": metrics, "importance": importance,
               "per_airport": per_ap, "per_runway": per_rwy}
        payload[split_name] = blk

        if split_name == "janjul":
            tail_by_ap, tail_by_rwy = per_ap, per_rwy
        make_plots(label, y, preds, um, ap, y - preds["E19-0"], va, per_ap, per_rwy)

    # ------- csv exports (janjul primary) -------
    jm = payload["janjul"]["metrics"]
    rows_m = []
    for vk in ORDER:
        m = jm[vk]
        rows_m.append({
            "variant": vk, "overall_rmse": m["rmse"], "mae": m["mae"],
            "matched_rmse": m["matched"]["rmse"], "matched_mae": m["matched"]["mae"],
            "unmatched_rmse": m["unmatched"]["rmse"],
            "lirf_unmatched_rmse": m["lirf_unmatched"]["rmse"],
            "non_lirf_unmatched_rmse": m["non_lirf_unmatched"]["rmse"],
            "sse": m["sse"], "n": m["n_val"],
            "abs300_sse_share": m["tail"]["abs_300"]["sse_share"],
            "pos300_sse_share": m["tail"]["pos_300"]["sse_share"],
            "top1pct_sse_share": m["tail"]["top1pct_sse_share"],
            "top5pct_sse_share": m["tail"]["top5pct_sse_share"],
        })
    pd.DataFrame(rows_m).to_csv(RES / "metrics.csv", index=False)

    rows_abl = []
    for vk in ["E19-A", "E19-B", "E19-C", "E19-D", "E19-E", "E19-F"]:
        d = jm[vk]["rmse"] - jm["E19-0"]["rmse"]
        dm = jm[vk]["matched"]["rmse"] - jm["E19-0"]["matched"]["rmse"]
        dd = payload["dec"]["metrics"][vk]["rmse"] - payload["dec"]["metrics"]["E19-0"]["rmse"]
        rows_abl.append({
            "variant": vk, "janjul_overall": jm[vk]["rmse"], "delta_vs_E18H": d,
            "janjul_matched": jm[vk]["matched"]["rmse"], "delta_matched": dm,
            "dec_overall": payload["dec"]["metrics"][vk]["rmse"], "delta_dec": dd,
            "tail_abs300_delta_pp": (jm[vk]["tail"]["abs_300"]["sse_share"] - jm["E19-0"]["tail"]["abs_300"]["sse_share"]) * 100,
        })
    pd.DataFrame(rows_abl).to_csv(RES / "ablation.csv", index=False)
    payload["janjul"]["per_airport"].to_csv(RES / "per_airport.csv", index=False)

    rows_tail = []
    for vk in ORDER:
        t = jm[vk]["tail"]
        row = {"variant": vk}
        for k, v in t.items():
            if isinstance(v, dict):
                for kk in ("n", "rmse", "mae", "sse", "sse_share"):
                    row[f"{k}_{kk}"] = v[kk]
            else:
                row[k] = v
        rows_tail.append(row)
    pd.DataFrame(rows_tail).to_csv(RES / "tail_metrics.csv", index=False)

    rows_fi = []
    for vk in ORDER:
        for f, g in payload["janjul"]["importance"][vk][:40]:
            rows_fi.append({"variant": vk, "feature": f, "gain": g,
                            "family": FAMILY_OF.get(f, "baseline")})
    pd.DataFrame(rows_fi).to_csv(RES / "feature_importance.csv", index=False)

    payload["verdict"] = decision(payload, jm)
    (RES / "E19.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    from copy import deepcopy

    save_payload = deepcopy(payload)
    for split_name in ("janjul", "dec"):
        save_payload[split_name]["per_airport"] = payload[split_name]["per_airport"].to_dict("records")
        save_payload[split_name]["per_runway"] = payload[split_name]["per_runway"].to_dict("records")
    save_result("E19", save_payload)
    write_summary(payload)
    print("WROTE", RES)


def model_feature_names(model, extra):
    return list(NUM_COLS) + [c for c in extra if c not in NUM_COLS] + CAT_COLS


def decision(payload, jm):
    d0 = jm["E19-0"]["rmse"]
    best = min(jm, key=lambda k: jm[k]["rmse"])
    dbest = jm[best]["rmse"] - d0
    dbest_m = jm[best]["matched"]["rmse"] - jm["E19-0"]["matched"]["rmse"]
    dec = payload["dec"]["metrics"]
    dd = dec[best]["rmse"] - dec["E19-0"]["rmse"]
    if -1 < dbest < 1:
        verdict = "REJECT (sub-1s gain)"
    elif -3 <= dbest <= -1:
        verdict = "WEAK (1-3s gain, unstable)"
    elif -7 < dbest < -3:
        verdict = "MEANINGFUL (3-7s gain)"
    elif -10 <= dbest <= -7:
        verdict = "STRONG (7-10s gain)"
    elif dbest < -10:
        verdict = "MAJOR (>10s gain)"
    else:
        verdict = "REJECT"
    if dd > 5:
        verdict += " — REJECT for December regression"
    reason = (f"best {best}: Jan+Jul {d0:.2f} -> {jm[best]['rmse']:.2f} (Δ{dbest:+.2f}, "
              f"matched {jm['E19-0']['matched']['rmse']:.2f} -> {jm[best]['matched']['rmse']:.2f} Δ{dbest_m:+.2f}); "
              f"Dec {dec['E19-0']['rmse']:.2f} -> {dec[best]['rmse']:.2f} (Δ{dd:+.2f}).")
    return {"verdict": verdict, "best": best, "reason": reason,
            "janjul_delta": dbest, "dec_delta": dd}


def write_summary(payload):
    lines = []
    a = lines.append
    jm = payload["janjul"]["metrics"]
    dm = payload["dec"]["metrics"]
    a("# E19 — RMSE tail reduction (causal local queue state)")
    a("")
    a(f"Baseline E18-H: Jan+Jul overall {jm['E19-0']['rmse']:.2f} / matched {jm['E19-0']['matched']['rmse']:.2f}; "
      f"Dec {dm['E19-0']['rmse']:.2f} / {dm['E19-0']['matched']['rmse']:.2f}.")
    a("")
    a("| Variant | Jan+Jul overall | Δ vs E18-H | matched | >300s SSE share | Dec overall | Δ Dec |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for vk in ORDER:
        m, d = jm[vk], dm[vk]
        a(f"| {vk} | {m['rmse']:.2f} | {m['rmse']-jm['E19-0']['rmse']:+.2f} | {m['matched']['rmse']:.2f} | "
          f"{100*m['tail']['abs_300']['sse_share']:.1f}% | {d['rmse']:.2f} | {d['rmse']-dm['E19-0']['rmse']:+.2f} |")
    a("")
    a(f"## Decision: {payload['verdict']['verdict']}")
    a("")
    a(payload["verdict"]["reason"])
    a("")
    a("Tail concentration (Jan+Jul, matched>300 s SSE share is the key row above). Full tail table in `tail_metrics.csv`.")
    a("")
    a("Top features per family (E19-F, Jan+Jul):")
    imp = payload["janjul"]["importance"]["E19-F"]
    fam = {}
    for f, g in imp:
        fam.setdefault(FAMILY_OF.get(f, "baseline"), []).append((f, g))
    for k in ["e16a", "E19-A", "E19-B", "E19-C", "E19-D", "E19-E", "E19-F", "baseline"]:
        if k in fam and fam[k]:
            top = ", ".join(f"{f}" for f, _ in fam[k][:4])
            a(f"- {k}: {top}")
    a("")
    a("Artifacts: metrics.csv, ablation.csv, per_airport.csv, tail_metrics.csv, feature_importance.csv, plots/.")
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
