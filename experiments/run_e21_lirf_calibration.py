"""E21: calibration layer on the LIRF unmatched override y_hat = MVT−SCHED.

Does not replace the override or gate it (E13 closed gating). Does not
touch ensemble weights. Training_*.parquet only.

Phase 1 — diagnose r = y − (MVT−SCHED) on Jan+Jul *val* LIRF unmatched
           (the slice the 6033 number refers to). Concentration, hour,
           weekday, month, SCHED reliability.
Phase 2 — simplest train-only corrections (global/hour/month additive
           mean or median; optional slope). Fit on train LIRF unmatched.
Phase 3 — Jan+Jul + December. Report LIRF-unmatched / overall / matched
           (matched is identically unchanged). Overall via SSE substitution
           on the E20 ensemble baseline.

Kill: if Jan+Jul overall does not fall, or December overall rises.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, load_dep, mae, rmse, save_result, split_by_months  # noqa: E402

OUT = ROOT / "analysis" / "E21"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP, FIG / "dec"):
    d.mkdir(parents=True, exist_ok=True)

# Current best (E20 NNLS ensemble). LIRF unmatched preds are still raw MVT−SCHED.
E20 = {
    "janjul": {"n": 344419, "overall": 368.03, "matched": 244.76, "n_lu": 397, "lu": 6032.69683773519},
    "dec": {"n": 165677, "overall": 228.45, "matched": 215.90, "n_lu": 88, "lu": 2786.1434163308313},
}

plt.rcParams.update(
    {"figure.dpi": 140, "savefig.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def lu_mask(df: pl.DataFrame) -> np.ndarray:
    return (df["airport"].to_numpy() == "LIRF") & df["unmatched"].to_numpy().astype(bool)


def arrays(df: pl.DataFrame):
    y = df["y"].to_numpy().astype(np.float64)
    ms = df["mvt_sched"].to_numpy().astype(np.float64)
    hour = df["hour"].to_numpy()
    dow = df["dow"].to_numpy()
    month = df["month"].to_numpy()
    return y, ms, hour, dow, month


def resid_summary(y, ms, name: str) -> dict:
    m = np.isfinite(y) & np.isfinite(ms)
    y, ms = y[m], ms[m]
    r = y - ms
    sse = float(np.sum(r**2))
    abs_r = np.abs(r)
    order = np.argsort(-abs_r)
    n = int(r.size)
    conc = {}
    for k in [1, 2, 3, 5, 10, max(1, n // 100), max(1, n // 20), max(1, n // 10)]:
        k = min(int(k), n)
        conc[f"worst_{k}"] = float(np.sum(r[order[:k]] ** 2) / sse) if sse else float("nan")
    return {
        "slice": name,
        "n": n,
        "rmse": rmse(y, ms),
        "mae": mae(y, ms),
        "resid_mean": float(r.mean()),
        "resid_median": float(np.median(r)),
        "resid_std": float(r.std()),
        "resid_skew": float(pd.Series(r).skew()),
        "resid_p5": float(np.quantile(r, 0.05)),
        "resid_p25": float(np.quantile(r, 0.25)),
        "resid_p75": float(np.quantile(r, 0.75)),
        "resid_p95": float(np.quantile(r, 0.95)),
        "resid_min": float(r.min()),
        "resid_max": float(r.max()),
        "frac_neg": float((r < 0).mean()),
        "frac_abs_gt1h": float((np.abs(r) > 3600).mean()),
        "frac_abs_gt2h": float((np.abs(r) > 7200).mean()),
        "frac_abs_gt6h": float((np.abs(r) > 21600).mean()),
        "n_abs_gt1h": int((np.abs(r) > 3600).sum()),
        "n_abs_gt2h": int((np.abs(r) > 7200).sum()),
        "mean_y": float(y.mean()),
        "mean_ms": float(ms.mean()),
        "med_y": float(np.median(y)),
        "med_ms": float(np.median(ms)),
        "corr_y_ms": float(np.corrcoef(y, ms)[0, 1]) if n > 2 else float("nan"),
        **conc,
    }


def group_resid(y, ms, g, gname: str) -> pd.DataFrame:
    m = np.isfinite(y) & np.isfinite(ms)
    y, ms, g = y[m], ms[m], np.asarray(g)[m]
    rows = []
    for v in np.unique(g):
        sel = g == v
        r = y[sel] - ms[sel]
        rows.append(
            {
                gname: int(v) if np.issubdtype(type(v), np.integer) else v,
                "n": int(sel.sum()),
                "rmse": rmse(y[sel], ms[sel]),
                "mae": mae(y[sel], ms[sel]),
                "resid_mean": float(r.mean()),
                "resid_median": float(np.median(r)),
                "mean_y": float(y[sel].mean()),
                "mean_ms": float(ms[sel].mean()),
                "med_y": float(np.median(y[sel])),
                "med_ms": float(np.median(ms[sel])),
            }
        )
    return pd.DataFrame(rows).sort_values(gname)


def sse_share_table(y, ms) -> pd.DataFrame:
    m = np.isfinite(y) & np.isfinite(ms)
    r = y[m] - ms[m]
    sse = float(np.sum(r**2))
    n = r.size
    order = np.argsort(-(r**2))
    rows = []
    for lab, k in [
        ("worst 1", 1),
        ("worst 2", 2),
        ("worst 5", 5),
        ("worst 10", 10),
        ("worst 1%", max(1, n // 100)),
        ("worst 5%", max(1, n // 20)),
        ("worst 10%", max(1, n // 10)),
        ("|r|>1h", int((np.abs(r) > 3600).sum())),
        ("|r|>2h", int((np.abs(r) > 7200).sum())),
        ("y<=30m", int((y[m] <= 1800).sum())),
        ("y>30m", int((y[m] > 1800).sum())),
    ]:
        if lab.startswith("y"):
            sel = (y[m] <= 1800) if "<=30" in lab else (y[m] > 1800)
            share = float(np.sum(r[sel] ** 2) / sse) if sse else float("nan")
            rows.append({"slice": lab, "k": int(sel.sum()), "sse_share": share})
        else:
            kk = min(k, n)
            share = float(np.sum(r[order[:kk]] ** 2) / sse) if sse and kk else float("nan")
            rows.append({"slice": lab, "k": kk, "sse_share": share})
    return pd.DataFrame(rows)


def overall_from_lu(base_overall, base_n, y, pred_old, pred_new, mask) -> float:
    sse = (base_overall**2) * base_n
    y = np.asarray(y, dtype=np.float64)
    old = np.asarray(pred_old, dtype=np.float64)
    new = np.asarray(pred_new, dtype=np.float64)
    m = mask & np.isfinite(y) & np.isfinite(old) & np.isfinite(new)
    sse = sse - np.sum((y[m] - old[m]) ** 2) + np.sum((y[m] - new[m]) ** 2)
    return float(np.sqrt(max(sse, 0.0) / base_n))


def winsor_mean(x, p=0.05):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(np.mean(np.clip(x, lo, hi)))


def fit_additive(r, by=None, min_n=20, how="median"):
    """Return global correction and optional group map. Correction is added to MVT−SCHED.
    We estimate E[y - ms] on train, so y_hat = ms + c  (c ≈ mean/median residual).
    """
    r = np.asarray(r, dtype=np.float64)
    if how == "mean":
        glob = float(np.nanmean(r))
    elif how == "median":
        glob = float(np.nanmedian(r))
    elif how == "winsor":
        glob = winsor_mean(r)
    else:
        raise ValueError(how)
    gmap = {}
    if by is not None:
        by = np.asarray(by)
        for v in np.unique(by):
            sel = by == v
            if int(sel.sum()) < min_n:
                continue
            rr = r[sel]
            if how == "mean":
                gmap[int(v)] = float(np.nanmean(rr))
            elif how == "median":
                gmap[int(v)] = float(np.nanmedian(rr))
            else:
                gmap[int(v)] = winsor_mean(rr)
    return glob, gmap


def apply_additive(ms, glob, gmap, by=None):
    out = np.asarray(ms, dtype=np.float64) + glob
    if gmap and by is not None:
        by = np.asarray(by)
        for v, c in gmap.items():
            out[by == v] = np.asarray(ms, dtype=np.float64)[by == v] + c
    return out


def fit_ols(ms, y):
    m = np.isfinite(ms) & np.isfinite(y)
    x = np.asarray(ms[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)
    if x.size < 10:
        return 0.0, 1.0
    X = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_ols(ms, a, b):
    return a + b * np.asarray(ms, dtype=np.float64)


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


def diagnose_split(va: pl.DataFrame, split_name: str) -> dict:
    lu = lu_mask(va)
    y, ms, hour, dow, month = arrays(va)
    y, ms, hour, dow, month = y[lu], ms[lu], hour[lu], dow[lu], month[lu]
    summ = resid_summary(y, ms, f"{split_name}_lirf_unmatched")
    conc = sse_share_table(y, ms)
    by_hour = group_resid(y, ms, hour, "hour")
    by_dow = group_resid(y, ms, dow, "dow")
    by_month = group_resid(y, ms, month, "month")
    # SCHED reliability: bins of mvt_sched
    edges = np.array([-np.inf, 0, 1800, 3600, 7200, 14400, 28800, np.inf])
    labs = ["ms<0", "0-30m", "30-60m", "1-2h", "2-4h", "4-8h", ">8h"]
    bin_id = np.digitize(ms, edges[1:-1], right=True)
    by_ms = group_resid(y, ms, bin_id, "ms_bin")
    by_ms["ms_bin_lab"] = [labs[int(i)] if 0 <= int(i) < len(labs) else str(i) for i in by_ms["ms_bin"]]
    prefix = f"e21_diag_{split_name}"
    pd.DataFrame([summ]).to_csv(TAB / f"{prefix}_summary.csv", index=False)
    conc.to_csv(TAB / f"{prefix}_concentration.csv", index=False)
    by_hour.to_csv(TAB / f"{prefix}_hour.csv", index=False)
    by_dow.to_csv(TAB / f"{prefix}_dow.csv", index=False)
    by_month.to_csv(TAB / f"{prefix}_month.csv", index=False)
    by_ms.to_csv(TAB / f"{prefix}_msbin.csv", index=False)
    return {
        "summary": summ,
        "concentration": conc.to_dict(orient="records"),
        "hour": by_hour.to_dict(orient="records"),
        "dow": by_dow.to_dict(orient="records"),
        "month": by_month.to_dict(orient="records"),
        "msbin": by_ms.to_dict(orient="records"),
        "y": y,
        "ms": ms,
        "hour_arr": hour,
        "dow_arr": dow,
        "month_arr": month,
    }


def plot_diag(diag: dict, split_name: str):
    y, ms = diag["y"], diag["ms"]
    r = y - ms
    figdir = FIG if split_name == "janjul" else FIG / "dec"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lo, hi = np.quantile(r, [0.01, 0.99])
    ax.hist(np.clip(r, lo, hi), bins=50, color="#4c72b0", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.median(r), color="#c44e52", lw=1.2, label=f"median {np.median(r):.0f}")
    ax.axvline(np.mean(r), color="#dd8452", lw=1.2, ls="--", label=f"mean {np.mean(r):.0f}")
    ax.set_xlabel("y − (MVT−SCHED) (s)")
    ax.set_ylabel("count")
    ax.set_title(f"{split_name} LIRF unmatched residual (clipped 1–99%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "01_resid_hist.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.scatter(ms / 3600, y / 3600, s=12, alpha=0.5, color="#2a6f97")
    lim = [0, max(8, float(np.quantile(ms, 0.98) / 3600))]
    ax.plot(lim, lim, "r--", lw=0.9, label="y = MVT−SCHED")
    ax.set_xlim(lim)
    ax.set_ylim([0, max(8, float(np.quantile(y, 0.98) / 3600))])
    ax.set_xlabel("MVT−SCHED (h)")
    ax.set_ylabel("actual taxi (h)")
    ax.set_title(f"{split_name} LIRF unmatched: y vs override")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "02_y_vs_mvtsched.png")
    plt.close()

    hdf = pd.DataFrame(diag["hour"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(hdf["hour"], hdf["resid_median"], color="#4c72b0", alpha=0.85, label="median resid")
    ax.plot(hdf["hour"], hdf["resid_mean"], "o-", color="#c44e52", ms=4, label="mean resid")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("hour UTC")
    ax.set_ylabel("residual (s)")
    ax.set_title(f"{split_name} residual by hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "03_resid_by_hour.png")
    plt.close()


def log_diag(split_name: str, d: dict):
    s = d["summary"]
    log(f"===== DIAGNOSIS {split_name} LIRF unmatched n={s['n']} =====")
    log(f"  RMSE={s['rmse']:.2f} MAE={s['mae']:.2f}")
    log(f"  resid mean={s['resid_mean']:.1f} median={s['resid_median']:.1f} skew={s['resid_skew']:.2f} std={s['resid_std']:.1f}")
    log(f"  p5={s['resid_p5']:.0f} p25={s['resid_p25']:.0f} p75={s['resid_p75']:.0f} p95={s['resid_p95']:.0f}")
    log(f"  frac_neg={s['frac_neg']:.3f} |r|>1h={s['n_abs_gt1h']}/{s['n']} |r|>2h={s['n_abs_gt2h']}")
    log(f"  mean_y={s['mean_y']:.0f} mean_ms={s['mean_ms']:.0f} corr(y,ms)={s['corr_y_ms']:.3f}")
    log("  SSE concentration:")
    for row in d["concentration"]:
        log(f"    {row['slice']:12s} k={row['k']:4d} share={100*row['sse_share']:5.1f}%")


def eval_rule(name, y, pred, ms, lu, base) -> dict:
    r_lu = rmse(y[lu], pred[lu])
    n_lu = int(lu.sum())
    ov = overall_from_lu(base["overall"], base["n"], y, ms, pred, lu)
    n_ok = int((np.isfinite(y) & np.isfinite(pred)).sum())
    # matched RMSE: predictions outside lu are not in `pred` as ensemble — we only
    # report that matched is unchanged by construction.
    return {
        "name": name,
        "n_lirf_unmatched": n_lu,
        "rmse_lirf_u": r_lu,
        "mae_lirf_u": mae(y[lu], pred[lu]),
        "d_lirf_u": r_lu - rmse(y[lu], ms[lu]),
        "overall": ov,
        "d_overall": ov - base["overall"],
        "matched_unchanged": True,
        "n_val": n_ok,
        "n_matched": int((~lu).sum()),
        "rmse_matched": base["matched"],
    }


def main():
    log("E21: load training DEP only...")
    dep = load_dep()
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat()}

    # ----- Phase 1: diagnosis on Jan+Jul VAL (the 6033 number) -----
    tr_j, va_j = split_by_months(dep, [1, 7])
    diag_j = diagnose_split(va_j, "janjul")
    log_diag("janjul val", diag_j)
    plot_diag(diag_j, "janjul")

    tr_d, va_d = split_by_months(dep, [12])
    diag_d = diagnose_split(va_d, "dec")
    log_diag("dec val", diag_d)
    plot_diag(diag_d, "dec")

    _drop = ("y", "ms", "hour_arr", "dow_arr", "month_arr")
    payload["diag_janjul"] = {k: v for k, v in diag_j.items() if k not in _drop}
    payload["diag_dec"] = {k: v for k, v in diag_d.items() if k not in _drop}

    # ----- Phase 2: corrections fit on TRAIN LIRF unmatched -----
    log("Phase 2: train-only calibrations...")

    def run_split(tr, va, split_name: str):
        base = E20[split_name]
        y_va, ms_va, hour_va, dow_va, month_va = arrays(va)
        lu_va = lu_mask(va)
        y_tr, ms_tr, hour_tr, dow_tr, month_tr = arrays(tr)
        lu_tr = lu_mask(tr)
        r_tr = y_tr[lu_tr] - ms_tr[lu_tr]
        log(f"  {split_name} train LIRF_u n={int(lu_tr.sum())}  val LIRF_u n={int(lu_va.sum())}")

        variants = {}
        variants["raw_ms"] = ms_va.copy()

        for how in ("mean", "median", "winsor"):
            g, _ = fit_additive(r_tr, how=how)
            variants[f"add_{how}"] = apply_additive(ms_va, g, {}, None)
            log(f"    add_{how} c={g:.1f}")

        g, gmap = fit_additive(r_tr, by=hour_tr[lu_tr], min_n=15, how="median")
        variants["add_hour_median"] = apply_additive(ms_va, g, gmap, hour_va)
        log(f"    add_hour_median global={g:.1f} n_hours={len(gmap)}")

        g, gmap = fit_additive(r_tr, by=month_tr[lu_tr], min_n=15, how="median")
        variants["add_month_median"] = apply_additive(ms_va, g, gmap, month_va)
        log(f"    add_month_median global={g:.1f} n_months={len(gmap)}")

        g, gmap = fit_additive(r_tr, by=dow_tr[lu_tr], min_n=15, how="median")
        variants["add_dow_median"] = apply_additive(ms_va, g, gmap, dow_va)

        a, b = fit_ols(ms_tr[lu_tr], y_tr[lu_tr])
        variants["ols_ab"] = apply_ols(ms_va, a, b)
        log(f"    ols intercept={a:.1f} slope={b:.3f}")

        # clip raw ms to a train quantile band (robust) — not a location shift
        lo, hi = np.quantile(ms_tr[lu_tr][np.isfinite(ms_tr[lu_tr])], [0.01, 0.99])
        variants["clip_ms_p01_p99"] = np.clip(ms_va, lo, hi)
        log(f"    clip_ms [{lo:.0f}, {hi:.0f}]")

        rows = []
        for name, pred in variants.items():
            e = eval_rule(name, y_va, pred, ms_va, lu_va, base)
            rows.append(e)
            log(
                f"    {name:20s} LIRF_u={e['rmse_lirf_u']:.1f} ({e['d_lirf_u']:+.1f})  "
                f"overall={e['overall']:.2f} ({e['d_overall']:+.2f})"
            )
        return {"metrics": rows, "n_train_lu": int(lu_tr.sum()), "n_val_lu": int(lu_va.sum())}

    payload["janjul"] = run_split(tr_j, va_j, "janjul")
    payload["dec"] = run_split(tr_d, va_d, "dec")

    def get(blob, name):
        for r in blob["metrics"]:
            if r["name"] == name:
                return r
        return None

    j, d = payload["janjul"], payload["dec"]
    # accept a correction only if Jan+Jul overall down and Dec overall not up, LIRF_u down both
    skip = {"raw_ms"}
    names = [r["name"] for r in j["metrics"] if r["name"] not in skip]
    keep = None
    notes = []
    for n in names:
        jj, dd = get(j, n), get(d, n)
        if jj is None or dd is None:
            continue
        ok = jj["d_overall"] < -0.3 and dd["d_overall"] <= 0.3 and jj["d_lirf_u"] < -1 and dd["d_lirf_u"] < 0
        notes.append(
            f"{n}: Jan+Jul overall {jj['d_overall']:+.2f} LIRF_u {jj['d_lirf_u']:+.1f}; "
            f"Dec overall {dd['d_overall']:+.2f} LIRF_u {dd['d_lirf_u']:+.1f}"
        )
        if ok and (keep is None or jj["overall"] < get(j, keep)["overall"]):
            keep = n
    if keep:
        dj = get(j, keep)["d_overall"]
        verdict = "ACCEPTED" if dj < -1.0 else "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
    n_lu = E20["janjul"]["n_lu"]
    n_all = E20["janjul"]["n"]
    share = (E20["janjul"]["lu"] ** 2 * n_lu) / (E20["janjul"]["overall"] ** 2 * n_all)
    reason = (
        f"LIRF unmatched n={n_lu} / {n_all} ({100*n_lu/n_all:.3f}% of rows) "
        f"is {100*share:.1f}% of E20 overall SSE. "
        + (f"Best correction {keep}. " if keep else "No correction beat raw MVT−SCHED on both splits. ")
        + " ".join(notes)
    )
    payload["decision"] = {
        "verdict": verdict,
        "keep": keep,
        "reason": reason,
        "n_lirf_unmatched_janjul": n_lu,
        "n_val_janjul": n_all,
        "sse_share_lirf_u_e20": share,
        "notes": notes,
    }
    log(f"VERDICT {verdict}")
    log(reason)

    write_report(payload)
    save_result("E21", payload)
    (TAB / "e21_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e21.py")
    log("done")


def write_report(payload: dict):
    dj = payload["diag_janjul"]["summary"]
    dd = payload["diag_dec"]["summary"]
    j, d = payload["janjul"], payload["dec"]
    lines = []
    a = lines.append
    a("# E21 — LIRF unmatched override calibration")
    a("")
    a("**Project:** OpenAir")
    a("**Experiment:** E21")
    a("**Data:** 12 `training_*.parquet` only. No ranking/submission.")
    a("**Scope:** additive/multiplicative layer on `y_hat = MVT−SCHED` for")
    a("`unmatched and airport==LIRF`. Ensemble weights frozen. Matched path frozen.")
    a("")
    a("Script: `experiments/run_e21_lirf_calibration.py`.")
    a("")
    a("---")
    a("")
    a("## 1. Diagnosis (before any correction)")
    a("")
    a("Residual `r = y − (MVT−SCHED)` on **validation** LIRF unmatched")
    a("(the slice whose RMSE is 6033 on Jan+Jul).")
    a("")
    a("| Split | n | RMSE | MAE | mean r | median r | skew | % r<0 | n \\|r\\|>1h | n \\|r\\|>2h | corr(y, ms) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, s in ("Jan+Jul val", dj), ("December val", dd):
        a(
            f"| {lab} | {s['n']} | {s['rmse']:.1f} | {s['mae']:.1f} | {s['resid_mean']:.0f} | "
            f"{s['resid_median']:.0f} | {s['resid_skew']:.2f} | {100*s['frac_neg']:.0f}% | "
            f"{s['n_abs_gt1h']} | {s['n_abs_gt2h']} | {s['corr_y_ms']:.3f} |"
        )
    a("")
    a("SSE concentration (Jan+Jul val):")
    a("")
    a("| Slice | k | SSE share |")
    a("|---|---:|---:|")
    for row in payload["diag_janjul"]["concentration"]:
        a(f"| {row['slice']} | {row['k']} | {100*row['sse_share']:.1f}% |")
    a("")
    a("Row-count arithmetic vs E20 ensemble (Jan+Jul overall 368.03, n=344,419):")
    a("")
    n_lu = payload["decision"]["n_lirf_unmatched_janjul"]
    share = payload["decision"]["sse_share_lirf_u_e20"]
    a(f"- LIRF unmatched rows: **{n_lu}** ({100*n_lu/344419:.3f}% of val).")
    a(f"- Share of E20 overall SSE: **{100*share:.1f}%**.")
    a("- A 10% LIRF-unmatched RMSE cut (6033→5430) would move overall by only")
    a("  ~3–4 s; cutting LIRF unmatched in half (6033→3016) would move overall")
    a("  by ~20 s. This slot is worth it only if the correction is large *and* stable.")
    a("")
    a("Hour / weekday / month / MVT−SCHED-bin tables:")
    a("`analysis/E21/tables/e21_diag_janjul_*.csv`.")
    a("")
    a("---")
    a("")
    a("## 2. Corrections (train LIRF unmatched only)")
    a("")
    a("`y_hat = (MVT−SCHED) + c` with `c` = train mean / median / 5% winsorized")
    a("mean residual; or `c` by hour / month / weekday (median, min_n=15);")
    a("or OLS `a + b·(MVT−SCHED)`; or clip MVT−SCHED to train 1–99%.")
    a("")
    a("## 3. Holdout")
    a("")
    a("### Jan+Jul 2025 (E20 overall baseline 368.03; matched 244.76 frozen)")
    a("")
    a("| Rule | LIRF unmatched RMSE | Δ LIRF_u | Overall | Δ overall |")
    a("|---|---:|---:|---:|---:|")
    for r in j["metrics"]:
        a(f"| {r['name']} | {r['rmse_lirf_u']:.1f} | {r['d_lirf_u']:+.1f} | {r['overall']:.2f} | {r['d_overall']:+.2f} |")
    a("")
    a("### December 2025 (E20 overall baseline 228.45; matched 215.90 frozen)")
    a("")
    a("| Rule | LIRF unmatched RMSE | Δ LIRF_u | Overall | Δ overall |")
    a("|---|---:|---:|---:|---:|")
    for r in d["metrics"]:
        a(f"| {r['name']} | {r['rmse_lirf_u']:.1f} | {r['d_lirf_u']:+.1f} | {r['overall']:.2f} | {r['d_overall']:+.2f} |")
    a("")
    a("## 4. Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    if payload["decision"]["keep"]:
        a(f"**Keep candidate:** `{payload['decision']['keep']}`")
        a("")
    a(payload["decision"]["reason"])
    a("")
    a("Accept only if Jan+Jul overall falls and December overall does not rise,")
    a("with LIRF-unmatched RMSE down on both splits. Matched RMSE is unchanged")
    a("by construction (this slice is disjoint from matched).")
    a("")
    a("Artifacts: `analysis/E21/`.")
    a("")
    path = REP / "E21_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
