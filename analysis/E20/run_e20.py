"""E20: LIRF unmatched history prior (prefix / ADES / flight number).

E13 rejected prefix/stand/dest as *hard exclusive triggers* for the extreme
half (low recall; OR with mvt_sched>1800 = always-on). This experiment uses
the same ranking-safe keys as a *train-only rate*:

  p = P(y>30 min | key) on train LIRF unmatched, shrunk to the global rate
  ŷ = (1-p)·geo_mean + p·(MVT−SCHED)

and as a “send the identifiable *normal* prefixes to geo” rule (the reverse
of E13’s extreme-only triggers).

LIRF unmatched currently always MVT−SCHED (E18-H). Matched path untouched.
Overall RMSE via SSE substitution (no LightGBM). Training files only.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, load_dep, rmse, save_result, split_by_months  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402

OUT = ROOT / "analysis" / "E20"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

TAIL = 1800.0
E18H = {
    "janjul": {"overall": 372.3646118481625, "n": 344419, "lirf_u": 6032.69683773519},
    "dec": {"overall": 238.01080378154546, "n": 165677, "lirf_u": 2786.1434163308313},
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("FLIGHT_mvt").fill_null("NA").str.slice(0, 3).fill_null("NA").alias("flt_prefix"),
        pl.col("STAND_mvt").fill_null("NA").str.slice(0, 1).fill_null("NA").alias("stand_p"),
        pl.col("ADES_mvt").fill_null("NA").alias("ades"),
        pl.col("FLIGHT_mvt").fill_null("NA").alias("flt"),
        (pl.col("y") > TAIL).cast(pl.Int8).alias("ext"),
    )


def rates(train_lu: pl.DataFrame, key: str, k: float, p0: float) -> pl.DataFrame:
    g = train_lu.group_by(key).agg(
        pl.len().alias("n"),
        pl.col("ext").mean().alias("p_raw"),
        pl.col("y").median().alias("med_y"),
    )
    return g.with_columns(
        ((pl.col("n") * pl.col("p_raw") + k * p0) / (pl.col("n") + k)).alias("p_hat")
    )


def map_p(val: pl.DataFrame, tbl: pl.DataFrame, key: str, p0: float) -> np.ndarray:
    j = val.join(tbl.select(key, "p_hat"), on=key, how="left")
    p = j["p_hat"].to_numpy().astype(np.float64)
    return np.where(np.isfinite(p), p, p0)


def mix(geo, ms, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return (1.0 - p) * np.asarray(geo, dtype=np.float64) + p * np.asarray(ms, dtype=np.float64)


def overall_sub(base_overall, base_n, y, old, new, mask) -> float:
    sse = (base_overall ** 2) * base_n
    y = np.asarray(y, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    m = mask & np.isfinite(y) & np.isfinite(old) & np.isfinite(new)
    sse = sse - np.sum((y[m] - old[m]) ** 2) + np.sum((y[m] - new[m]) ** 2)
    return float(np.sqrt(max(sse, 0.0) / base_n))


def lu_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))


def eval_pred(name, y, pred, ms, geo, lu, base_overall, base_n) -> dict:
    r_lu = rmse(y[lu], pred[lu])
    r_n = rmse(y[lu & (y <= TAIL)], pred[lu & (y <= TAIL)])
    r_x = rmse(y[lu & (y > TAIL)], pred[lu & (y > TAIL)])
    ov = overall_sub(base_overall, base_n, y, ms, pred, lu)
    return {
        "name": name,
        "rmse_lirf_u": r_lu,
        "rmse_normal": r_n,
        "rmse_extreme": r_x,
        "overall": ov,
        "d_overall": ov - base_overall,
        "d_lirf_u": r_lu - rmse(y[lu], ms[lu]),
    }


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


def run_split(dep: pl.DataFrame, months: list[int], split_name: str) -> dict:
    base = E18H[split_name]
    tr0, va0 = split_by_months(dep, months)
    tabs = geometry_tables(tr0.filter(~pl.col("unmatched")))
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)
    tr_lu = lu_frame(tr)
    va_lu_n = lu_frame(va).height
    p0 = float(tr_lu["ext"].mean()) if tr_lu.height else 0.5
    log(f"  {split_name} train LIRF_u n={tr_lu.height} p0={p0:.3f}  val LIRF_u n={va_lu_n}")

    y = va["y"].to_numpy().astype(np.float64)
    um = va["unmatched"].to_numpy().astype(bool)
    ap = va["airport"].to_numpy()
    lu = um & (ap == "LIRF")
    ms = va["mvt_sched"].to_numpy().astype(np.float64)
    geo = va["geo_mean"].to_numpy().astype(np.float64)

    # train tables
    pref = rates(tr_lu, "flt_prefix", k=10.0, p0=p0)
    ades = rates(tr_lu, "ades", k=10.0, p0=p0)
    st = rates(tr_lu, "stand_p", k=10.0, p0=p0)
    flt = rates(tr_lu, "flt", k=2.0, p0=p0)
    if split_name == "janjul":
        pref.sort("n", descending=True).head(20).write_csv(TAB / "e20_prefix_rates_train_janjul.csv")
        ades.sort("n", descending=True).head(20).write_csv(TAB / "e20_ades_rates_train_janjul.csv")

    p_pref = map_p(va, pref, "flt_prefix", p0)
    p_ades = map_p(va, ades, "ades", p0)
    p_st = map_p(va, st, "stand_p", p0)
    p_flt = map_p(va, flt, "flt", p0)
    p_mean = np.nanmean(np.vstack([p_pref, p_ades, p_st]), axis=0)
    p_max = np.nanmax(np.vstack([p_pref, p_ades]), axis=0)

    always = ms.copy()
    oracle = np.where(lu & (y <= TAIL) & np.isfinite(geo), geo, always)
    oracle = np.where(lu & (y > TAIL), ms, oracle)

    preds = {
        "always_sched": always,
        "mix_prefix": mix(geo, ms, p_pref),
        "mix_ades": mix(geo, ms, p_ades),
        "mix_stand": mix(geo, ms, p_st),
        "mix_flight": mix(geo, ms, p_flt),
        "mix_mean3": mix(geo, ms, p_mean),
        "mix_max": mix(geo, ms, p_max),
        "oracle": oracle,
    }

    # train-chosen T: if p_prefix < T → geo else sched  (identify NORMAL)
    # and if p_prefix > T → sched else geo (identify EXTREME, E13-style)
    tr_y = tr_lu["y"].to_numpy().astype(np.float64)
    tr_ms = tr_lu["mvt_sched"].to_numpy().astype(np.float64)
    tr_geo = tr_lu["geo_mean"].to_numpy().astype(np.float64)
    tr_p = map_p(tr_lu, pref, "flt_prefix", p0)
    best_lo, best_hi = None, None
    best_lo_r, best_hi_r = 1e18, 1e18
    always_tr = rmse(tr_y, tr_ms)
    for T in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        lo = np.where(tr_p < T, tr_geo, tr_ms)  # low p → geo (normal)
        hi = np.where(tr_p > T, tr_ms, tr_geo)  # high p → sched (extreme); else geo
        r_lo, r_hi = rmse(tr_y, lo), rmse(tr_y, hi)
        if r_lo < best_lo_r:
            best_lo_r, best_lo = r_lo, T
        if r_hi < best_hi_r:
            best_hi_r, best_hi = r_hi, T
    log(f"    train always_sched={always_tr:.0f}  best low-p→geo T={best_lo} RMSE={best_lo_r:.0f}  "
        f"best high-p→sched T={best_hi} RMSE={best_hi_r:.0f}")

    p_pref_va = p_pref
    preds["gate_normal"] = np.where(lu & (p_pref_va < best_lo), geo, always)
    preds["gate_extreme"] = np.where(lu, np.where(p_pref_va > best_hi, ms, geo), always)

    rows = []
    for name, pred in preds.items():
        e = eval_pred(name, y, pred, ms, geo, lu, base["overall"], base["n"])
        rows.append(e)
        log(f"    {name:16s} overall={e['overall']:.2f} ({e['d_overall']:+.2f})  "
            f"LIRF_u={e['rmse_lirf_u']:.0f}  norm={e['rmse_normal']:.0f}  ext={e['rmse_extreme']:.0f}")
    return {
        "p0": p0,
        "n_train_lu": tr_lu.height,
        "n_val_lu": int(lu.sum()),
        "best_lo_T": best_lo,
        "best_hi_T": best_hi,
        "metrics": rows,
    }


def main():
    log("E20: LIRF history prior (training only)...")
    dep = add_keys(load_dep())
    payload = {
        "janjul": run_split(dep, [1, 7], "janjul"),
        "dec": run_split(dep, [12], "dec"),
    }

    def get(blob, name, field):
        for r in blob["metrics"]:
            if r["name"] == name:
                return r[field]
        return float("nan")

    j, d = payload["janjul"], payload["dec"]
    # pick best deployable (not oracle) by Jan+Jul overall, require Dec not worse than +0.5
    deploy = [r["name"] for r in j["metrics"] if r["name"] not in ("oracle",)]
    ranked = sorted(deploy, key=lambda n: get(j, n, "overall"))
    keep = None
    for n in ranked:
        if n == "always_sched":
            continue
        if get(j, n, "d_overall") < -0.5 and get(d, n, "d_overall") <= 0.5:
            keep = n
            break
    if keep:
        verdict = "ACCEPTED" if get(j, keep, "d_overall") < -2 else "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
        keep = None
    reason = (
        f"Jan+Jul always={get(j,'always_sched','overall'):.2f} best-keep={keep} "
        f"{get(j, keep, 'overall') if keep else float('nan'):.2f} "
        f"({get(j, keep, 'd_overall') if keep else 0:+.2f}); "
        f"Dec {get(d, keep, 'd_overall') if keep else 0:+.2f}. "
        f"Oracle {get(j,'oracle','overall'):.2f} / {get(d,'oracle','overall'):.2f}."
    )
    payload["decision"] = {"verdict": verdict, "keep": keep, "reason": reason}
    log(f"VERDICT {verdict}")
    log(reason)
    write_report(payload)
    save_result("E20", payload)
    (TAB / "e20_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e20.py")
    log("done")


def write_report(payload):
    j, d = payload["janjul"], payload["dec"]
    lines = []
    a = lines.append
    a("# E20 — LIRF unmatched history prior")
    a("")
    a("Training files only. Script: `experiments/run_e20_lirf_history_prior.py`.")
    a("")
    a("E13 rejected prefix/dest/stand as exclusive extreme *triggers*. This uses")
    a("train-only P(y>30 | prefix or ADES) as a mixture weight, and as a")
    a("low-p → geo (identify the *normal* half) rule.")
    a("")
    a("## Jan+Jul 2025")
    a("")
    a("| Rule | Overall | Δ | LIRF unmatched | Normal half | Extreme half |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in j["metrics"]:
        a(f"| {r['name']} | {r['overall']:.2f} | {r['d_overall']:+.2f} | {r['rmse_lirf_u']:.0f} | {r['rmse_normal']:.0f} | {r['rmse_extreme']:.0f} |")
    a("")
    a("## December 2025")
    a("")
    a("| Rule | Overall | Δ | LIRF unmatched | Normal half | Extreme half |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in d["metrics"]:
        a(f"| {r['name']} | {r['overall']:.2f} | {r['d_overall']:+.2f} | {r['rmse_lirf_u']:.0f} | {r['rmse_normal']:.0f} | {r['rmse_extreme']:.0f} |")
    a("")
    a(f"Train-chosen T (Jan+Jul): low-p→geo T={j['best_lo_T']}, high-p→sched T={j['best_hi_T']}.")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    path = REP / "E20_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
