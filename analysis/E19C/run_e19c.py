"""E19-C: LIRF unmatched gate-vs-taxi using matched neighbours.

E13: two TAXITIME regimes on LIRF unmatched; MVT−SCHED is large in both;
hard gates failed. The missing piece was “was the late takeoff a gate hold
or a long taxi?” — AOBT/BLOCK, which unmatched rows do not have.

Matched neighbours at the same airport do have AOBT by the scored MVT.
neigh_taxi_frac = mean (MVT−AOBT)/(MVT−SCHED) of other pushes with
AOBT in (T−30m, T]. If neighbours are holding at the gate, unmatched
LIRF with large MVT−SCHED may still have short taxi → geo_mean.
If neighbours are sitting on the taxiway → keep MVT−SCHED.

Phase 2: does neighbour mix identify y>30 min on LIRF unmatched?
Phase 3: only if yes — train-only threshold/logistic, splice onto E18-H
(LIRF unmatched currently = MVT−SCHED; matched path untouched).

Training files only. No LightGBM. No ranking/submitting.
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

from common import (  # noqa: E402
    AIRPORTS,
    ROOT,
    load_dep,
    mae,
    rmse,
    save_result,
    split_by_months,
)
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402

OUT = ROOT / "analysis" / "E19C"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

W_NS = 30 * 60 * 1_000_000_000
TAIL = 1800.0
E18H_JANJUL_OVERALL = 372.3646118481625
E18H_JANJUL_N = 344419
E18H_DEC_OVERALL = 238.01080378154546
E18H_DEC_N = 165677


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ns_array(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype("datetime64[ns]").astype(np.int64)


def add_neighbor_mix(dep: pl.DataFrame, window_ns: int = W_NS) -> pl.DataFrame:
    """Causal neighbour push/taxi mix. Neighbours contribute iff AOBT ≤ scored MVT."""
    n = dep.height
    taxi_mean = np.full(n, np.nan)
    push_mean = np.full(n, np.nan)
    taxi_frac = np.full(n, np.nan)
    n_nb = np.zeros(n, dtype=np.int32)
    ap = dep["airport"].to_numpy()
    mvt = ns_array(dep["MVT_TIME_UTC_mvt"])
    aobt = ns_array(dep["AOBT_3_flt"])
    mvt_aobt = dep["mvt_aobt"].to_numpy().astype(np.float64)
    aobt_sched = dep["aobt_sched"].to_numpy().astype(np.float64)
    mvt_sched = dep["mvt_sched"].to_numpy().astype(np.float64)
    unmatched = dep["unmatched"].to_numpy().astype(bool)

    for airport in AIRPORTS:
        idx = np.where(ap == airport)[0]
        if idx.size == 0:
            continue
        # neighbour pool: matched with finite AOBT and mvt_aobt
        ok = (~unmatched[idx]) & np.isfinite(mvt_aobt[idx]) & np.isfinite(aobt[idx].astype(np.float64))
        pool = idx[ok]
        if pool.size < 5:
            continue
        order = np.argsort(aobt[pool])
        pool = pool[order]
        pt = aobt[pool]
        taxi = mvt_aobt[pool]
        push = aobt_sched[pool]
        ms = mvt_sched[pool]
        tf = np.where(np.isfinite(ms) & (ms > 300) & np.isfinite(taxi), taxi / ms, np.nan)
        tz = np.where(np.isfinite(taxi), taxi, 0.0)
        pz = np.where(np.isfinite(push), push, 0.0)
        fz = np.where(np.isfinite(tf), tf, 0.0)
        fn = np.isfinite(taxi).astype(np.int64)
        pn = np.isfinite(push).astype(np.int64)
        tn = np.isfinite(tf).astype(np.int64)
        cs_t = np.concatenate([[0.0], np.cumsum(tz)])
        cs_p = np.concatenate([[0.0], np.cumsum(pz)])
        cs_f = np.concatenate([[0.0], np.cumsum(fz)])
        cn_t = np.concatenate([[0], np.cumsum(fn)])
        cn_p = np.concatenate([[0], np.cumsum(pn)])
        cn_f = np.concatenate([[0], np.cumsum(tn)])
        qt = mvt[idx]
        hi = np.searchsorted(pt, qt, side="right")
        lo = np.searchsorted(pt, qt - window_ns, side="right")
        nt = (cn_t[hi] - cn_t[lo]).astype(np.float64)
        np_ = (cn_p[hi] - cn_p[lo]).astype(np.float64)
        nf = (cn_f[hi] - cn_f[lo]).astype(np.float64)
        taxi_mean[idx] = np.where(nt > 0, (cs_t[hi] - cs_t[lo]) / nt, np.nan)
        push_mean[idx] = np.where(np_ > 0, (cs_p[hi] - cs_p[lo]) / np_, np.nan)
        taxi_frac[idx] = np.where(nf > 0, (cs_f[hi] - cs_f[lo]) / nf, np.nan)
        n_nb[idx] = nt.astype(np.int32)
    return dep.with_columns(
        pl.Series("neigh_taxi_mean_30m", taxi_mean),
        pl.Series("neigh_push_mean_30m", push_mean),
        pl.Series("neigh_taxi_frac_30m", taxi_frac),
        pl.Series("neigh_n_30m", n_nb),
    )


def qcut8(x):
    s = pd.Series(x)
    try:
        return pd.qcut(s, 8, labels=[f"Q{i}" for i in range(1, 9)], duplicates="drop").astype(str).to_numpy()
    except ValueError:
        return pd.qcut(s.rank(method="first"), 8, labels=[f"Q{i}" for i in range(1, 9)]).astype(str).to_numpy()


def lirf_unmatched_diag(va: pl.DataFrame) -> dict:
    lu = va.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))
    y = lu["y"].to_numpy().astype(np.float64)
    ms = lu["mvt_sched"].to_numpy().astype(np.float64)
    geo = lu["geo_mean"].to_numpy().astype(np.float64)
    frac = lu["neigh_taxi_frac_30m"].to_numpy().astype(np.float64)
    push = lu["neigh_push_mean_30m"].to_numpy().astype(np.float64)
    taxi = lu["neigh_taxi_mean_30m"].to_numpy().astype(np.float64)
    ext = y > TAIL
    out = {
        "n": int(lu.height),
        "n_ext": int(ext.sum()),
        "n_norm": int((~ext).sum()),
        "corr_frac_y": _corr(frac, y),
        "corr_frac_ext": _corr(frac, ext.astype(np.float64)),
        "corr_push_y": _corr(push, y),
        "corr_taxi_y": _corr(taxi, y),
        "corr_frac_resid_sched": _corr(frac, y - ms),
        "rmse_always_sched": rmse(y, ms),
        "rmse_always_geo": rmse(y, geo),
    }
    # Q1 vs Q8 of taxi_frac among finite
    m = np.isfinite(frac)
    rows = []
    if m.sum() >= 40:
        lab = qcut8(frac[m])
        for q in [f"Q{i}" for i in range(1, 9)]:
            sel = lab == q
            if not sel.any():
                continue
            yy = y[m][sel]
            rows.append(
                {
                    "q": q,
                    "n": int(sel.sum()),
                    "mean_y": float(yy.mean()),
                    "med_y": float(np.median(yy)),
                    "frac_gt30": float((yy > TAIL).mean()),
                    "mean_frac": float(np.nanmean(frac[m][sel])),
                    "rmse_sched": rmse(yy, ms[m][sel]),
                    "rmse_geo": rmse(yy, geo[m][sel]),
                }
            )
    out["by_frac_q"] = rows
    if rows:
        q1 = next((r for r in rows if r["q"] == "Q1"), None)
        q8 = next((r for r in rows if r["q"] == "Q8"), None)
        if q1 and q8 and q1["frac_gt30"] > 0:
            out["q8_q1_gt30_ratio"] = q8["frac_gt30"] / q1["frac_gt30"]
        else:
            out["q8_q1_gt30_ratio"] = float("nan")
    else:
        out["q8_q1_gt30_ratio"] = float("nan")
    return out


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def train_threshold(tr: pl.DataFrame, col: str) -> dict:
    """Train-only: among LIRF unmatched, pick threshold on col to min LIRF-u RMSE
    of (geo if col<=T else mvt_sched). Also try the reverse."""
    lu = tr.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))
    y = lu["y"].to_numpy().astype(np.float64)
    ms = lu["mvt_sched"].to_numpy().astype(np.float64)
    geo = lu["geo_mean"].to_numpy().astype(np.float64)
    x = lu[col].to_numpy().astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ms) & np.isfinite(geo)
    y, ms, geo, x = y[m], ms[m], geo[m], x[m]
    always = rmse(y, ms)
    best = {"col": col, "T": None, "dir": "gt", "train_rmse": always, "always_sched": always}
    if x.size < 30:
        return best
    for T in np.quantile(x, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        for dir_, use_sched in (("gt", x > T), ("lt", x < T)):
            pred = np.where(use_sched, ms, geo)
            r = rmse(y, pred)
            if r < best["train_rmse"] - 1:
                best = {"col": col, "T": float(T), "dir": dir_, "train_rmse": r, "always_sched": always,
                        "n_sched": int(use_sched.sum()), "n_geo": int((~use_sched).sum())}
    return best


def apply_rule(va: pl.DataFrame, rule: dict) -> np.ndarray:
    lu_mask = (va["airport"].to_numpy() == "LIRF") & va["unmatched"].to_numpy().astype(bool)
    ms = va["mvt_sched"].to_numpy().astype(np.float64)
    geo = va["geo_mean"].to_numpy().astype(np.float64)
    x = va[rule["col"]].to_numpy().astype(np.float64)
    pred = ms.copy()
    finite = np.isfinite(x) & np.isfinite(geo)
    if rule["T"] is None:
        return pred
    if rule["dir"] == "gt":
        use_sched = finite & lu_mask & (x > rule["T"])
        use_geo = finite & lu_mask & (x <= rule["T"])
    else:
        use_sched = finite & lu_mask & (x < rule["T"])
        use_geo = finite & lu_mask & (x >= rule["T"])
    pred[use_geo] = geo[use_geo]
    # use_sched already ms
    return pred, use_geo, use_sched


def overall_from_lirf_u(base_overall, base_n, y, pred_old, pred_new, lu_mask) -> float:
    """Replace LIRF unmatched predictions; rest of SSE unchanged."""
    sse = (base_overall ** 2) * base_n
    y = np.asarray(y, dtype=np.float64)
    old = np.asarray(pred_old, dtype=np.float64)
    new = np.asarray(pred_new, dtype=np.float64)
    m = lu_mask & np.isfinite(y) & np.isfinite(old) & np.isfinite(new)
    sse = sse - np.sum((y[m] - old[m]) ** 2) + np.sum((y[m] - new[m]) ** 2)
    return float(np.sqrt(sse / base_n))


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


def main():
    log("E19-C: load training DEP, geometry, neighbour mix...")
    dep = load_dep().with_columns(
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"),
        pl.col("FLIGHT_mvt").fill_null("NA").str.slice(0, 3).fill_null("NA").alias("flt_prefix"),
    )
    dep = add_neighbor_mix(dep)
    log(f"  neigh_taxi_frac finite {int(np.isfinite(dep['neigh_taxi_frac_30m'].to_numpy()).sum()):,}/{dep.height:,}")

    payload = {}
    for split_name, months, base_rmse, base_n in [
        ("janjul", [1, 7], E18H_JANJUL_OVERALL, E18H_JANJUL_N),
        ("dec", [12], E18H_DEC_OVERALL, E18H_DEC_N),
    ]:
        log(f"===== {split_name} =====")
        tr0, va0 = split_by_months(dep, months)
        tabs = geometry_tables(tr0.filter(~pl.col("unmatched")))
        tr = attach_geometry(tr0, tabs, 30)
        va = attach_geometry(va0, tabs, 30)
        diag_tr = lirf_unmatched_diag(tr)
        diag_va = lirf_unmatched_diag(va)
        log(f"  train LIRF_u n={diag_tr['n']} ext={diag_tr['n_ext']} corr(frac,y)={diag_tr['corr_frac_y']:.3f} "
            f"corr(frac,>30)={diag_tr['corr_frac_ext']:.3f} Q8/Q1 gt30={diag_tr['q8_q1_gt30_ratio']}")
        log(f"  val   LIRF_u n={diag_va['n']} ext={diag_va['n_ext']} corr(frac,y)={diag_va['corr_frac_y']:.3f} "
            f"corr(frac,>30)={diag_va['corr_frac_ext']:.3f} Q8/Q1 gt30={diag_va['q8_q1_gt30_ratio']}")
        pd.DataFrame(diag_va["by_frac_q"]).to_csv(TAB / f"e19c_val_frac_q_{split_name}.csv", index=False)

        rules = [train_threshold(tr, c) for c in ("neigh_taxi_frac_30m", "neigh_push_mean_30m", "neigh_taxi_mean_30m")]
        for r in rules:
            log(f"  train-best {r['col']} {r['dir']} T={r['T']} RMSE {r['always_sched']:.0f}→{r['train_rmse']:.0f}")

        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy().astype(bool)
        ap = va["airport"].to_numpy()
        lu = um & (ap == "LIRF")
        ms = va["mvt_sched"].to_numpy().astype(np.float64)
        geo = va["geo_mean"].to_numpy().astype(np.float64)
        always = ms.copy()
        # oracle: y<=30 → geo, else sched
        oracle = np.where(lu & (y <= TAIL) & np.isfinite(geo), geo, always)
        oracle = np.where(lu & (y > TAIL), ms, oracle)

        split_block = {
            "diag_train": diag_tr,
            "diag_val": diag_va,
            "rules": rules,
            "val_always_sched_lirf_u": rmse(y[lu], ms[lu]),
            "val_always_geo_lirf_u": rmse(y[lu], geo[lu]),
            "val_oracle_lirf_u": rmse(y[lu], oracle[lu]),
            "overall_always": base_rmse,
            "overall_oracle": overall_from_lirf_u(base_rmse, base_n, y, always, oracle, lu),
        }
        # apply each train-best rule
        applied = []
        for r in rules:
            if r["T"] is None:
                continue
            pred, use_geo, use_sched = apply_rule(va, r)
            lirf_u_rmse = rmse(y[lu], pred[lu])
            ov = overall_from_lirf_u(base_rmse, base_n, y, always, pred, lu)
            applied.append(
                {
                    **r,
                    "val_lirf_u_rmse": lirf_u_rmse,
                    "val_overall": ov,
                    "n_geo_val": int(use_geo.sum()),
                    "n_sched_val": int((use_sched & lu).sum()),
                }
            )
            log(f"    apply {r['col']} {r['dir']}: LIRF_u RMSE {split_block['val_always_sched_lirf_u']:.0f}→{lirf_u_rmse:.0f}  "
                f"overall {base_rmse:.2f}→{ov:.2f}  geo_n={int(use_geo.sum())}")
        split_block["applied"] = applied
        payload[split_name] = split_block

    # decision: need val overall down on Jan+Jul and not up on Dec, and Phase 2 signal
    j = payload["janjul"]
    d = payload["dec"]
    signal = (
        (np.isfinite(j["diag_val"]["corr_frac_ext"]) and abs(j["diag_val"]["corr_frac_ext"]) >= 0.15)
        or (np.isfinite(j["diag_val"]["q8_q1_gt30_ratio"]) and (
            j["diag_val"]["q8_q1_gt30_ratio"] >= 1.5 or j["diag_val"]["q8_q1_gt30_ratio"] <= 1 / 1.5
        ))
    )
    best_j = min(j["applied"], key=lambda r: r["val_overall"]) if j["applied"] else None
    best_d = None
    if best_j:
        # same rule family on dec: find matching col
        cands = [a for a in d["applied"] if a["col"] == best_j["col"] and a["dir"] == best_j["dir"]]
        best_d = cands[0] if cands else None
    if signal and best_j and best_d and best_j["val_overall"] < j["overall_always"] - 1 and best_d["val_overall"] <= d["overall_always"] + 0.5:
        verdict = "ACCEPTED"
    elif (best_j and best_j["val_overall"] < j["overall_always"] - 1) and not (best_d and best_d["val_overall"] <= d["overall_always"] + 0.5):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
    reason = (
        f"Phase2 val corr(taxi_frac, y>30)={j['diag_val']['corr_frac_ext']:.3f} "
        f"Q8/Q1={j['diag_val']['q8_q1_gt30_ratio']}. "
        f"Oracle LIRF mix overall {j['overall_always']:.2f}→{j['overall_oracle']:.2f} (Jan+Jul), "
        f"{d['overall_always']:.2f}→{d['overall_oracle']:.2f} (Dec). "
    )
    if best_j:
        reason += (
            f"Best deployable {best_j['col']} {best_j['dir']}: Jan+Jul {best_j['val_overall']:.2f} "
            f"(LIRF_u {best_j['val_lirf_u_rmse']:.0f}); "
        )
        if best_d:
            reason += f"Dec {best_d['val_overall']:.2f}."
    payload["decision"] = {"verdict": verdict, "reason": reason, "phase2_signal": signal, "best_janjul": best_j, "best_dec": best_d}
    log(f"VERDICT {verdict}")
    log(reason)

    write_report(payload)
    save_result("E19C", payload)
    (TAB / "e19c_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e19c.py")
    log("done")


def write_report(payload):
    j, d = payload["janjul"], payload["dec"]
    lines = []
    a = lines.append
    a("# E19-C — LIRF unmatched neighbour gate-vs-taxi")
    a("")
    a("Training files only. No LightGBM. Script: `experiments/run_e19c_lirf_neighbor_split.py`.")
    a("")
    a("Matched neighbours with AOBT already known at scored MVT contribute")
    a("`neigh_taxi_frac_30m` = mean `(MVT−AOBT)/(MVT−SCHED)` over the previous 30 min.")
    a("Hypothesis: high neighbour taxi fraction ⇒ unmatched LIRF taxi really is")
    a("takeoff−schedule; low ⇒ gate hold, use `geo_mean`.")
    a("")
    a("## Phase 2 (LIRF unmatched only)")
    a("")
    a("| Split | n | n y>30 | corr(frac, y) | corr(frac, y>30) | Q8/Q1 P(y>30) |")
    a("|---|---:|---:|---:|---:|---:|")
    for name, b in ("Jan+Jul val", j), ("Jan+Jul train", payload["janjul"]), ("Dec val", d):
        pass
    a(
        f"| Jan+Jul val | {j['diag_val']['n']} | {j['diag_val']['n_ext']} | "
        f"{j['diag_val']['corr_frac_y']:.3f} | {j['diag_val']['corr_frac_ext']:.3f} | "
        f"{j['diag_val']['q8_q1_gt30_ratio']} |"
    )
    a(
        f"| Dec val | {d['diag_val']['n']} | {d['diag_val']['n_ext']} | "
        f"{d['diag_val']['corr_frac_y']:.3f} | {d['diag_val']['corr_frac_ext']:.3f} | "
        f"{d['diag_val']['q8_q1_gt30_ratio']} |"
    )
    a("")
    a("Always MVT−SCHED vs always geo vs oracle (y≤30→geo else SCHED) on LIRF unmatched:")
    a("")
    a("| Split | Always SCHED | Always geo | Oracle mix | Overall if oracle |")
    a("|---|---:|---:|---:|---:|")
    a(f"| Jan+Jul | {j['val_always_sched_lirf_u']:.0f} | {j['val_always_geo_lirf_u']:.0f} | {j['val_oracle_lirf_u']:.0f} | {j['overall_oracle']:.2f} |")
    a(f"| December | {d['val_always_sched_lirf_u']:.0f} | {d['val_always_geo_lirf_u']:.0f} | {d['val_oracle_lirf_u']:.0f} | {d['overall_oracle']:.2f} |")
    a("")
    a("## Train-threshold rules applied to val")
    a("")
    a("| Split | Feature | Dir | T | LIRF_u RMSE | Overall | n→geo |")
    a("|---|---|---|---:|---:|---:|---:|")
    for split, b in ("janjul", j), ("dec", d):
        for r in b["applied"]:
            a(f"| {split} | {r['col']} | {r['dir']} | {r['T']:.3f} | {r['val_lirf_u_rmse']:.0f} | {r['val_overall']:.2f} | {r['n_geo_val']} |")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    path = REP / "E19C_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
