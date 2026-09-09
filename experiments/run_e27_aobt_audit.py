"""E27 — AOBT off-block anchor audit.

Same-flight operational off-block timestamps (NM table) vs the hidden movement
BLOCK_TIME. Answer: does `MVT - AOBT_3_flt` reconstruct TAXITIME closely
enough to become the primary architecture?

Phases:
  1. Exact training audit of delta = BLOCK_TIME - AOBT_3_flt (by airport)
  2. Proxy comparison AOBT / LOBT / IOBT / EOBT / SCHED on temporal validation
  3. Small AOBT-delta calibration model (per-airport + LightGBM)
  4. Off-block source disagreement vs E20 residual / tail
  5. Hybrid: calibrated AOBT proxy on matched + E20 on unmatched
  6. Leakage audit (AOBT < MVT)
  7. Temporal validation (Jan+Jul / December; 2026 ranking never used)

Ranking coverage was measured separately: 98.47% of ranking DEP have AOBT.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import AIRPORTS, load_dep, mae, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E27"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1

NUM_CAL = ["mvt_aobt", "aobt_eobt", "aobt_iobt", "aobt_lobt", "hour", "dow", "month"]
CAT_CAL = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def add_proxies(dep: pl.DataFrame) -> pl.DataFrame:
    return dep.with_columns([
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("proxy_aobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("proxy_lobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("proxy_iobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("proxy_eobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("delta_block_aobt"),
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("d_aobt_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("d_aobt_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("d_aobt_eobt"),
        (pl.col("LOBT_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("d_lobt_iobt"),
        (pl.col("LOBT_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("d_lobt_eobt"),
        (pl.col("AOBT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("leak_aobt_mvt"),
    ])


def stats_err(d):
    d = np.asarray(d, dtype=np.float64)
    m = np.isfinite(d)
    ad = np.abs(d[m])
    return {
        "n": int(m.sum()),
        "exact_rate": float(np.mean(d[m] == 0)) if m.any() else float("nan"),
        "median_delta": float(np.median(d[m])),
        "mae": float(mae(d[m], np.zeros_like(d[m]))),
        "rmse": float(rmse(d[m], np.zeros_like(d[m]))),
        "p50_abs": float(np.median(ad)),
        "p90_abs": float(np.quantile(ad, 0.90)),
        "p95_abs": float(np.quantile(ad, 0.95)),
        "p99_abs": float(np.quantile(ad, 0.99)),
        "max_abs": float(ad.max()) if ad.size else float("nan"),
    }


def score_t(y, p, um, ap):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~np.asarray(um, dtype=bool)
    out = {"rmse": rmse(y[ok], p[ok]), "matched": rmse(y[matched], p[matched]) if matched.any() else float("nan")}
    for t, k in [(1800, "gt30m"), (2700, "gt45m"), (3600, "gt60m")]:
        s = matched & (y > t)
        out[k] = rmse(y[s], p[s]) if s.any() else float("nan")
    return out


def main():
    log("E27: load training_*.parquet only...")
    dep = add_proxies(load_dep().with_columns(
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null")
    ))
    log(f"rows {dep.height:,}")

    # ---------------- PHASE 1: exact training audit ----------------
    ph1 = {}
    matched_all = dep.filter(~pl.col("unmatched"))
    ph1["all"] = stats_err(matched_all["delta_block_aobt"].to_numpy())
    ph1["airports"] = {a: stats_err(matched_all.filter(pl.col("airport") == a)["delta_block_aobt"].to_numpy()) for a in AIRPORTS}
    log("Phase1 delta=BLOCK-AOBT (matched, all training): " +
        f"exact {ph1['all']['exact_rate']*100:.2f}%  med {ph1['all']['median_delta']:.0f}s  "
        f"MAE {ph1['all']['mae']:.0f}s  RMSE {ph1['all']['rmse']:.0f}s  p90abs {ph1['all']['p90_abs']:.0f}s")

    # ---------------- PHASE 6: leakage audit ----------------
    leak = dep.filter(~pl.col("unmatched"))["leak_aobt_mvt"].to_numpy()
    ph6 = {"violations": int((leak > 0).sum()), "n_checked": int(np.isfinite(leak).sum()),
           "min_sec": float(np.nanmin(leak)), "p1_sec": float(np.nanquantile(leak, 0.01))}
    log(f"Phase6 leak AOBT<MVT: violations {ph6['violations']} / {ph6['n_checked']:,}")

    # ---------------- temporal validation ----------------
    payload = {"phase1": ph1, "phase6": ph6}
    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
        log(f"========== {split_name} ==========")
        tr0 = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy().astype(bool)
        ap = va["airport"].to_numpy()
        mvt_sched = va["mvt_sched"].to_numpy().astype(np.float64)
        lirf_mask = um & (ap == "LIRF") & np.isfinite(mvt_sched)

        # cached E20
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split_name}.parquet")
        oof = oof.join(va.select(["MVT_ID_mvt", "proxy_aobt", "proxy_lobt", "proxy_iobt", "proxy_eobt",
                                  "delta_block_aobt", "d_aobt_lobt", "d_aobt_iobt", "d_aobt_eobt",
                                  "d_lobt_iobt", "d_lobt_eobt"]), on="MVT_ID_mvt", how="left")
        yo = oof["y"].to_numpy()
        umo = oof["unmatched"].to_numpy().astype(bool)
        apo = oof["airport"].to_numpy()
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()

        scores = {"E20": score_t(yo, e20, umo, apo)}

        # ---------------- PHASE 2: raw proxies ----------------
        for name, col in [("AOBT", "proxy_aobt"), ("LOBT", "proxy_lobt"),
                          ("IOBT", "proxy_iobt"), ("EOBT", "proxy_eobt"), ("SCHED", "mvt_sched")]:
            p = oof[col].to_numpy().astype(np.float64).copy()
            p[lirf_mask] = mvt_sched[lirf_mask]
            scores[name] = score_t(yo, p, umo, apo)
            log(f"  proxy {name}: overall={scores[name]['rmse']:.2f} matched={scores[name]['matched']:.2f} "
                f"gt30={scores[name]['gt30m']:.1f} coverage={100*np.mean(np.isfinite(p)):.2f}%")

        # ---------------- PHASE 3: delta calibration ----------------
        log("  calibrating delta = BLOCK - AOBT (per-airport + small LGB)...")
        tr_m = tr0.filter(~pl.col("unmatched"))
        va_m = va.filter(~pl.col("unmatched"))
        cal_t = tr_m.select(NUM_CAL + CAT_CAL + ["delta_block_aobt"]).to_pandas()
        cal_v = va_m.select(NUM_CAL + CAT_CAL + ["delta_block_aobt"]).to_pandas()
        for c in CAT_CAL:
            cal_t[c] = cal_t[c].astype("category")
            cal_v[c] = cal_v[c].astype("category")
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=80,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(cal_t[NUM_CAL + CAT_CAL], cal_t["delta_block_aobt"].to_numpy(),
                  categorical_feature=CAT_CAL)
        delta_hat = np.full(len(yo), np.nan)
        d_hat_m = np.asarray(model.predict(cal_v[NUM_CAL + CAT_CAL]), dtype=np.float64)
        delta_hat[~umo] = d_hat_m
        aobt_cal = oof["proxy_aobt"].to_numpy().astype(np.float64) - delta_hat
        aobt_cal[lirf_mask] = mvt_sched[lirf_mask]
        scores["AOBT_cal"] = score_t(yo, aobt_cal, umo, apo)
        log(f"  AOBT_cal: overall={scores['AOBT_cal']['rmse']:.2f} matched={scores['AOBT_cal']['matched']:.2f} "
            f"gt30={scores['AOBT_cal']['gt30m']:.1f}")

        # ---------------- PHASE 5: hybrid (proxy matched + E20 unmatched) ----------------
        hyb = oof["proxy_aobt"].to_numpy().astype(np.float64).copy()
        hyb[umo] = e20[umo]
        hyb[~np.isfinite(hyb)] = e20[~np.isfinite(hyb)]
        hyb[lirf_mask] = mvt_sched[lirf_mask]
        scores["AOBT+E20_unmatched"] = score_t(yo, hyb, umo, apo)

        hyb_cal = aobt_cal.copy()
        hyb_cal[umo] = e20[umo]
        hyb_cal[lirf_mask] = mvt_sched[lirf_mask]
        scores["AOBT_cal+E20_unmatched"] = score_t(yo, hyb_cal, umo, apo)
        log(f"  hybrid cal+E20: overall={scores['AOBT_cal+E20_unmatched']['rmse']:.2f} "
            f"matched={scores['AOBT_cal+E20_unmatched']['matched']:.2f}")

        # ---------------- PHASE 4: source disagreement ----------------
        e20_resid = np.abs(yo - e20)
        matched = ~umo & np.isfinite(e20_resid)
        disp = {}
        for dcol in ["d_aobt_eobt", "d_aobt_iobt", "d_aobt_lobt", "d_lobt_iobt", "d_lobt_eobt"]:
            d = oof[dcol].to_numpy().astype(np.float64)
            m = matched & np.isfinite(d)
            q = np.nanquantile(np.abs(d[m]), [0.5, 0.9, 0.99])
            tail = yo > 1800
            r = np.corrcoef(np.abs(d[m]), e20_resid[m])[0, 1]
            # tail rate above p90 of |disagreement|
            thr = q[1]
            hi = m & (np.abs(d) > thr)
            disp[dcol] = {"corr_abs_e20resid": float(r),
                          "tail_rate_lo": float(np.mean(tail[m & (np.abs(d) <= thr)])),
                          "tail_rate_hi": float(np.mean(tail[hi])) if hi.any() else float("nan")}
        scores.setdefault("_phase4", disp)
        payload.setdefault(split_name, {})["phase4"] = disp

        payload[split_name] = {
            "scores": scores,
            "coverage": {k: float(np.mean(np.isfinite(oof[col].to_numpy()))) for k, col in
                         [("AOBT", "proxy_aobt"), ("LOBT", "proxy_lobt"), ("IOBT", "proxy_iobt"),
                          ("EOBT", "proxy_eobt"), ("SCHED", "mvt_sched")]},
            "phase4": disp,
            "n_val": int(len(yo)), "n_matched": int((~umo).sum()), "n_unmatched": int(umo.sum()),
        }
        log(f"  PHASE4 disagreement -> E20 |resid| corr: { {k: round(v['corr_abs_e20resid'],3) for k,v in disp.items()} }")

    # ---------------- report ----------------
    write_report(payload)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    (RES / "E27.json").write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")
    save_result("E27", conv(payload))
    log("WROTE " + str(RES))


def write_report(payload):
    L = ["# E27 — AOBT off-block anchor audit", ""]
    L.append(f"**Ranking DEP AOBT coverage: 98.47%** (missing = the 5,290 unmatched rows). "
             f"Same null-set for LOBT/IOBT/EOBT. SCHED 100%.")
    L.append("")
    L.append("## Phase 1 — delta = BLOCK_TIME − AOBT (matched, full training)")
    p1 = payload["phase1"]
    a = p1["all"]
    L.append(f"| exact rate | median Δ | MAE | RMSE | p50 abs | p90 abs | p95 abs | p99 abs | max abs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    L.append(f"| {a['exact_rate']*100:.2f}% | {a['median_delta']:.0f}s | {a['mae']:.0f}s | {a['rmse']:.0f}s | "
             f"{a['p50_abs']:.0f}s | {a['p90_abs']:.0f}s | {a['p95_abs']:.0f}s | {a['p99_abs']:.0f}s | {a['max_abs']:.0f}s |")
    L.append("")
    L.append("| Airport | median Δ | MAE | RMSE | p90 abs |")
    L.append("|---|---:|---:|---:|---:|")
    for ap in AIRPORTS:
        r = p1["airports"][ap]
        L.append(f"| {ap} | {r['median_delta']:.0f}s | {r['mae']:.0f}s | {r['rmse']:.0f}s | {r['p90_abs']:.0f}s |")
    L.append("")
    for split, label in [("janjul", "Jan+Jul 2025 (primary)"), ("dec", "December 2025 (sanity)")]:
        p = payload[split]
        L.append(f"## Phase 2/3/5 — {label}")
        L.append("")
        L.append("| Model | overall | matched | >30m | >45m | >60m |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k in ["E20", "AOBT", "LOBT", "IOBT", "EOBT", "SCHED", "AOBT_cal",
                  "AOBT+E20_unmatched", "AOBT_cal+E20_unmatched"]:
            s = p["scores"][k]
            L.append(f"| {k} | {s['rmse']:.2f} | {s['matched']:.2f} | {s['gt30m']:.1f} | "
                     f"{s['gt45m']:.1f} | {s['gt60m']:.1f} |")
        L.append("")
        L.append(f"Proxy coverage: {p['coverage']}")
        L.append("")
        L.append("## Phase 4 — off-block source disagreement vs E20 residual (matched)")
        L.append("")
        L.append("| source pair | corr(abs diff, E20 |resid|) | >30m rate low | >30m rate high(p90+) |")
        L.append("|---|---:|---:|---:|")
        for k, v in p["phase4"].items():
            L.append(f"| {k} | {v['corr_abs_e20resid']:.3f} | {v['tail_rate_lo']:.3f} | {v['tail_rate_hi']:.3f} |")
        L.append("")
    L.append("## Phase 6 — leakage audit (AOBT must be < MVT)")
    L.append("")
    L.append(f"violations {payload['phase6']['violations']} / {payload['phase6']['n_checked']:,} "
             f"(AOBT−MVT p1 = {payload['phase6']['p1_sec']:.0f}s; min = {payload['phase6']['min_sec']:.0f}s)")
    L.append("")
    j = payload["janjul"]["scores"]
    L.append("## Bottom line")
    L.append("")
    L.append(f"AOBT raw proxy matched {j['AOBT']['matched']:.2f}; calibrated {j['AOBT_cal']['matched']:.2f}; "
             f"hybrid (cal + E20 unmatched) {j['AOBT_cal+E20_unmatched']['matched']:.2f} vs E20 {j['E20']['matched']:.2f}.")
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
