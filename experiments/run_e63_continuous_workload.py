"""E63 — continuous runway workload decomposition on v13.

Base = v13 (grec). Causal continuous runway workload: per (airport, runway)
session backlog_cum (entering AOBT - served MVT, reset on idle gaps), plus
service/queue features from E62. Residual LGB (y - v13) cross-month OOF;
alpha/gated blend; compare vs v13.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402
from run_e62_runway_queue import add_queue, QF  # noqa: E402

RES = HERE / "results" / "E63"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
EXTRA = ["backlog_cum", "wl_ratio"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
BNUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred"]
ALLF = QF + EXTRA


def pdf(df):
    p = df.select(BNUM + ALLF + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p[BNUM + ALLF + CAT]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def add_workload(dep):
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    mv = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy(); rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    ao = dep["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    bc = np.zeros(n); wlr = np.zeros(n)
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        ri = rw[sel]; ti = mv[sel]; aoi = ao[sel]
        v = ~np.isnat(aoi.astype("datetime64[ns]"))
        for r in np.unique(ri):
            m = ri == r
            idx = np.flatnonzero(m)
            order = np.argsort(ti[m], kind="stable")
            ts = ti[m][order]; aos = aoi[m][order]; vs = v[m][order]
            backlog = 0.0; last_t = None
            for k in range(len(ts)):
                if vs[k] and last_t is not None and (ts[k] - last_t) > 45 * 60 * 10**9:
                    backlog = 0.0  # idle gap resets the wave
                backlog = max(0.0, backlog + (1.0 if vs[k] else 0.0) - ((ts[k] - last_t) / 1e9 / 300.0 if last_t is not None else 0.0))
                bc[sel[idx[order[k]]]] = backlog
                wlr[sel[idx[order[k]]]] = backlog / 1.0
                if vs[k]:
                    last_t = ts[k]
    return dep.with_columns(pl.Series("backlog_cum", bc), pl.Series("wl_ratio", wlr))


def main():
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ])
    dep = add_queue(dep)
    dep = add_workload(dep)
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))

    def v13g(fit, ev):
        y0 = fit["y"].to_numpy().astype(float); e0 = fit["e20_pred"].to_numpy().astype(float)
        P0 = np.clip(fit["mvt_aobt"].to_numpy().astype(float), 0, None)
        evv = ev["e20_pred"].to_numpy().astype(float); Pv = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(fit), y0 - P0, categorical_feature=CAT)
        rec = np.clip(Pv + np.asarray(m.predict(pdf(ev)), float), 0, None)
        p90 = float(np.quantile(e0, 0.90))
        g = ((evv > p90) | ((rec - evv) > 200)).astype(float)
        return evv + 0.5 * (rec - evv) * g

    def resid_model(fit, target):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=200,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(fit), target, categorical_feature=CAT)
        return m

    payload = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            ys, v13s, crs = [], [], []
            for a, b in ((1, 7), (7, 1)):
                fit = fr["janjul"].filter(pl.col("_m") == a); ev = fr["janjul"].filter(pl.col("_m") == b)
                v = v13g(fit, ev)
                rm = resid_model(fit, fit["y"].to_numpy().astype(float) - v13g(fit, fit))
                cr = np.asarray(rm.predict(pdf(ev)), float)
                ys.append(ev["y"].to_numpy().astype(float)); v13s.append(v); crs.append(cr)
            y = np.concatenate(ys); v13 = np.concatenate(v13s); corr = np.concatenate(crs)
        else:
            fit = fr["janjul"]; ev = fr["dec"]
            v = v13g(fit, ev)
            rm = resid_model(fit, fit["y"].to_numpy().astype(float) - v13g(fit, fit))
            corr = np.asarray(rm.predict(pdf(ev)), float)
            y = ev["y"].to_numpy().astype(float); v13 = v
        base = {"matched": float(rmse(y, v13)), "top1": top1(y, v13)}
        cands = {"v13": base}
        for al in (0.1, 0.2, 0.3, 0.5):
            p = v13 + al * corr
            cands[f"a{al}"] = {"matched": float(rmse(y, p)), "top1": top1(y, p)}
        payload[split] = {"v13": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for nm, c in best:
            log(f"  [{split}] {nm}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}%")
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
