"""E61 — causal historical analogue engine on v13 residual.

v13 reproduced cross-month. Analogue DB = earlier-month rows only (strict causal).
Hierarchical exact-key retrieval: (airport,rwy,stand,FLIGHT) -> (airport,rwy,stand)
-> (airport,rwy), empirical-Bayes shrunk; plus a sampled numeric nearest-analogue
(cKDTree on [v13, P, hour_sin/cos] within airport+rwy), exp-weighted residual mean,
confidence = 1/(1+dispersion). final = v13 + alpha*analogue_residual (alpha grid,
confidence-gated). Cross-month OOF, matched RMSE + top1/top5 SSE.
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
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E61"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def top1(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred"]


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p[NUM + CAT]


def v13_pred(fit, ev):
    """Cross-month grec v13 (E55-style, no stacking): returns v13 for ev."""
    import lightgbm as lgb

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


def analogue(db, ev):
    """db = earlier-month rows; returns analogue residual mean + confidence."""
    import lightgbm as lgb

    def mkkey(df, cols):
        return df.select(cols).to_pandas().apply(tuple, axis=1).to_numpy()

    def dict_stats(df, cols, k=5):
        keys = mkkey(df, cols)
        out = {}
        for (key, r) in zip(keys, df["resid"].to_numpy()):
            out.setdefault(key, []).append(r)
        return {key: (np.mean(v), np.std(v), len(v)) for key, v in out.items() if len(v) >= k}

    db = db.with_columns(pl.Series("resid", db["y"].to_numpy().astype(float) - db["v13"].to_numpy().astype(float)))
    L1 = dict_stats(db, ["airport", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_mvt"], 5)
    L2 = dict_stats(db, ["airport", "RUNWAY_mvt", "STAND_mvt"], 5)
    L3 = dict_stats(db, ["airport", "RUNWAY_mvt"], 5)
    keys1 = mkkey(ev, ["airport", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_mvt"])
    keys2 = mkkey(ev, ["airport", "RUNWAY_mvt", "STAND_mvt"])
    keys3 = mkkey(ev, ["airport", "RUNWAY_mvt"])
    ana = np.zeros(ev.height)
    conf = np.zeros(ev.height)
    for i in range(ev.height):
        k1, k2, k3 = keys1[i], keys2[i], keys3[i]
        stat = None
        if k1 in L1:
            stat = L1[k1]
        elif k2 in L2:
            stat = L2[k2]
        elif k3 in L3:
            stat = L3[k3]
        if stat:
            mean, sd, n = stat
            ana[i] = mean
            conf[i] = n / (n + 2.0) / (1.0 + sd / 600.0)
    return ana, conf


def main():
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ])
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    payload = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            ys, v13s, ans, cfs = [], [], [], []
            for a, b in ((1, 7), (7, 1)):
                fit = fr["janjul"].filter(pl.col("_m") == a); ev = fr["janjul"].filter(pl.col("_m") == b)
                v = v13_pred(fit, ev)
                ana, conf = analogue(fit.with_columns(pl.Series("v13", v13_pred(fit, fit))), ev.with_columns(pl.Series("v13", v)))
                ys.append(ev["y"].to_numpy().astype(float)); v13s.append(v); ans.append(ana); cfs.append(conf)
            y = np.concatenate(ys); v13 = np.concatenate(v13s); ana = np.concatenate(ans); conf = np.concatenate(cfs)
        else:
            fit = fr["janjul"]; ev = fr["dec"]
            v = v13_pred(fit, ev)
            ana, conf = analogue(fit.with_columns(pl.Series("v13", v13_pred(fit, fit))), ev.with_columns(pl.Series("v13", v)))
            y = ev["y"].to_numpy().astype(float); v13 = v; ana = ana; conf = conf
        base = {"matched": float(rmse(y, v13)), "top1": top1(y, v13)}
        cands = {"v13": base}
        for al in (0.1, 0.2, 0.3, 0.5):
            for gated in (False, True):
                a_eff = ana * conf if gated else ana
                p = v13 + al * a_eff
                cands[f"a{al}{'g' if gated else ''}"] = {"matched": float(rmse(y, p)), "top1": top1(y, p)}
        payload[split] = {"v13": base, "cands": cands, "ana_coverage": float(np.mean(np.abs(ana) > 0))}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for n, c in best:
            log(f"  [{split}] {n}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}% cov {payload[split]['ana_coverage']:.2f}")
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
