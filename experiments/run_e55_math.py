"""E55 — extreme-tail non-linear stacking + magnitude-conditioned blend.

Math: collect diverse predictors (e20, Δ-rec1/rec2, q85-quantile Δ, residual
hat), train a non-linear LightGBM stacker on y (hard-row amplified), then
blend into v13 with magnitude-conditioned lambda (piecewise in e20 magnitude).
Cross-month OOF. Matched RMSE + top1/top5 SSE.
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
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E55"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FEA = ["pred_C", "pred_D", "pred_E", "e20_pred", "mvt_aobt", "mvt_sched", "aobt_eobt",
       "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month",
       "rec1", "rec2", "q85", "hat"]
RAW = [c for c in FEA if c not in ("rec1", "rec2", "q85", "hat")]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols=FEA):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def fit(tr, target, obj="regression", leaves=31, seed=SEED, n=500, w=None, cols=RAW):
    m = lgb.LGBMRegressor(objective=obj, n_estimators=n, learning_rate=0.04, num_leaves=leaves,
                          min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(pdf(tr, cols), target, sample_weight=w, categorical_feature=CAT)
    return m


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def stage(tr, ev):
    y0 = tr["y"].to_numpy().astype(float); e0 = tr["e20_pred"].to_numpy().astype(float)
    P0 = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yv = ev["y"].to_numpy().astype(float); ev0 = ev["e20_pred"].to_numpy().astype(float)
    Pv = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    m1 = fit(tr, y0 - P0, leaves=31, seed=1)
    m2 = fit(tr, y0 - P0, leaves=63, seed=2)
    mq = fit(tr, y0 - P0, obj="quantile", leaves=31, seed=4, n=400)
    mq.set_params(alpha=0.85)
    rec1 = np.clip(Pv + np.asarray(m1.predict(pdf(ev, RAW)), float), 0, None)
    rec2 = np.clip(Pv + np.asarray(m2.predict(pdf(ev, RAW)), float), 0, None)
    q85 = np.clip(Pv + np.asarray(mq.predict(pdf(ev, RAW)), float), 0, None)
    rec1_t = np.clip(P0 + np.asarray(m1.predict(pdf(tr, RAW)), float), 0, None)
    rec2_t = np.clip(P0 + np.asarray(m2.predict(pdf(tr, RAW)), float), 0, None)
    q85_t = np.clip(P0 + np.asarray(mq.predict(pdf(tr, RAW)), float), 0, None)
    mh = fit(tr, y0 - e0, leaves=31, seed=3)
    hat = np.asarray(mh.predict(pdf(ev, RAW)), float)
    hat_t = np.asarray(mh.predict(pdf(tr, RAW)), float)
    for nm, arr_t, arr_v in [("rec1", rec1_t, rec1), ("rec2", rec2_t, rec2), ("q85", q85_t, q85), ("hat", hat_t, hat)]:
        tr = tr.with_columns(pl.Series(nm, arr_t))
        ev = ev.with_columns(pl.Series(nm, arr_v))
    p90 = float(np.quantile(e0, 0.90))
    def v13(e, P, rec, h):
        g = ((e > p90) | ((rec - e) > 200)).astype(float)
        return e + 0.5 * (rec - e) * g + 0.25 * h
    v13t = v13(e0, P0, rec1_t, hat_t)
    v13v = v13(ev0, Pv, rec1, hat)
    # non-linear stacker on y, hard-amplified
    r = y0 - v13t
    w = np.abs(r) + 1.0
    hard = np.abs(r) > np.quantile(np.abs(r), 0.9)
    w = np.where(hard, w * 3.0, w)
    ms = fit(tr, y0, n=600, w=w, cols=FEA)
    stack = np.asarray(ms.predict(pdf(ev, FEA)), float)
    return {"y": yv, "e": ev0, "v13": v13v, "stack": stack, "p90": p90, "P": Pv, "ids": ev["MVT_ID_mvt"].to_numpy()}


def concat_parts(parts):
    keys = [k for k in parts[0] if k != "p90"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    return out


def main():
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ])
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    parts = []
    for a, b in ((1, 7), (7, 1)):
        parts.append(stage(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    ojj = concat_parts(parts)
    odc = stage(fr["janjul"], fr["dec"])
    payload = {}
    for name, o in (("janjul", ojj), ("dec", odc)):
        base = {"matched": float(rmse(o["y"], o["v13"])), "top1": top1(o["y"], o["v13"])}
        cands = {"v13": base}
        s = o["stack"]
        # magnitude-conditioned blend: lambda by e20 quantile bins
        ebins = np.quantile(o["e"], [0, 0.8, 0.92, 1.0])
        for lam_grid in [(0.2, 0.4, 0.6), (0.3, 0.5, 0.7)]:
            lam = np.where(o["e"] > ebins[2], lam_grid[2], np.where(o["e"] > ebins[1], lam_grid[1], lam_grid[0]))
            p = o["v13"] + lam * (s - o["v13"])
            cands[f"mag_{lam_grid[0]}_{lam_grid[1]}_{lam_grid[2]}"] = {"matched": float(rmse(o["y"], p)), "top1": top1(o["y"], p)}
        for g_lam in (0.3, 0.5):
            g = ((np.abs(s - o["v13"]) > 150) | (o["e"] > o["p90"])).astype(float)
            p = o["v13"] + g_lam * (s - o["v13"]) * g
            cands[f"gate_{g_lam}"] = {"matched": float(rmse(o["y"], p)), "top1": top1(o["y"], p)}
        payload[name] = {"v13": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for n, c in best:
            log(f"  [{name}] {n}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}%")
    (RES / "E55_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
