"""E54 — meta-layer corrector on v13 (beast mode, clean).

Genuinely new meta features vs E53: individual E20 experts (pred_C/D/E) and
seed-variance of the Δ model, plus P/e20 ratio. Stage-3 LGB target r3 = y − v13
with hard-tail amplification; v13 + λ·r3hat·gate. Cross-month OOF.
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

RES = HERE / "results" / "E54"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
META = ["pred_C", "pred_D", "pred_E", "e20_pred", "mvt_aobt", "mvt_sched", "aobt_eobt",
        "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "seed_std", "P_over_e20"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


BASE = [c for c in META if c != "seed_std"]


def pdf(df, cols=META):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def fit(tr, target, leaves=31, seed=SEED, n=400, w=None, cols=BASE):
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.04, num_leaves=leaves, min_child_samples=100,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(pdf(tr, cols), target, sample_weight=w, categorical_feature=CAT)
    return m


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def stage(tr, ev):
    y0 = tr["y"].to_numpy().astype(float); e0 = tr["e20_pred"].to_numpy().astype(float)
    P0 = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yv = ev["y"].to_numpy().astype(float); evv = ev["e20_pred"].to_numpy().astype(float)
    Pv = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    # Δ rec via 3 seeds; seed_std
    recs_t, recs_v = [], []
    for s in (1, 2, 3):
        m = fit(tr, y0 - P0, seed=s)
        recs_v.append(np.clip(Pv + np.asarray(m.predict(pdf(ev, BASE)), float), 0, None))
        recs_t.append(np.clip(P0 + np.asarray(m.predict(pdf(tr, BASE)), float), 0, None))
    rec1, rec_t = recs_v[0], recs_t[0]
    tr = tr.with_columns(pl.Series("seed_std", np.std(recs_t, axis=0)))
    ev = ev.with_columns(pl.Series("seed_std", np.std(recs_v, axis=0)))
    mh = fit(tr, y0 - e0, seed=3, cols=META)
    hat_t = np.asarray(mh.predict(pdf(tr)), float)
    hat_v = np.asarray(mh.predict(pdf(ev)), float)
    p90 = float(np.quantile(e0, 0.90))
    def v13(e, P, rec, hat):
        g = ((e > p90) | ((rec - e) > 200)).astype(float)
        return e + 0.5 * (rec - e) * g + 0.25 * hat
    v13t = v13(e0, P0, rec_t, hat_t)
    v13v = v13(evv, Pv, rec1, hat_v)
    r3 = y0 - v13t
    w = np.abs(r3) + 1.0
    hard = np.abs(r3) > np.quantile(np.abs(r3), 0.9)
    w = np.where(hard, w * 3.0, w)
    m3 = fit(tr, r3, n=500, w=w, cols=META)
    r3hat = np.asarray(m3.predict(pdf(ev)), float)
    return {"y": yv, "e": evv, "v13": v13v, "r3hat": r3hat, "p90": p90,
            "ids": ev["MVT_ID_mvt"].to_numpy()}


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
        f = f.with_columns((pl.col("mvt_aobt").clip(0, None) / (pl.col("e20_pred").abs() + 60)).alias("P_over_e20"))
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
        r = o["r3hat"]
        for lam in (0.1, 0.2, 0.3):
            for tag, p in (("", o["v13"] + lam * r),
                           ("g", o["v13"] + lam * r * ((np.abs(r) > 150) | (o["e"] > o["p90"])).astype(float))):
                cands[f"lam{lam}{tag}"] = {"matched": float(rmse(o["y"], p)), "top1": top1(o["y"], p)}
        payload[name] = {"v13": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for n, c in best:
            log(f"  [{name}] {n}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}%")
    (RES / "E54_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
