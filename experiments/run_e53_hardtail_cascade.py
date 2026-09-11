"""E53 — stage-3 hard-tail residual-of-residual cascade (crazy mode).

v13 = e20 + 0.5·(P+Δ−e20)·gate + 0.25·hat. Build residual3 = y − v13, train a
stage-3 LGB on residual3 with synthetic hard-tail amplification (rows weighted
by |residual3|, top-10% oversampled), features = v13's inputs + e20/P/rec/hat
meta. Apply as λ3·r3hat·gate3 (only where stage-3 predicts a large correction
or e20 is in the tail). Cross-month OOF. Metrics: matched RMSE, top1/top5 SSE.
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
from run_e50_matched_rich import pdf, fit_lgb, top_sse  # noqa: E402

RES = HERE / "results" / "E53"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_frames():
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ])
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20_pred", e20)), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def stage(tr, ev):
    ytr = tr["y"].to_numpy().astype(float); etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float); eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    d1 = fit_lgb(tr, ytr - Ptr, leaves=31, seed=1)
    rec1 = np.clip(Pev + np.asarray(d1.predict(pdf(ev)), float), 0, None)
    r1 = fit_lgb(tr, ytr - etr, leaves=31, seed=3)
    hat = np.asarray(r1.predict(pdf(ev)), float)
    p90 = float(np.quantile(etr, 0.90))
    gate2 = ((eev > p90) | ((rec1 - eev) > 200)).astype(float)
    v13 = eev + 0.5 * (rec1 - eev) * gate2 + 0.25 * hat
    # stage 3 target on TRAIN (in-sample v13 uses train's own models -> use hat/rec trained cross-month? use in-sample proxy)
    d1t = fit_lgb(tr, ytr - Ptr, leaves=31, seed=1)
    rec_t = np.clip(Ptr + np.asarray(d1t.predict(pdf(tr)), float), 0, None)
    hat_t = np.asarray(r1.predict(pdf(tr)), float)
    g2t = ((etr > p90) | ((rec_t - etr) > 200)).astype(float)
    v13_t = etr + 0.5 * (rec_t - etr) * g2t + 0.25 * hat_t
    r3 = ytr - v13_t
    # synthetic hard-tail amplification: weight by |r3|, top-10% *3
    w = np.abs(r3) + 1.0
    hard = np.abs(r3) > np.quantile(np.abs(r3), 0.9)
    w = np.where(hard, w * 3.0, w)
    m3 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    m3.fit(pdf(tr), r3, sample_weight=w, categorical_feature=["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"])
    r3hat = np.asarray(m3.predict(pdf(ev)), float)
    return {"y": yev, "e": eev, "P": Pev, "rec1": rec1, "hat": hat, "v13": v13, "r3hat": r3hat,
            "p90": p90, "ids": ev["MVT_ID_mvt"].to_numpy(), "ap": ev["airport"].to_numpy()}


def concat_parts(parts):
    keys = ["y", "e", "P", "rec1", "hat", "v13", "r3hat", "ids", "ap"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    return out


def main():
    fr = load_frames()
    parts = []
    for a, b in ((1, 7), (7, 1)):
        parts.append(stage(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    ojj = concat_parts(parts)
    odc = stage(fr["janjul"], fr["dec"])
    payload = {}
    for name, o in (("janjul", ojj), ("dec", odc)):
        y, e, v13 = o["y"], o["e"], o["v13"]
        base = {"matched": float(rmse(y, v13)), "top1": top_sse(y, v13, 0.01), "top5": top_sse(y, v13, 0.05)}
        cands = {"v13": base}
        r3 = o["r3hat"]
        for lam in (0.1, 0.2, 0.3, 0.5):
            p = v13 + lam * r3
            cands[f"v13_lam{lam}"] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "top5": top_sse(y, p, 0.05)}
            g = ((np.abs(r3) > 200) | (e > o["p90"])).astype(float)
            pg = v13 + lam * r3 * g
            cands[f"v13_g{lam}"] = {"matched": float(rmse(y, pg)), "top1": top_sse(y, pg, 0.01), "top5": top_sse(y, pg, 0.05)}
        payload[name] = {"v13": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for n, c in best:
            log(f"  [{name}] {n}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}% top5 {100*(c['top5']-base['top5'])/base['top5']:+.1f}%")
    (RES / "E53_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
