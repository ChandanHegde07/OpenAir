"""E51c — tail-weighted and quantile Δ. Must beat 238.52 / 212.08."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse

RES = HERE / "results" / "E51"
SEED = 1
NUM = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(m, flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def gated(e, corr, p90, hthr=200.0):
    return ((e > p90) | (corr > hthr)).astype(float)


def load_frames():
    dep = load_dep().with_columns(
        [
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
            pl.col("ac_family").fill_null("NA"),
        ]
    )
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20_pred", e20)), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def fit(tr, target, w=None, alpha=None, seed=1):
    kw = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1)
    if alpha is not None:
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **kw)
    else:
        m = lgb.LGBMRegressor(**kw)
    m.fit(pdf(tr), target, sample_weight=w, categorical_feature=CAT)
    return m


def pair(tr, ev):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    dlt = ytr - Ptr
    rec_l2 = np.clip(Pev + fit(tr, dlt).predict(pdf(ev)), 0, None)
    w = np.clip((ytr / 1200.0) ** 1.5, 0.5, 6.0)
    rec_w = np.clip(Pev + fit(tr, dlt, w=w).predict(pdf(ev)), 0, None)
    rec_q = np.clip(Pev + fit(tr, dlt, alpha=0.65).predict(pdf(ev)), 0, None)
    rec_mix = 0.7 * rec_l2 + 0.3 * rec_q
    hat = np.asarray(fit(tr, ytr - etr, seed=7).predict(pdf(ev)), float)
    p90 = float(np.quantile(etr, 0.90))
    ids = ev["MVT_ID_mvt"].to_numpy()
    return dict(y=yev, e=eev, rec_l2=rec_l2, rec_w=rec_w, rec_q=rec_q, rec_mix=rec_mix, hat=hat, p90=p90, ids=ids)


def gated(e, corr, p90):
    return ((e > p90) | (corr > 200)).astype(float)


def main():
    log("E51c load")
    fr = load_frames()
    parts = []
    for a, b in ((1, 7), (7, 1)):
        log(f" fold {a}->{b}")
        parts.append(pair(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    def cat(key):
        return np.concatenate([p[key] for p in parts])
    ojj = {k: cat(k) for k in ("y", "e", "rec_l2", "rec_w", "rec_q", "rec_mix", "hat", "ids")}
    ojj["p90"] = float(np.mean([p["p90"] for p in parts]))
    log(" Dec")
    odc = pair(fr["janjul"], fr["dec"])
    payload = {}
    for split, o in (("janjul", ojj), ("dec", odc)):
        y, e, p90 = o["y"], o["e"], o["p90"]
        cands = {"e20": e}
        for tag in ("rec_l2", "rec_w", "rec_q", "rec_mix"):
            rec = o[tag]
            grec = e + 0.5 * (rec - e) * gated(e, rec - e, p90)
            cands[f"{tag}_g"] = grec
            cands[f"{tag}_gh25"] = grec + 0.25 * o["hat"]
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        base = rmse(y, e)
        t1 = top_sse(y, e, 0.01)
        out = {}
        for n, p in cands.items():
            pf = e20f.copy(); pf[idx] = p
            out[n] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "overall": float(rmse(yf, pf))}
        payload[split] = out
        for n, c in sorted(out.items(), key=lambda kv: kv[1]["matched"])[:8]:
            log(f"  [{split}] {n:16s} {c['matched']:.2f} ({c['matched']-base:+.2f}) top1 {(c['top1']-t1)/t1:+.1%} ov {c['overall']:.2f}")
    (RES / "E51c_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E51c")


if __name__ == "__main__":
    main()
