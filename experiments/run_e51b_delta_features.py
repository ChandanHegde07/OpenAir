"""E51b — v13 recipe + unused ranking-safe features in Δ.

Adds AIRCRAFT_OPERATOR, WK_TBL_CAT, causal roll30 aobt_eobt.
Must beat v13 238.52 / 212.08.
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
from common import add_causal_rolling, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E51"
SEED = 1
NUM = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
    "roll10_mean_mvt_aobt", "roll10_rwy_mean_mvt_aobt",
]
CAT = [
    "airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt",
    "MARKET_SEGMENT_flt", "ac_family", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt",
]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select([c for c in NUM + CAT if c in df.columns]).to_pandas()
    for c in CAT:
        if c in p.columns:
            p[c] = p[c].fillna("NA").astype("category")
    return p


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def fit_lgb(tr, target, seed=1):
    cols = [c for c in NUM + CAT if c in tr.columns]
    p = tr.select(cols).to_pandas()
    cats = [c for c in CAT if c in p.columns]
    for c in cats:
        p[c] = p[c].fillna("NA").astype("category")
    m = lgb.LGBMRegressor(
        n_estimators=450, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(p, target, categorical_feature=cats)
    return m, cols, cats


def pred_lgb(m, df, cols, cats):
    p = df.select(cols).to_pandas()
    for c in cats:
        p[c] = p[c].fillna("NA").astype("category")
    return np.asarray(m.predict(p), float)


def load_frames():
    dep = add_causal_rolling(load_dep()).with_columns(
        [
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
            pl.col("ac_family").fill_null("NA"),
            pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"),
            pl.col("WK_TBL_CAT_flt").fill_null("NA"),
        ]
    )
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        f = dep.join(
            oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20_pred", e20)),
            on="MVT_ID_mvt",
            how="inner",
        )
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite())
        f = f.filter(pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def gated(e, corr, p90, hthr=200.0):
    return ((e > p90) | (corr > hthr)).astype(float)


def pair(tr, ev):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    dmod, cols, cats = fit_lgb(tr, ytr - Ptr, seed=1)
    rec = np.clip(Pev + pred_lgb(dmod, ev, cols, cats), 0, None)
    hmod, hc, hca = fit_lgb(tr, ytr - etr, seed=7)
    hat = pred_lgb(hmod, ev, hc, hca)
    p90 = float(np.quantile(etr, 0.90))
    grec = eev + 0.5 * (rec - eev) * gated(eev, rec - eev, p90, 200)
    v13 = grec + 0.25 * hat
    return yev, eev, rec, hat, grec, v13, p90, ev["MVT_ID_mvt"].to_numpy()


def main():
    log("E51b load+rolling")
    fr = load_frames()
    ys, es, recs, hats, g13, ids = [], [], [], [], [], []
    for a, b in ((1, 7), (7, 1)):
        log(f"  fold {a}->{b}")
        y, e, rec, hat, grec, v13, p90, idv = pair(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b))
        ys.append(y); es.append(e); recs.append(rec); hats.append(hat); g13.append(v13); ids.append(idv)
        log(f"    fold v13-like {rmse(y, v13):.2f}")
    ojj = {
        "y": np.concatenate(ys), "e": np.concatenate(es), "rec": np.concatenate(recs),
        "hat": np.concatenate(hats), "v13": np.concatenate(g13), "ids": np.concatenate(ids),
    }
    log("  Dec")
    y, e, rec, hat, grec, v13, p90, idv = pair(fr["janjul"], fr["dec"])
    odc = {"y": y, "e": e, "rec": rec, "hat": hat, "v13": v13, "ids": idv, "p90": p90}

    payload = {"splits": {}}
    for split, o in (("janjul", ojj), ("dec", odc)):
        y, e = o["y"], o["e"]
        # extra λ on leftover
        cands = {"e20": e, "v13feat": o["v13"]}
        for lam in (0.20, 0.25, 0.30, 0.35):
            cands[f"h{lam}"] = o["v13"]  # already 0.25; rebuild from rec
        # rebuild grec from rec with p90 of this split via e quantile
        p90 = float(np.quantile(e, 0.90))
        grec = e + 0.5 * (o["rec"] - e) * gated(e, o["rec"] - e, p90, 200)
        cands["grec"] = grec
        for lam in (0.20, 0.25, 0.30, 0.35, 0.40):
            cands[f"grec_h{lam}"] = grec + lam * o["hat"]
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        base = rmse(y, e)
        t1 = top_sse(y, e, 0.01)
        out = {}
        for n, p in cands.items():
            pf = e20f.copy()
            pf[idx] = p
            out[n] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "overall": float(rmse(yf, pf))}
        payload["splits"][split] = {"e20": base, "cands": out}
        for n, c in sorted(out.items(), key=lambda kv: kv[1]["matched"])[:8]:
            log(f"  [{split}] {n:12s} {c['matched']:.2f} ({c['matched']-base:+.2f}) top1 {(c['top1']-t1)/t1:+.1%} ov {c['overall']:.2f}")

    (RES / "E51b_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE E51b")


if __name__ == "__main__":
    main()
