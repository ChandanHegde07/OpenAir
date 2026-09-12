"""E66 — reproduce v18 matched OOF and dump residual diagnostics.

Diagnostic only (no new model). Writes an OOF parquet for Jan+Jul and Dec with
y, e20, and the v18-family predictions so the remaining matched SSE can be
decomposed by airport, delta, and tail.
"""
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
from matched_submit import CAT, NUM, add_extra, e20_col, grec, log
from run_e55_inbound_v2 import CAT2, NUM_NEW, attach, load_arr

RES = HERE / "results" / "E66"
RES.mkdir(parents=True, exist_ok=True)


def pdf(df, cols, cats=CAT2):
    p = df.select([c for c in cols if c in df.columns] + [c for c in cats if c in df.columns]).to_pandas()
    for c in cats:
        if c in p.columns:
            p[c] = p[c].fillna("NA").astype("category")
    return p


KW = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
          subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=1)


def fit_predict(tr, ev, cols, cats=CAT2):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    eev = ev["e20_pred"].to_numpy().astype(float)
    cats_p = [c for c in cats if c in tr.columns]
    dmod = lgb.LGBMRegressor(**KW)
    dmod.fit(pdf(tr, cols, cats), ytr - Ptr, categorical_feature=cats_p)
    rec = np.clip(Pev + np.asarray(dmod.predict(pdf(ev, cols, cats)), float), 0, None)
    hmod = lgb.LGBMRegressor(**{**KW, "random_state": 7})
    hmod.fit(pdf(tr, cols, cats), ytr - etr, categorical_feature=cats_p)
    hat = np.asarray(hmod.predict(pdf(ev, cols, cats)), float)
    p90 = float(np.quantile(etr, 0.90))
    g = grec(eev, rec, 0.5, p90, 200)
    return dict(y=ev["y"].to_numpy().astype(float), e=eev, rec=rec, hat=hat,
                v18=g + 0.28 * hat, g=g, p90=p90)


def top_sse(y, p, frac):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum()), float(e2.sum())


def main():
    log("E66 load")
    dep = add_extra(load_dep())
    arr = load_arr()
    dep_all = dep.select(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        f = attach(f, arr, dep_all)
        frames[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))

    payload = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                parts.append(fit_predict(frames["janjul"].filter(pl.col("_m") == a),
                                         frames["janjul"].filter(pl.col("_m") == b), NUM_NEW))
            o = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "e", "rec", "hat", "v18", "g")}
            o["p90"] = float(np.mean([p["p90"] for p in parts]))
            ap = np.concatenate([frames["janjul"].filter(pl.col("_m") == b)["airport"].to_numpy() for a, b in ((1, 7), (7, 1))])
            ids = np.concatenate([frames["janjul"].filter(pl.col("_m") == b)["MVT_ID_mvt"].to_numpy() for a, b in ((1, 7), (7, 1))])
            P = np.concatenate([np.clip(frames["janjul"].filter(pl.col("_m") == b)["mvt_aobt"].to_numpy().astype(float), 0, None) for a, b in ((1, 7), (7, 1))])
            mon = np.concatenate([frames["janjul"].filter(pl.col("_m") == b)["month"].to_numpy() for a, b in ((1, 7), (7, 1))]).astype(int)
        else:
            o = fit_predict(frames["janjul"], frames["dec"], NUM_NEW)
            ap = frames["dec"]["airport"].to_numpy()
            ids = frames["dec"]["MVT_ID_mvt"].to_numpy()
            P = np.clip(frames["dec"]["mvt_aobt"].to_numpy().astype(float), 0, None)
            mon = np.zeros(len(o["y"]), dtype=int)
        out = pl.DataFrame({
            "MVT_ID_mvt": ids, "airport": ap, "month": mon,
            "y": o["y"], "e20": o["e"], "P": P,
            "rec": o["rec"], "hat": o["hat"], "g": o["g"], "v18": o["v18"],
        })
        out.write_parquet(RES / f"oof_v18_{split}.parquet")

        y, p = o["y"], o["v18"]
        r = y - p
        sse1, ssetot = top_sse(y, p, 0.01)
        sse5, _ = top_sse(y, p, 0.05)
        diag = {
            "matched": float(rmse(y, p)),
            "mean_resid": float(np.mean(r)),
            "top1_sse_share": sse1 / ssetot,
            "top5_sse_share": sse5 / ssetot,
            "delta_mean": float(np.mean(y - P)),
            "delta_p50": float(np.median(y - P)),
            "delta_p99": float(np.quantile(y - P, 0.99)),
        }
        # per-airport
        per = {}
        for a in np.unique(ap):
            m = ap == a
            e2 = (y[m] - p[m]) ** 2
            per[a] = {"n": int(m.sum()), "rmse": float(np.sqrt(e2.mean())),
                      "sse_share": float(e2.sum() / ssetot)}
        diag["per_airport"] = per
        # bins by actual y
        bins = [(0, 600), (600, 900), (900, 1200), (1200, 1800), (1800, 3600), (3600, 7200), (7200, 1e12)]
        bstat = []
        for lo, hi in bins:
            m = (y >= lo) & (y < hi)
            if m.sum() == 0:
                continue
            e2 = (y[m] - p[m]) ** 2
            bstat.append({"lo": lo, "hi": hi, "n": int(m.sum()), "rmse": float(np.sqrt(e2.mean())),
                          "sse_share": float(e2.sum() / ssetot), "mean_resid": float(np.mean((y - p)[m]))})
        diag["by_y"] = bstat
        # bins by delta = y-P
        dbins = [(-1e12, 0), (0, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 1e12)]
        dstat = []
        d = y - P
        for lo, hi in dbins:
            m = (d >= lo) & (d < hi)
            if m.sum() == 0:
                continue
            e2 = (y[m] - p[m]) ** 2
            dstat.append({"lo": lo, "hi": hi, "n": int(m.sum()), "rmse": float(np.sqrt(e2.mean())),
                          "sse_share": float(e2.sum() / ssetot), "mean_resid": float(np.mean((y - p)[m]))})
        diag["by_delta"] = dstat
        payload[split] = diag
        log(f"[{split}] matched {diag['matched']:.2f} top1 {diag['top1_sse_share']:.3f} top5 {diag['top5_sse_share']:.3f}")

    (RES / "E66_diag.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E66 diag")


if __name__ == "__main__":
    main()
