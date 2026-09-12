"""E53 — LIRF-stronger grec + previous ARR taxi-in. Beat v13 238.52/212.08 both splits."""
from __future__ import annotations

import glob
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
from common import TRAIN_FILES, load_dep, rmse
from matched_submit import CAT, NUM, add_extra, e20_col, grec, log

RES = HERE / "results" / "E53"
RES.mkdir(parents=True, exist_ok=True)
V13 = (238.52, 212.08)
NUM2 = NUM + ["arr_taxiin", "since_arr"]


def pdf(df, cols):
    p = df.select([c for c in cols if c in df.columns] + [c for c in CAT if c in df.columns]).to_pandas()
    for c in CAT:
        if c in p.columns:
            p[c] = p[c].fillna("NA").astype("category")
    return p


def load_arr_feats():
    arr = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            [
                pl.col("ADES_mvt").alias("airport"),
                "STAND_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
            ]
        )
        .drop_nulls()
        .collect()
        .sort(["airport", "STAND_mvt", "aibt"])
    )
    return arr


def attach_arr(dep, arr):
    d = dep.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    j = d.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "STAND_mvt"], strategy="backward")
    j = j.with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("aibt")).dt.total_seconds().alias("since_arr"),
    )
    j = j.with_columns(
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600))
        .then(pl.col("arr_taxiin"))
        .otherwise(None)
        .alias("arr_taxiin"),
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600))
        .then(pl.col("since_arr"))
        .otherwise(None)
        .alias("since_arr"),
    )
    return j


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def pair(tr, ev, use_arr=False, lirf_lam=0.5):
    cols = NUM2 if use_arr else NUM
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    kw = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=1)
    dmod = lgb.LGBMRegressor(**kw)
    dmod.fit(pdf(tr, cols), ytr - Ptr, categorical_feature=[c for c in CAT if c in tr.columns])
    rec = np.clip(Pev + np.asarray(dmod.predict(pdf(ev, cols)), float), 0, None)
    hmod = lgb.LGBMRegressor(**{**kw, "random_state": 7})
    hmod.fit(pdf(tr, cols), ytr - etr, categorical_feature=[c for c in CAT if c in tr.columns])
    hat = np.asarray(hmod.predict(pdf(ev, cols)), float)
    p90 = float(np.quantile(etr, 0.90))
    g = grec(eev, rec, 0.5, p90, 200)
    # LIRF stronger grec
    ap = ev["airport"].to_numpy()
    g2 = g.copy()
    m = ap == "LIRF"
    g2[m] = grec(eev[m], rec[m], lirf_lam, p90, 200)
    v13 = g + 0.25 * hat
    v_lirf = g2 + 0.25 * hat
    return dict(y=yev, e=eev, v13=v13, v_lirf=v_lirf, g=g, hat=hat, rec=rec, p90=p90, ap=ap, ids=ev["MVT_ID_mvt"].to_numpy())


def main():
    log("E53 load")
    dep = add_extra(load_dep())
    arr = load_arr_feats()
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        f = attach_arr(f, arr)
        frames[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
        log(f"  {split} arr_taxiin coverage {float(f['arr_taxiin'].is_not_null().mean()):.3f}")

    payload = {}
    for use_arr in (False, True):
        tag = "arr" if use_arr else "base"
        log(f" spec {tag}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            parts.append(pair(frames["janjul"].filter(pl.col("_m") == a), frames["janjul"].filter(pl.col("_m") == b), use_arr=use_arr, lirf_lam=0.7))
        ojj = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "e", "v13", "v_lirf", "g", "hat", "rec", "ids")}
        odc = pair(frames["janjul"], frames["dec"], use_arr=use_arr, lirf_lam=0.7)
        payload[tag] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            y, e = o["y"], o["e"]
            cands = {
                "v13": o["v13"],
                "lirf70": o["v_lirf"],
                "h20": o["g"] + 0.20 * o["hat"],
            }
            oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
            e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
            yf = oof["y"].to_numpy().astype(float)
            pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
            idx = np.array([pos[int(i)] for i in o["ids"]])
            out = {}
            for n, p in cands.items():
                pf = e20f.copy()
                pf[idx] = p
                out[n] = {"matched": float(rmse(y, p)), "overall": float(rmse(yf, pf))}
            payload[tag][split] = out
            for n, c in out.items():
                log(f"  [{tag}/{split}] {n:8s} {c['matched']:.2f} ov {c['overall']:.2f}")

    (RES / "E53_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E53")


if __name__ == "__main__":
    main()
