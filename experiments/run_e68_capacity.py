"""E68 — capacity / seed-averaging / extra stand-history features for the v18 Δ+hat.

Protocol as E67 (Jan<->Jul, Jan+Jul->Dec). Δ model screened with a fixed hat
(base config, lambda=0.45); winner layer re-evaluated with matched hat capacity.
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

RES = HERE / "results" / "E68"
RES.mkdir(parents=True, exist_ok=True)
LAM = 0.45
NUM_E = NUM_NEW + ["prev_mvt_aobt", "prev_clk_range", "n_arr_3h", "since_prev_arr_dep"]


def pdf(df, cols, cats):
    cols = [c for c in cols if c in df.columns]
    cats = [c for c in cats if c in df.columns]
    p = df.select(cols + cats).to_pandas()
    for c in cats:
        p[c] = p[c].fillna("NA").astype("category")
    return p, cats


def make_model(seed=1, **over):
    kw = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=seed)
    kw.update(over)
    return lgb.LGBMRegressor(**kw)


def attach_extra(df, dep_all, arr):
    d = df.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    prev = dep_all.select(["airport", "STAND_mvt", pl.col("MVT_TIME_UTC_mvt").alias("pmvt"),
                           pl.col("mvt_aobt").alias("prev_mvt_aobt"),
                           pl.col("clock_range").alias("prev_clk_range")]).sort(["airport", "STAND_mvt", "pmvt"])
    out = d.join_asof(prev, left_on="MVT_TIME_UTC_mvt", right_on="pmvt", by=["airport", "STAND_mvt"], strategy="backward")
    # n arrivals at stand in previous 3h (causal)
    a = arr.sort(["airport", "STAND_mvt", "aibt"]).select(["airport", "STAND_mvt", pl.col("aibt").cast(pl.Int64).alias("at")])
    d2 = out.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"]).with_columns(pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64).alias("mt"))
    d2 = d2.join_asof(a.rename({"at": "at2"}), left_on="mt", right_on="at2", by=["airport", "STAND_mvt"], strategy="backward")
    # count arrivals within 3h before via a second asof offset is complex; use since_arr-based flag from attach
    d2 = d2.with_columns(pl.col("since_arr").fill_null(1e9).alias("_sa"))
    d2 = d2.with_columns((pl.col("_sa") < 3 * 3600).cast(pl.Int32).alias("n_arr_3h"))
    if "since_arr_aobt" in d2.columns:
        d2 = d2.with_columns((pl.col("since_arr_aobt") - pl.col("since_arr")).alias("since_prev_arr_dep"))
    return d2.drop(["_sa", "at2", "mt"])


def screen(frames, cols, cats, base_hat, over):
    res = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                tr = frames["janjul"].filter(pl.col("_m") == a)
                ev = frames["janjul"].filter(pl.col("_m") == b)
                Xtr, cp = pdf(tr, cols, cats)
                Xev, _ = pdf(ev, cols, cats)
                m = make_model(**over)
                m.fit(Xtr, tr["y"].to_numpy().astype(float) - np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None), categorical_feature=cp)
                Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
                rec = np.clip(Pev + np.asarray(m.predict(Xev), float), 0, None)
                g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
                parts.append({"y": ev["y"].to_numpy().astype(float), "v": g + LAM * np.maximum(base_hat[split][b], 0)})
            o = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "v")}
        else:
            tr, ev = frames["janjul"], frames["dec"]
            Xtr, cp = pdf(tr, cols, cats); Xev, _ = pdf(ev, cols, cats)
            m = make_model(**over)
            m.fit(Xtr, tr["y"].to_numpy().astype(float) - np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None), categorical_feature=cp)
            Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
            rec = np.clip(Pev + np.asarray(m.predict(Xev), float), 0, None)
            g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
            o = {"y": ev["y"].to_numpy().astype(float), "v": g + LAM * np.maximum(base_hat[split], 0)}
        e2 = (o["y"] - o["v"]) ** 2
        k1 = max(1, int(0.01 * len(e2))); k5 = max(1, int(0.05 * len(e2)))
        res[split] = {"matched": float(rmse(o["y"], o["v"])), "top1": float(np.sort(e2)[::-1][:k1].sum() / e2.sum()),
                      "top5": float(np.sort(e2)[::-1][:k5].sum() / e2.sum())}
    return res


def main():
    log("E68 load")
    dep = add_extra(load_dep())
    arr = load_arr()
    dep_all = dep.select(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt", "mvt_aobt", "clock_range"])
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        f = attach(f, arr, dep_all)
        f = attach_extra(f, dep_all, arr)
        frames[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
        log(f"  {split} ready n={f.height}")

    # fixed base hat per split/month for screening
    base_hat = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            h = {}
            for a, b in ((1, 7), (7, 1)):
                tr = frames["janjul"].filter(pl.col("_m") == a); ev = frames["janjul"].filter(pl.col("_m") == b)
                Xtr, cp = pdf(tr, NUM_NEW, CAT2); Xev, _ = pdf(ev, NUM_NEW, CAT2)
                m = make_model(seed=7); m.fit(Xtr, tr["y"].to_numpy().astype(float) - tr["e20_pred"].to_numpy().astype(float), categorical_feature=cp)
                h[b] = np.asarray(m.predict(Xev), float)
            base_hat["janjul"] = h
        else:
            tr, ev = frames["janjul"], frames["dec"]
            Xtr, cp = pdf(tr, NUM_NEW, CAT2); Xev, _ = pdf(ev, NUM_NEW, CAT2)
            m = make_model(seed=7); m.fit(Xtr, tr["y"].to_numpy().astype(float) - tr["e20_pred"].to_numpy().astype(float), categorical_feature=cp)
            base_hat["dec"] = np.asarray(m.predict(Xev), float)
    log("  base hat done")

    configs = {
        "cap900": (NUM_NEW, CAT2, dict(n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40)),
        "deep": (NUM_NEW, CAT2, dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, min_child_samples=20, colsample_bytree=0.7)),
        "wide": (NUM_NEW, CAT2, dict(n_estimators=1500, learning_rate=0.025, num_leaves=255, min_child_samples=10, colsample_bytree=0.6, reg_lambda=10.0)),
        "e_feats": (NUM_E, CAT2, dict(n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40)),
        "deep_e": (NUM_E, CAT2, dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, min_child_samples=20, colsample_bytree=0.7)),
    }
    payload = {}
    for name, (cols, cats, over) in configs.items():
        log(f" spec {name}")
        payload[name] = screen(frames, cols, cats, base_hat, over)
        for s, v in payload[name].items():
            log(f"  [{name}/{s}] matched {v['matched']:8.3f} top1 {v['top1']:.4f} top5 {v['top5']:.4f}")

    (RES / "E68_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E68")


if __name__ == "__main__":
    main()
