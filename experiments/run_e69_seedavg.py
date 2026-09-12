"""E69 — seed-averaged deep Δ + deep leftover hat, tuned λ.

Protocol as E67/E68. Baseline v18 236.59 / 210.22.
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
from common import rmse
from matched_submit import e20_col, grec, log
from run_e55_inbound_v2 import CAT2, NUM_NEW
from run_e68_capacity import attach_extra, pdf, make_model

RES = HERE / "results" / "E69"
RES.mkdir(parents=True, exist_ok=True)
from common import load_dep
from matched_submit import add_extra
from run_e55_inbound_v2 import attach, load_arr

DEEP = dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, min_child_samples=20, colsample_bytree=0.7)
DSEEDS = [1, 2, 3]
HSEEDS = [7, 11]


def rec_avg(tr, ev, cols, cats, seeds):
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    Xtr, cp = pdf(tr, cols, cats); Xev, _ = pdf(ev, cols, cats)
    out = np.zeros(ev.height)
    for s in seeds:
        m = make_model(seed=s, **DEEP)
        m.fit(Xtr, tr["y"].to_numpy().astype(float) - np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None), categorical_feature=cp)
        out += np.asarray(m.predict(Xev), float)
    return np.clip(Pev + out / len(seeds), 0, None)


def hat_avg(tr, ev, cols, cats, seeds):
    Xtr, cp = pdf(tr, cols, cats); Xev, _ = pdf(ev, cols, cats)
    out = np.zeros(ev.height)
    for s in seeds:
        m = make_model(seed=s, **DEEP)
        m.fit(Xtr, tr["y"].to_numpy().astype(float) - tr["e20_pred"].to_numpy().astype(float), categorical_feature=cp)
        out += np.asarray(m.predict(Xev), float)
    return out / len(seeds)


def main():
    log("E69 load")
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

    cols, cats = NUM_NEW, CAT2
    log("train deep seed-avg Δ and hat...")
    res = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                tr = frames["janjul"].filter(pl.col("_m") == a)
                ev = frames["janjul"].filter(pl.col("_m") == b)
                rec = rec_avg(tr, ev, cols, cats, DSEEDS)
                hat = hat_avg(tr, ev, cols, cats, HSEEDS)
                g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
                parts.append({"y": ev["y"].to_numpy().astype(float), "g": g, "hat": hat})
            res[split] = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "g", "hat")}
        else:
            tr, ev = frames["janjul"], frames["dec"]
            rec = rec_avg(tr, ev, cols, cats, DSEEDS)
            hat = hat_avg(tr, ev, cols, cats, HSEEDS)
            g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
            res[split] = {"y": ev["y"].to_numpy().astype(float), "g": g, "hat": hat}
        log(f"  {split} g={rmse(res[split]['y'],res[split]['g']):.3f}")

    payload = {}
    for split in ("janjul", "dec"):
        y, g, hat = res[split]["y"], res[split]["g"], res[split]["hat"]
        payload[split] = {}
        for lam in (0.0, 0.3, 0.4, 0.45, 0.5, 0.6):
            p = g + lam * np.maximum(hat, 0)
            e2 = (y - p) ** 2
            k1 = max(1, int(0.01 * len(e2))); k5 = max(1, int(0.05 * len(e2)))
            payload[split][f"lam{lam}"] = {"matched": float(rmse(y, p)), "top1": float(np.sort(e2)[::-1][:k1].sum() / e2.sum()),
                                           "top5": float(np.sort(e2)[::-1][:k5].sum() / e2.sum())}
            log(f"  [{split}] lam={lam:4.2f} matched {rmse(y,p):8.3f} top1 {payload[split][f'lam{lam}']['top1']:.4f}")

    (RES / "E69_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E69")


if __name__ == "__main__":
    main()
