"""E70 — cap900 + 3-seed Δ / 2-seed hat + λ tuning (v19 recipe).

Cross-month OOF as E54/E55. Reports matched and pooled overall so the v19
submission is documented on both metrics.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse
from matched_submit import add_extra, e20_col, grec, log
from run_e55_inbound_v2 import CAT2, NUM_NEW, attach, load_arr
from run_e68_capacity import attach_extra, pdf, make_model

RES = HERE / "results" / "E70"
RES.mkdir(parents=True, exist_ok=True)
OVER = dict(n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40)
DS, HS = [1, 2, 3], [101, 102]


def rec_avg(tr, ev):
    Xtr, cp = pdf(tr, NUM_NEW, CAT2)
    Xev, _ = pdf(ev, NUM_NEW, CAT2)
    o = np.zeros(ev.height)
    for s in DS:
        m = make_model(seed=s, **OVER)
        m.fit(Xtr, tr["y"].to_numpy().astype(float) - np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None), categorical_feature=cp)
        o += np.asarray(m.predict(Xev), float)
    return np.clip(np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None) + o / len(DS), 0, None)


def hat_avg(tr, ev):
    Xtr, cp = pdf(tr, NUM_NEW, CAT2)
    Xev, _ = pdf(ev, NUM_NEW, CAT2)
    o = np.zeros(ev.height)
    for s in HS:
        m = make_model(seed=s, **OVER)
        m.fit(Xtr, tr["y"].to_numpy().astype(float) - tr["e20_pred"].to_numpy().astype(float), categorical_feature=cp)
        o += np.asarray(m.predict(Xev), float)
    return o / len(HS)


def main():
    log("E70 load")
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

    res = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                tr = frames["janjul"].filter(pl.col("_m") == a)
                ev = frames["janjul"].filter(pl.col("_m") == b)
                rec, hat = rec_avg(tr, ev), hat_avg(tr, ev)
                g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
                parts.append({"y": ev["y"].to_numpy().astype(float), "g": g, "hat": hat,
                              "ids": ev["MVT_ID_mvt"].to_numpy(), "e20": ev["e20_pred"].to_numpy().astype(float)})
            res[split] = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "g", "hat", "ids", "e20")}
        else:
            tr, ev = frames["janjul"], frames["dec"]
            rec, hat = rec_avg(tr, ev), hat_avg(tr, ev)
            g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
            res[split] = {"y": ev["y"].to_numpy().astype(float), "g": g, "hat": hat,
                          "ids": ev["MVT_ID_mvt"].to_numpy(), "e20": ev["e20_pred"].to_numpy().astype(float)}
        log(f"  {split} g={rmse(res[split]['y'],res[split]['g']):.3f}")

    payload = {}
    for split in ("janjul", "dec"):
        o = res[split]
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        payload[split] = {}
        out_oof = pl.DataFrame({"MVT_ID_mvt": o["ids"], "y": o["y"], "g": o["g"], "hat": o["hat"]})
        for lam in (0.25, 0.30, 0.35, 0.40):
            p = o["g"] + lam * np.maximum(o["hat"], 0)
            pf = e20f.copy(); pf[idx] = p
            e2 = (o["y"] - p) ** 2
            k1 = max(1, int(0.01 * len(e2)))
            payload[split][f"lam{lam}"] = {
                "matched": float(rmse(o["y"], p)), "overall": float(rmse(yf, pf)),
                "top1": float(np.sort(e2)[::-1][:k1].sum() / e2.sum()),
            }
            log(f"  [{split}] lam={lam:.2f} matched {payload[split][f'lam{lam}']['matched']:.3f} overall {payload[split][f'lam{lam}']['overall']:.3f}")
        out_oof.write_parquet(RES / f"oof_v19_{split}.parquet")

    (RES / "E70_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E70")


if __name__ == "__main__":
    main()
