"""E67 — matched Δ-model improvements: clock-disagreement features, unused
categoricals, capacity, and per-airport Δ experts.

Evaluated with the same cross-month OOF protocol as E54/E55 (Jan<->Jul, Jan+Jul->Dec).
Baseline v18 = 236.59 / 210.22.
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

RES = HERE / "results" / "E67"
RES.mkdir(parents=True, exist_ok=True)

BASE_COLS = list(NUM_NEW)
BASE_CATS = list(CAT2)
NEW_CATS = ["WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ADES_FILED_flt"]
NEW_NUM = [
    "mvt_min4", "mvt_max4", "mvt_med4", "mvt_eobt", "aobt_is_latest", "aobt_is_earliest",
    "clk_disp_ae", "clk_disp_ai", "clk_disp_al", "min_minus_aobt",
]

CLOCKS = ["mvt_aobt", "mvt_eobt", "mvt_iobt", "mvt_lobt"]


def add_clock_feats(df: pl.DataFrame) -> pl.DataFrame:
    M = np.column_stack([df[c].to_numpy().astype(float) for c in CLOCKS])
    with np.errstate(all="ignore"):
        mn = np.nanmin(M, axis=1)
        mx = np.nanmax(M, axis=1)
        md = np.nanmedian(M, axis=1)
    a = df["mvt_aobt"].to_numpy().astype(float)
    return df.with_columns(
        pl.Series("mvt_min4", mn), pl.Series("mvt_max4", mx), pl.Series("mvt_med4", md),
        pl.Series("aobt_is_latest", a >= mx - 1e-9), pl.Series("aobt_is_earliest", a <= mn + 1e-9),
        pl.Series("clk_disp_ae", a - df["mvt_eobt"].to_numpy().astype(float)),
        pl.Series("clk_disp_ai", a - df["mvt_iobt"].to_numpy().astype(float)),
        pl.Series("clk_disp_al", a - df["mvt_lobt"].to_numpy().astype(float)),
        pl.Series("min_minus_aobt", mn - a),
    )


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


def fit_pair(tr, ev, cols, cats, d_over=None, h_over=None):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Xtr, cp = pdf(tr, cols, cats)
    Xev, _ = pdf(ev, cols, cats)
    dmod = make_model(**(d_over or {}))
    dmod.fit(Xtr, ytr - Ptr, categorical_feature=cp)
    rec = np.clip(Pev + np.asarray(dmod.predict(Xev), float), 0, None)
    hmod = make_model(seed=7, **(h_over or {}))
    hmod.fit(Xtr, ytr - etr, categorical_feature=cp)
    hat = np.asarray(hmod.predict(Xev), float)
    return rec, hat


def per_airport_rec(tr, ev, cols, cats, d_over=None):
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    rec = Pev.copy()
    for ap in np.unique(ev["airport"].to_numpy()):
        t = tr.filter(pl.col("airport") == ap)
        e = ev.filter(pl.col("airport") == ap)
        if t.height < 500 or e.height == 0:
            continue
        Xtr, cp = pdf(t, cols, cats)
        Xev, _ = pdf(e, cols, cats)
        ytr = t["y"].to_numpy().astype(float)
        Ptr = np.clip(t["mvt_aobt"].to_numpy().astype(float), 0, None)
        m = make_model(**(d_over or {}))
        m.fit(Xtr, ytr - Ptr, categorical_feature=cp)
        rec[ev["airport"].to_numpy() == ap] = np.clip(Pev[ev["airport"].to_numpy() == ap] + np.asarray(m.predict(Xev), float), 0, None)
    return rec


def top_sse(y, p, frac):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum()), float(e2.sum())


def main():
    log("E67 load")
    dep = add_extra(load_dep())
    dep = add_clock_feats(dep)
    arr = load_arr()
    dep_all = dep.select(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        f = attach(f, arr, dep_all)
        frames[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))

    specs = {
        "base": (BASE_COLS, BASE_CATS),
        "clk": (BASE_COLS + NEW_NUM, BASE_CATS),
        "cat": (BASE_COLS, BASE_CATS + NEW_CATS),
        "clkcat": (BASE_COLS + NEW_NUM, BASE_CATS + NEW_CATS),
    }
    payload = {}
    for name, (cols, cats) in specs.items():
        log(f" spec {name}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            tr = frames["janjul"].filter(pl.col("_m") == a)
            ev = frames["janjul"].filter(pl.col("_m") == b)
            rec, hat = fit_pair(tr, ev, cols, cats)
            g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
            parts.append({"y": ev["y"].to_numpy().astype(float), "v": g + 0.28 * np.maximum(hat, 0)})
        ojj = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "v")}
        rec, hat = fit_pair(frames["janjul"], frames["dec"], cols, cats)
        g = grec(frames["dec"]["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(frames["janjul"]["e20_pred"], 0.9)), 200)
        odc = {"y": frames["dec"]["y"].to_numpy().astype(float), "v": g + 0.28 * np.maximum(hat, 0)}
        payload[name] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            s1, st = top_sse(o["y"], o["v"], 0.01)
            s5, _ = top_sse(o["y"], o["v"], 0.05)
            payload[name][split] = {"matched": float(rmse(o["y"], o["v"])), "top1": s1 / st, "top5": s5 / st}
            log(f"  [{name}/{split}] matched {rmse(o['y'], o['v']):8.3f} top1 {s1/st:.4f} top5 {s5/st:.4f}")

    # per-airport Δ on richest feature set
    for name, over in (("perap", None), ("bigcap", dict(n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40))):
        cols, cats = BASE_COLS + NEW_NUM, BASE_CATS + NEW_CATS
        log(f" spec {name}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            tr = frames["janjul"].filter(pl.col("_m") == a)
            ev = frames["janjul"].filter(pl.col("_m") == b)
            if name == "perap":
                rec = per_airport_rec(tr, ev, cols, cats)
            else:
                rec, _ = fit_pair(tr, ev, cols, cats, d_over=over)
            _, hat = fit_pair(tr, ev, cols, cats, d_over=over)
            g = grec(ev["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(tr["e20_pred"], 0.9)), 200)
            parts.append({"y": ev["y"].to_numpy().astype(float), "v": g + 0.28 * np.maximum(hat, 0)})
        ojj = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "v")}
        if name == "perap":
            rec = per_airport_rec(frames["janjul"], frames["dec"], cols, cats)
        else:
            rec, _ = fit_pair(frames["janjul"], frames["dec"], cols, cats, d_over=over)
        _, hat = fit_pair(frames["janjul"], frames["dec"], cols, cats, d_over=over)
        g = grec(frames["dec"]["e20_pred"].to_numpy().astype(float), rec, 0.5, float(np.quantile(frames["janjul"]["e20_pred"], 0.9)), 200)
        odc = {"y": frames["dec"]["y"].to_numpy().astype(float), "v": g + 0.28 * np.maximum(hat, 0)}
        payload[name] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            s1, st = top_sse(o["y"], o["v"], 0.01)
            s5, _ = top_sse(o["y"], o["v"], 0.05)
            payload[name][split] = {"matched": float(rmse(o["y"], o["v"])), "top1": s1 / st, "top5": s5 / st}
            log(f"  [{name}/{split}] matched {rmse(o['y'], o['v']):8.3f} top1 {s1/st:.4f} top5 {s5/st:.4f}")

    (RES / "E67_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E67")


if __name__ == "__main__":
    main()
