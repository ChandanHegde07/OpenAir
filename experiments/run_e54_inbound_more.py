"""E54 — richer inbound stand state on top of v16. Beat 236.86 / 210.54 both splits."""
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
from common import TRAIN_FILES, load_dep, rmse
from matched_submit import CAT, NUM, add_extra, e20_col, grec, log

RES = HERE / "results" / "E54"
RES.mkdir(parents=True, exist_ok=True)
V16 = (236.86, 210.54)
NUM_IN = NUM + ["arr_taxiin", "since_arr", "arr_delay", "since_dep", "n_arr_2h"]


def pdf(df, cols):
    p = df.select([c for c in cols if c in df.columns] + [c for c in CAT if c in df.columns]).to_pandas()
    for c in CAT:
        if c in p.columns:
            p[c] = p[c].fillna("NA").astype("category")
    return p


def load_arr():
    return (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            [
                pl.col("ADES_mvt").alias("airport"),
                "STAND_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("MVT_TIME_UTC_mvt").alias("aldt"),
                pl.col("SCHED_TIME_UTC_mvt").alias("sibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
            ]
        )
        .drop_nulls(["airport", "STAND_mvt", "aibt", "arr_taxiin"])
        .collect()
        .with_columns((pl.col("aibt") - pl.col("sibt")).dt.total_seconds().alias("arr_delay"))
        .sort(["airport", "STAND_mvt", "aibt"])
    )


def attach(dep, arr, dep_all):
    d = dep.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    a = arr.sort(["airport", "STAND_mvt", "aibt"])
    j = d.join_asof(a, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "STAND_mvt"], strategy="backward")
    j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("aibt")).dt.total_seconds().alias("since_arr"))
    # AOBT-asof: last ARR inblock before this off-block
    if "AOBT_3_flt" in j.columns:
        d2 = j.sort(["airport", "STAND_mvt", "AOBT_3_flt"])
        j2 = d2.join_asof(
            a.rename({"arr_taxiin": "arr_taxiin_aobt", "arr_delay": "arr_delay_aobt", "aibt": "aibt2"}),
            left_on="AOBT_3_flt",
            right_on="aibt2",
            by=["airport", "STAND_mvt"],
            strategy="backward",
        )
        j = j2.with_columns((pl.col("AOBT_3_flt") - pl.col("aibt2")).dt.total_seconds().alias("since_arr_aobt"))
    else:
        j = j.with_columns(pl.lit(None).alias("arr_taxiin_aobt"), pl.lit(None).alias("since_arr_aobt"), pl.lit(None).alias("arr_delay_aobt"))
    # previous DEP takeoff at stand
    prev = dep_all.select(["airport", "STAND_mvt", pl.col("MVT_TIME_UTC_mvt").alias("prev_mvt")]).sort(
        ["airport", "STAND_mvt", "prev_mvt"]
    )
    j = j.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    j = j.join_asof(
        prev, left_on="MVT_TIME_UTC_mvt", right_on="prev_mvt", by=["airport", "STAND_mvt"], strategy="backward"
    )
    j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("prev_mvt")).dt.total_seconds().alias("since_dep"))
    j = j.with_columns(pl.lit(None).cast(pl.Float64).alias("n_arr_2h"))
    # window 8h on mvt-asof taxi-in
    j = j.with_columns(
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600)).then(pl.col("arr_taxiin")).otherwise(None).alias("arr_taxiin"),
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600)).then(pl.col("since_arr")).otherwise(None).alias("since_arr"),
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600)).then(pl.col("arr_delay")).otherwise(None).alias("arr_delay"),
        pl.when((pl.col("since_dep") > 1) & (pl.col("since_dep") < 12 * 3600)).then(pl.col("since_dep")).otherwise(None).alias("since_dep"),
    )
    if "since_arr_aobt" in j.columns:
        j = j.with_columns(
            pl.when((pl.col("since_arr_aobt") > 0) & (pl.col("since_arr_aobt") < 8 * 3600))
            .then(pl.col("arr_taxiin_aobt"))
            .otherwise(None)
            .alias("arr_taxiin_aobt"),
            pl.when((pl.col("since_arr_aobt") > 0) & (pl.col("since_arr_aobt") < 8 * 3600))
            .then(pl.col("since_arr_aobt"))
            .otherwise(None)
            .alias("since_arr_aobt"),
        )
    return j


def pair(tr, ev, cols, lirf_lam=0.5):
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
    ap = ev["airport"].to_numpy()
    g2 = g.copy()
    m = ap == "LIRF"
    g2[m] = grec(eev[m], rec[m], lirf_lam, p90, 200)
    return dict(y=yev, e=eev, v16=g + 0.25 * hat, v_lirf=g2 + 0.25 * hat, ids=ev["MVT_ID_mvt"].to_numpy())


def main():
    log("E54 load")
    dep = add_extra(load_dep())
    arr = load_arr()
    dep_all = dep.select(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        log(f"  attach {split}...")
        f = attach(f, arr, dep_all)
        frames[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
        log(f"  {split} arr_taxiin {float(f['arr_taxiin'].is_not_null().mean()):.3f}")

    specs = {
        "v16": NUM + ["arr_taxiin", "since_arr"],
        "plus": NUM + ["arr_taxiin", "since_arr", "arr_delay", "since_dep"],
        "aobt": NUM + ["arr_taxiin", "since_arr", "arr_taxiin_aobt", "since_arr_aobt"],
        "all": NUM + ["arr_taxiin", "since_arr", "arr_delay", "since_dep", "arr_taxiin_aobt", "since_arr_aobt"],
    }
    payload = {}
    for name, cols in specs.items():
        log(f" spec {name}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            parts.append(pair(frames["janjul"].filter(pl.col("_m") == a), frames["janjul"].filter(pl.col("_m") == b), cols, 0.6))
        ojj = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "e", "v16", "v_lirf", "ids")}
        odc = pair(frames["janjul"], frames["dec"], cols, 0.6)
        payload[name] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            y = o["y"]
            oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
            e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
            yf = oof["y"].to_numpy().astype(float)
            pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
            idx = np.array([pos[int(i)] for i in o["ids"]])
            out = {}
            for n, p in (("v16", o["v16"]), ("lirf60", o["v_lirf"])):
                pf = e20f.copy()
                pf[idx] = p
                out[n] = {"matched": float(rmse(y, p)), "overall": float(rmse(yf, pf))}
            payload[name][split] = out
            for n, c in out.items():
                log(f"  [{name}/{split}] {n:7s} {c['matched']:.2f} ov {c['overall']:.2f}")

    (RES / "E54_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E54")


if __name__ == "__main__":
    main()
