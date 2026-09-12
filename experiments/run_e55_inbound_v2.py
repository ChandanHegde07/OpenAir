"""E55 — 2nd ARR taxi-in, inbound type, same-runway ARR. Beat v17 236.80 / 210.29."""
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

RES = HERE / "results" / "E55"
RES.mkdir(parents=True, exist_ok=True)
V17 = (236.80, 210.29)
NUM_V17 = NUM + ["arr_taxiin", "since_arr", "arr_delay", "since_dep", "arr_taxiin_aobt", "since_arr_aobt"]
NUM_NEW = NUM_V17 + ["arr_taxiin_2", "rwy_arr_taxiin"]
CAT2 = CAT + ["arr_type"]


def pdf(df, cols, cats=CAT):
    p = df.select([c for c in cols if c in df.columns] + [c for c in cats if c in df.columns]).to_pandas()
    for c in cats:
        if c in p.columns:
            p[c] = p[c].fillna("NA").astype("category")
    return p


def load_arr():
    arr = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            [
                pl.col("ADES_mvt").alias("airport"),
                "STAND_mvt",
                "RUNWAY_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("SCHED_TIME_UTC_mvt").alias("sibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
                pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type"),
            ]
        )
        .drop_nulls(["airport", "STAND_mvt", "aibt", "arr_taxiin"])
        .collect()
        .with_columns((pl.col("aibt") - pl.col("sibt")).dt.total_seconds().alias("arr_delay"))
        .sort(["airport", "STAND_mvt", "aibt"])
        .with_columns(pl.col("arr_taxiin").shift(1).over(["airport", "STAND_mvt"]).alias("arr_taxiin_2"))
    )
    return arr


def attach(dep, arr, dep_all):
    d = dep.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    a = arr.sort(["airport", "STAND_mvt", "aibt"])
    j = d.join_asof(a, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "STAND_mvt"], strategy="backward")
    j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("aibt")).dt.total_seconds().alias("since_arr"))
    # AOBT asof
    a2 = a.rename({"arr_taxiin": "arr_taxiin_aobt", "aibt": "aibt2"}).select(
        ["airport", "STAND_mvt", "aibt2", "arr_taxiin_aobt"]
    )
    j = j.sort(["airport", "STAND_mvt", "AOBT_3_flt"]).join_asof(
        a2, left_on="AOBT_3_flt", right_on="aibt2", by=["airport", "STAND_mvt"], strategy="backward"
    )
    j = j.with_columns((pl.col("AOBT_3_flt") - pl.col("aibt2")).dt.total_seconds().alias("since_arr_aobt"))
    prev = dep_all.select(["airport", "STAND_mvt", pl.col("MVT_TIME_UTC_mvt").alias("prev_mvt")]).sort(
        ["airport", "STAND_mvt", "prev_mvt"]
    )
    j = j.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"]).join_asof(
        prev, left_on="MVT_TIME_UTC_mvt", right_on="prev_mvt", by=["airport", "STAND_mvt"], strategy="backward"
    )
    j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("prev_mvt")).dt.total_seconds().alias("since_dep"))
    # same-runway last ARR
    ar = arr.sort(["airport", "RUNWAY_mvt", "aibt"]).select(
        ["airport", "RUNWAY_mvt", "aibt", pl.col("arr_taxiin").alias("rwy_arr_taxiin")]
    )
    j = j.sort(["airport", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]).join_asof(
        ar, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "RUNWAY_mvt"], strategy="backward", suffix="_rwy"
    )
    win = (pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600)
    w2 = (pl.col("since_arr_aobt") > 0) & (pl.col("since_arr_aobt") < 8 * 3600)
    wd = (pl.col("since_dep") > 1) & (pl.col("since_dep") < 12 * 3600)
    return j.with_columns(
        pl.when(win).then(pl.col("arr_taxiin")).otherwise(None).alias("arr_taxiin"),
        pl.when(win).then(pl.col("since_arr")).otherwise(None).alias("since_arr"),
        pl.when(win).then(pl.col("arr_delay")).otherwise(None).alias("arr_delay"),
        pl.when(win).then(pl.col("arr_taxiin_2")).otherwise(None).alias("arr_taxiin_2"),
        pl.when(win).then(pl.col("arr_type")).otherwise(None).alias("arr_type"),
        pl.when(w2).then(pl.col("arr_taxiin_aobt")).otherwise(None).alias("arr_taxiin_aobt"),
        pl.when(w2).then(pl.col("since_arr_aobt")).otherwise(None).alias("since_arr_aobt"),
        pl.when(wd).then(pl.col("since_dep")).otherwise(None).alias("since_dep"),
    )


def pair(tr, ev, cols, cats, lam_hat=0.25):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    kw = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=1)
    cats_p = [c for c in cats if c in tr.columns]
    dmod = lgb.LGBMRegressor(**kw)
    dmod.fit(pdf(tr, cols, cats), ytr - Ptr, categorical_feature=cats_p)
    rec = np.clip(Pev + np.asarray(dmod.predict(pdf(ev, cols, cats)), float), 0, None)
    hmod = lgb.LGBMRegressor(**{**kw, "random_state": 7})
    hmod.fit(pdf(tr, cols, cats), ytr - etr, categorical_feature=cats_p)
    hat = np.asarray(hmod.predict(pdf(ev, cols, cats)), float)
    p90 = float(np.quantile(etr, 0.90))
    g = grec(eev, rec, 0.5, p90, 200)
    return dict(y=yev, e=eev, rec=rec, hat=hat, g=g, p90=p90, ids=ev["MVT_ID_mvt"].to_numpy())


def main():
    log("E55 load")
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
        log(f"  {split} arr2 {float(f['arr_taxiin_2'].is_not_null().mean()):.3f} rwy {float(f['rwy_arr_taxiin'].is_not_null().mean()):.3f}")

    specs = {
        "new": (NUM_NEW, CAT2),
    }
    payload = {}
    for name, (cols, cats) in specs.items():
        log(f" spec {name}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            parts.append(pair(frames["janjul"].filter(pl.col("_m") == a), frames["janjul"].filter(pl.col("_m") == b), cols, cats))
        ojj = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "e", "rec", "hat", "g", "ids")}
        ojj["p90"] = float(np.mean([p["p90"] for p in parts]))
        odc = pair(frames["janjul"], frames["dec"], cols, cats)
        payload[name] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            y, e = o["y"], o["e"]
            cands = {}
            for lam in (0.22, 0.25, 0.28):
                cands[f"h{lam}"] = o["g"] + lam * o["hat"]
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
            payload[name][split] = out
            for n, c in sorted(out.items(), key=lambda kv: kv[1]["matched"]):
                log(f"  [{name}/{split}] {n:8s} {c['matched']:.2f} ov {c['overall']:.2f}")

    (RES / "E55_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("WROTE E55")


if __name__ == "__main__":
    main()
