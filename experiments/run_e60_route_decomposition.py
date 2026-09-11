"""E60 — physical route decomposition (runway-geometry subset).

OurAirports (public domain) runway geometry acquired. Stand->runway distance is
NOT reproducible (no stand coordinates; E37/E59). Test the obtainable physical
features (runway length/heading/count) in the v13 Δ harness, cross-month OOF.
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
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E60"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
GEO = ["rwy_len_m", "rwy_heading", "n_runways"]
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt",
       "hour", "dow", "month", "e20_pred"] + GEO
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def main():
    # load OurAirports runway geometry
    rw = pl.read_csv(Path("/var/folders/r8/ycgrjy_d7ms114916zfsnszr0000gn/T/kilo/runways.csv"), ignore_errors=True,
                     schema_overrides={"length_ft": pl.Float64, "le_heading_degT": pl.Float64, "he_heading_degT": pl.Float64})
    rw = rw.with_columns((pl.col("length_ft") * 0.3048).alias("rwy_len_m"))
    nrun = rw.group_by("airport_ident").agg(pl.len().alias("n_runways"))
    # map RUNWAY_mvt -> OurAirports le_ident/he_ident via normalization (strip L/C suffixes for match flexibility)
    heads = rw.select(["airport_ident", "le_ident", "rwy_len_m", "le_heading_degT"]).rename(
        {"airport_ident": "airport", "le_ident": "RUNWAY_mvt", "le_heading_degT": "rwy_heading"})
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("_fp")])
    dep = dep.join(heads, on=["airport", "RUNWAY_mvt"], how="left").join(nrun.rename({"airport_ident": "airport"}), on="airport", how="left")
    dep = dep.with_columns(pl.col("rwy_len_m").fill_null(pl.col("rwy_len_m").mean()).fill_null(0.0),
                           pl.col("rwy_heading").fill_null(pl.col("rwy_heading").mean()).fill_null(0.0),
                           pl.col("n_runways").fill_null(0).cast(pl.Float64))
    dep = dep.with_columns(pl.col("AOBT_3_flt").clip(None, None), pl.col("mvt_aobt").clip(0, None))
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = fr["janjul" if split == "janjul" else "dec"]
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                t = fr["janjul"].filter(pl.col("_m") == a); ev = fr["janjul"].filter(pl.col("_m") == b)
                y0 = t["y"].to_numpy().astype(float); P0 = t["mvt_aobt"].to_numpy().astype(float)
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
                m.fit(pdf(t), y0 - P0, categorical_feature=CAT)
                yv = ev["y"].to_numpy().astype(float); Pv = ev["mvt_aobt"].to_numpy().astype(float)
                rec = np.clip(Pv + np.asarray(m.predict(pdf(ev)), float), 0, None)
                parts.append({"y": yv, "rec": rec})
            y = np.concatenate([p["y"] for p in parts]); rec = np.concatenate([p["rec"] for p in parts])
        else:
            t = fr["janjul"]; ev = fr["dec"]
            y0 = t["y"].to_numpy().astype(float); P0 = t["mvt_aobt"].to_numpy().astype(float)
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(pdf(t), y0 - P0, categorical_feature=CAT)
            y = ev["y"].to_numpy().astype(float); Pv = ev["mvt_aobt"].to_numpy().astype(float)
            rec = np.clip(Pv + np.asarray(m.predict(pdf(ev)), float), 0, None)
        payload[split] = {"rec_rmse": float(rmse(y, rec)), "top1": top1(y, rec), "n": int(len(y))}
        log(f"  [{split}] geometry-featured Δ reconstruction matched {payload[split]['rec_rmse']:.2f} top1 {payload[split]['top1']:.3e} (v13 grec ~238)")
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
