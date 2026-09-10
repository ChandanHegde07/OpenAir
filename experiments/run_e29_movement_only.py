"""E29 PHASE 1/3 — movement-only supervised model for unmatched rows (C27).

Trained on movement-only features (no NM fields), target TAXITIME, fit on
train months, evaluated on validation unmatched rows and compared to E20.
E20 untouched.
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
import xgboost as xgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, mae, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E29"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1

MOVE_NUM = ["sched_min", "mvt_min", "mvt_sched", "hour", "dow", "month", "sched_hour"]
MOVE_CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "flt_prefix", "weekday_cat"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def featurize_move(dep: pl.DataFrame) -> pl.DataFrame:
    return dep.with_columns([
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("sched_hour"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.hour() * 60 + pl.col("SCHED_TIME_UTC_mvt").dt.minute()).cast(pl.Int64).alias("sched_min"),
        (pl.col("MVT_TIME_UTC_mvt").dt.hour() * 60 + pl.col("MVT_TIME_UTC_mvt").dt.minute()).cast(pl.Int64).alias("mvt_min"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("weekday_cat"),
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"),
    ])


def fit_lgb(x, y, cats, seed=SEED, n=600):
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=63, min_child_samples=60,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(x, y, categorical_feature=cats)
    return m


def main():
    log("E29 movement-only: load...")
    dep = featurize_move(load_dep())
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        # train on ALL rows (matched+unmatched), movement-only
        def pdf(df):
            p = df.select(MOVE_NUM + MOVE_CAT).to_pandas()
            for c in MOVE_CAT:
                p[c] = p[c].astype("category")
            return p
        xtr = pdf(tr); ytr = tr["y"].to_numpy().astype(np.float64)
        xva = pdf(va)
        m = fit_lgb(xtr, ytr, MOVE_CAT)
        pred = np.asarray(m.predict(xva), dtype=np.float64)

        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy().astype(bool)
        ap = va["airport"].to_numpy()
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        e20 = (0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy())

        def sc(mask, name):
            return {"n": int(mask.sum()), "rmse_move": rmse(y[mask], pred[mask]),
                    "rmse_e20": rmse(y[mask], e20[mask]), "mae_move": mae(y[mask], pred[mask])}
        res = {
            "movement_only_all": sc(np.ones(len(y), bool), "all"),
            "movement_only_matched": sc(~um, "m"),
            "unmatched": sc(um, "u"),
            "lirf_unmatched": sc(um & (ap == "LIRF"), "lu"),
            "nonlirf_unmatched": sc(um & (ap != "LIRF"), "nlu"),
            "e20_all": {"rmse_e20": rmse(y, e20)},
        }
        log(f"  [{split}] unmatched: move {res['unmatched']['rmse_move']:.0f} vs E20 {res['unmatched']['rmse_e20']:.0f}")
        log(f"  [{split}] LIRF unmatched: move {res['lirf_unmatched']['rmse_move']:.0f} vs E20 {res['lirf_unmatched']['rmse_e20']:.0f} (n={res['lirf_unmatched']['n']})")
        log(f"  [{split}] non-LIRF unmatched: move {res['nonlirf_unmatched']['rmse_move']:.0f} vs E20 {res['nonlirf_unmatched']['rmse_e20']:.0f}")
        payload[split] = res
        if split == "janjul":
            # feature importance for LIRF behaviour
            imp = sorted(zip(MOVE_NUM + MOVE_CAT, m.booster_.feature_importance(importance_type="gain").tolist()), key=lambda t: -t[1])
            payload["lirf_feature_gain"] = imp[:15]

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (RES / "E29_movement_only.json").write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")
    save_result("E29_movement_only", conv(payload))
    log("WROTE " + str(RES / "E29_movement_only.json"))


if __name__ == "__main__":
    main()
