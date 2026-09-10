"""E34c — ensemble G models (LGB+CatBoost+XGB) for the LIRF gate delay."""
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
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402

RES = HERE / "results" / "E34"
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def catdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype(str)
    return p


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr).filter(pl.col("airport") == "LIRF")
    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        Dt = tr["mvt_sched"].to_numpy().astype(float); yt = tr["y"].to_numpy().astype(float)
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        target = Dt - yt
        # LGB
        mL = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, num_leaves=31, min_child_samples=15,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
        mL.fit(pdf(tr), target, categorical_feature=CAT)
        gL = np.asarray(mL.predict(pdf(va)), float)
        # XGB (numeric-encoded cats via category codes)
        pxt = tr.select(NUM + CAT).to_pandas()
        pxv = va.select(NUM + CAT).to_pandas()
        for c in CAT:
            pxt[c] = pxt[c].astype("category").cat.codes.astype(float)
            pxv[c] = pxv[c].astype("category").cat.codes.astype(float)
        mX = xgb.XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, tree_method="hist")
        mX.fit(pxt, target)
        gX = np.asarray(mX.predict(pxv), float)
        # CatBoost
        mC = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, loss_function="RMSE",
                               random_seed=SEED, verbose=False, thread_count=-1)
        mC.fit(catdf(tr), target, cat_features=CAT)
        gC = np.asarray(mC.predict(catdf(va)), float)
        ens = (gL + gX + gC) / 3.0
        grid = {}
        for al in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for tag, gg in [("L", gL), ("X", gX), ("C", gC), ("ENS", ens)]:
                p = np.maximum(D - al * gg, 0.0)
                grid[f"{tag}_{al}"] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        out[split] = {"n": int(va.height), "grid": grid}
        best = min(grid, key=lambda k: grid[k]["sse"])
        log(f"  [{split}] best {best} rmse {grid[best]['rmse']:.0f} | ENS_1.0 {grid['ENS_1.0']['rmse']:.0f} | C_1.0 {grid['C_1.0']['rmse']:.0f} | X_1.0 {grid['X_1.0']['rmse']:.0f}")
    (RES / "E34c_gens.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E34c_gens.json"))


if __name__ == "__main__":
    main()
