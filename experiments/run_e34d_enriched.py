"""E34d — enriched LIRF gate-delay features + deeper CatBoost to cross LB 280."""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402

RES = HERE / "results" / "E34"
SEED = 1
EXTRA = ["d_log", "d_rank", "d_x_arr15", "d_x_dep15", "arr_minus_dep_15", "ev_rate_15", "hour_sin", "hour_cos"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def enrich(df, dref):
    d = np.maximum(df["mvt_sched"].to_numpy().astype(float), 0.0)
    a = df["sarr_15"].to_numpy().astype(float); p = df["sdep_15"].to_numpy().astype(float)
    h = df["hour"].to_numpy().astype(float)
    return df.with_columns([
        pl.Series("d_log", np.log1p(d)), pl.Series("d_rank", np.searchsorted(np.sort(dref), d, "left") / dref.size),
        pl.Series("d_x_arr15", d * a), pl.Series("d_x_dep15", d * p),
        pl.Series("arr_minus_dep_15", a - p), pl.Series("ev_rate_15", (a + p) / 15.0),
        pl.Series("hour_sin", np.sin(2 * np.pi * h / 24)), pl.Series("hour_cos", np.cos(2 * np.pi * h / 24)),
    ])


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr).filter(pl.col("airport") == "LIRF")
    feats = NUM + EXTRA
    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        tr = enrich(tr, tr["mvt_sched"].to_numpy().astype(float)); va = enrich(va, tr["mvt_sched"].to_numpy().astype(float))
        Dt = tr["mvt_sched"].to_numpy().astype(float); yt = tr["y"].to_numpy().astype(float)
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        def cdf(df):
            x = df.select(feats + CAT).to_pandas()
            for c in CAT:
                x[c] = x[c].fillna("NA").astype(str)
            return x
        mC = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, loss_function="RMSE",
                               random_seed=SEED, verbose=False, thread_count=-1, od_type="Iter", od_wait=80)
        mC.fit(cdf(tr), Dt - yt, cat_features=CAT)
        g = np.asarray(mC.predict(cdf(va)), float)
        grid = {}
        for al in [0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
            p = np.maximum(D - al * g, 0.0)
            grid[al] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        out[split] = {"n": int(va.height), "grid": grid}
        b = min(grid, key=lambda a: grid[a]["sse"])
        log(f"  [{split}] best alpha {b} rmse {grid[b]['rmse']:.0f} | a1.0 {grid[1.0]['rmse']:.0f}")
    (RES / "E34d_enriched.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E34d_enriched.json"))


if __name__ == "__main__":
    main()
