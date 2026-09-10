"""E34b — apply the G-decomposition to non-LIRF unmatched rows too.

Train a G model on all unmatched training rows (all airports), reconstruct
T = D - alpha*G on non-LIRF unmatched val; evaluate SSE vs E20 on both splits.
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


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr)
    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched") & (pl.col("airport") != "LIRF"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, num_leaves=63, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
        Dt = tr["mvt_sched"].to_numpy().astype(float); yt = tr["y"].to_numpy().astype(float)
        m.fit(pdf(tr), Dt - yt, categorical_feature=CAT)
        g = np.asarray(m.predict(pdf(va)), float)
        grid = {}
        for al in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            p = np.maximum(D - al * g, 0.0)
            grid[al] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        best = min(grid, key=lambda a: grid[a]["sse"])
        out[split] = {"n": int(va.height), "e20_rmse": float(rmse(y, e20)), "e20_sse": float(np.sum((y - e20) ** 2)),
                      "best_alpha": best, **grid[best], "grid": grid}
        log(f"  [{split}] nonLIRF E20 {out[split]['e20_rmse']:.0f} | dec alpha={best} {out[split]['rmse']:.0f} (sse {out[split]['sse']:.3e} vs {out[split]['e20_sse']:.3e})")
    (RES / "E34b_nonlirf.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E34b_nonlirf.json"))


if __name__ == "__main__":
    main()
