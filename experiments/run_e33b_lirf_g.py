"""E33b — refine the LIRF gate-delay G model and pick a split-robust alpha.

G = BLOCK - SCHED = D - T. Train G on LIRF training rows (matched+unmatched vs
unmatched-only), reconstruct T_hat = D - G_hat on LIRF unmatched val, blend with
E20 by alpha. Choose alpha robust across Jan+Jul and Dec.
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
from common import AIRPORTS, TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402

RES = HERE / "results" / "E33"
SEED = 1
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT).to_pandas()
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
    dep = dep.filter(pl.col("airport") == "LIRF")
    log(f"LIRF rows {dep.height:,}")

    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        vmap = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * vmap["pred_C"].to_numpy() + 0.053 * vmap["pred_D"].to_numpy() + 0.491 * vmap["pred_E"].to_numpy()
        y = vmap["y"].to_numpy().astype(float); D = vmap["mvt_sched"].to_numpy().astype(float)
        out[split] = {"n": int(vmap.height), "e20_rmse": rmse(y, e20), "e20_sse": float(np.sum((y - e20) ** 2))}
        for tag, trn in [("all_lirf", tr), ("unm_lirf", tr.filter(pl.col("unmatched")))]:
            if trn.height < 150:
                continue
            Dt = trn["mvt_sched"].to_numpy().astype(float); yt = trn["y"].to_numpy().astype(float)
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=20,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(pdf(trn, NUM), Dt - yt, categorical_feature=CAT)
            ghat = np.asarray(m.predict(pdf(vmap, NUM)), dtype=float)
            tdec = D - ghat
            grid = {}
            for al in ALPHAS:
                p = al * tdec + (1 - al) * e20
                grid[al] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
            out[split][tag] = {"n_train": int(trn.height), "dec_rmse": float(rmse(y, tdec)),
                               "dec_sse": float(np.sum((y - tdec) ** 2)), "grid": grid}
            log(f"  [{split}] {tag}: dec standalone {rmse(y,tdec):.0f}")
            for al in ALPHAS:
                log(f"      alpha {al}: {grid[al]['rmse']:.0f}")

    # robust alpha: minimize normalized SSE across both splits relative to E20
    best = None
    for al in ALPHAS:
        tot = 0.0
        for s in out:
            tag = "all_lirf" if "all_lirf" in out[s] else "unm_lirf"
            tot += out[s][tag]["grid"][al]["sse"] / out[s]["e20_sse"]
        if best is None or tot < best[1]:
            best = (al, tot)
    out["robust_alpha"] = {"alpha": best[0], "normalized_sse_ratio": best[1]}
    log(f"ROBUST ALPHA {best[0]} (normalized ratio {best[1]:.3f})")
    (RES / "E33b_lirf_g.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E33b_lirf_g.json"))


if __name__ == "__main__":
    main()
