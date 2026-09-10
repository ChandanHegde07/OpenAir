"""E43 STEP 2 — D−G decomposition for LFPG unmatched (reuse E33/E34 machinery).

D = MVT - SCHED (known when BLOCK missing); G = BLOCK - SCHED predicted with the
E34 CatBoost feature family; T_hat = D - G_hat; blend with E20 per-airport.
Evaluate Jan+Jul and Dec; LFPG unmatched only.
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
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402
from run_e34d_enriched import EXTRA, enrich  # noqa: E402

RES = HERE / "results" / "E43"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FEATS = NUM + EXTRA
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def cdf(df):
    x = df.select(FEATS + CAT).to_pandas()
    for c in CAT:
        x[c] = x[c].fillna("NA").astype(str)
    return x


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr).filter(pl.col("airport") == "LFPG")
    dep = enrich(dep, dep["mvt_sched"].to_numpy().astype(float))
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        out = {"n": int(va.height), "e20_rmse": float(rmse(y, e20)), "e20_sse": float(np.sum((y - e20) ** 2))}
        cands = {}
        for tag, trn in [("all", tr), ("unm", tr.filter(pl.col("unmatched")))]:
            if trn.height < 200:
                continue
            Dt = trn["mvt_sched"].to_numpy().astype(float); yt = trn["y"].to_numpy().astype(float)
            m = CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=7, l2_leaf_reg=4.0, loss_function="RMSE",
                                  random_seed=SEED, verbose=False, thread_count=-1, od_type="Iter", od_wait=80)
            m.fit(cdf(trn), Dt - yt, cat_features=CAT)
            g = np.asarray(m.predict(cdf(va)), float)
            cands[tag] = {"n_train": int(trn.height), "g": g}
        # alpha/blend grid per candidate
        best = None
        for tag, c in cands.items():
            t_dec = np.maximum(D - c["g"], 0.0)
            for al in ALPHAS:
                p = al * t_dec + (1 - al) * e20
                s = float(np.sum((y - p) ** 2))
                if best is None or s < best[0]:
                    best = (s, tag, al, float(rmse(y, p)))
        out["best"] = {"tag": best[1], "alpha": best[2], "rmse": best[3], "sse": best[0],
                       "vanilla_rmse": float(rmse(y, np.maximum(D - cands[best[1]]["g"], 0.0)))}
        out["candidates"] = {t: c["n_train"] for t, c in cands.items()}
        payload[split] = out
        log(f"  [{split}] LFPG unmatched n={out['n']} E20 {out['e20_rmse']:.1f} (sse {out['e20_sse']:.3e}) -> best {best[1]} a={best[2]} {best[3]:.1f}")
    (RES / "E43_lfpg.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    # combined overall estimate
    for split in payload:
        pass
    log("WROTE " + str(RES / "E43_lfpg.json"))


if __name__ == "__main__":
    main()
