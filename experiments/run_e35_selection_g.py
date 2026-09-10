"""E35 — selection-aware latent gate-delay model (LIRF unmatched).

Idea: G is trained mostly on matched flights but deployed on unmatched flights.
Train a propensity classifier P(M=1|x) from prediction-time features and weight
the matched G-training rows by P(M=0|x)/P(M=1|x) (clipped), plus an
unmatched-only G model. Compare to v8 (enriched CatBoost G, alpha=1.0) on
LIRF-unmatched, both splits, with alpha robustness.
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
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402
from run_e34d_enriched import EXTRA, enrich  # noqa: E402

RES = HERE / "results" / "E35"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FEATS = NUM + EXTRA
ALPHAS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def catdf(df):
    x = df.select(FEATS + CAT).to_pandas()
    for c in CAT:
        x[c] = x[c].fillna("NA").astype(str)
    return x


def fit_cat(tr, target, w=None):
    m = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, loss_function="RMSE",
                          random_seed=SEED, verbose=False, thread_count=-1, od_type="Iter", od_wait=80)
    m.fit(catdf(tr), target, sample_weight=w, cat_features=CAT)
    return m


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr).filter(pl.col("airport") == "LIRF")
    dep = enrich(dep, dep["mvt_sched"].to_numpy().astype(float))
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        Dt = tr["mvt_sched"].to_numpy().astype(float); yt = tr["y"].to_numpy().astype(float)
        Mt = (~tr["unmatched"].to_numpy().astype(bool)).astype(int)  # 1 matched
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        # propensity P(matched|x)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        pcat = tr.select(FEATS + CAT).to_pandas()
        for c in CAT:
            pcat[c] = pcat[c].astype("category")
        clf.fit(pcat, Mt, categorical_feature=CAT)
        p1 = clf.predict_proba(pcat)[:, 1]
        p1 = np.clip(p1, 1e-3, 1 - 1e-3)
        w_raw = (1 - p1) / p1
        res = {"n": int(va.height)}
        cands = {}
        # v8: unweighted CatBoost on all LIRF
        m0 = fit_cat(tr, Dt - yt)
        g0 = np.asarray(m0.predict(catdf(va)), float)
        cands["v8_all"] = g0
        # weighted (clip 99 / 95)
        for tag, q in [("w99", 0.99), ("w95", 0.95)]:
            cap = np.quantile(w_raw, q)
            w = np.clip(w_raw, 0, cap)
            mw = fit_cat(tr, Dt - yt, w=w)
            cands[f"w_{tag}"] = np.asarray(mw.predict(catdf(va)), float)
        # unmatched-only
        tr_u = tr.filter(pl.col("unmatched"))
        if tr_u.height >= 200:
            mu = fit_cat(tr_u, tr_u["mvt_sched"].to_numpy().astype(float) - tr_u["y"].to_numpy().astype(float))
            cands["unm_only"] = np.asarray(mu.predict(catdf(va)), float)
        # scores
        out = {}
        for name, g in cands.items():
            grid = {}
            for al in ALPHAS:
                p = np.maximum(D - al * g, 0.0)
                grid[al] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
            b = min(grid, key=lambda a: grid[a]["sse"])
            out[name] = {"best_alpha": b, "rmse": grid[b]["rmse"], "grid": grid}
            log(f"  [{split}] {name}: best a={b} rmse {grid[b]['rmse']:.0f} | a1.0 {grid[1.0]['rmse']:.0f} | a0.85 {grid[0.85]['rmse']:.0f}")
        payload[split] = {"n": int(va.height), "e20_rmse": float(rmse(y, D)), "cands": out}
    (RES / "E35_selection.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    # robust pick across splits (min normalized SSE, favour configs improving both)
    ref = {"janjul": 397, "dec": 88}
    print("robust configs (min sum normalized sse):")
    names = set(payload["janjul"]["cands"]).intersection(payload["dec"]["cands"])
    rows = []
    for name in names:
        for al_s in payload["janjul"]["cands"][name]["grid"]:
            al = float(al_s)
            sj = payload["janjul"]["cands"][name]["grid"][al_s]["sse"]
            sd = payload["dec"]["cands"][name]["grid"][al_s]["sse"]
            rows.append((name, al, sj / (397 * 6033 ** 2) + sd / (88 * 2786.2 ** 2)))
    for name, al, r in sorted(rows, key=lambda t: t[2])[:8]:
        print(f"  {name} a={al} norm={r:.3f} jj_LIRF={payload['janjul']['cands'][name]['grid'][al]['rmse']:.0f} "
              f"dec_LIRF={payload['dec']['cands'][name]['grid'][al]['rmse']:.0f}")
    log("WROTE " + str(RES / "E35_selection.json"))


if __name__ == "__main__":
    main()
