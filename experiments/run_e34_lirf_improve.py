"""E34 — improve the v7 latent gate-delay model (LIRF unmatched).

v7 = D - alpha*G_hat with global alpha=0.6 (E20's LIRF prediction == D).
Candidates: richer G features (+D percentile/interactions), CatBoost G,
taxi-fraction Q=T/D, gate-fraction R=G/D, and conditional alpha(d) calibrated
on a train-internal temporal holdout. Evaluate LIRF-unmatched SSE on Jan+Jul
and Dec; map to overall/LB.
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
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse, save_result  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402

RES = HERE / "results" / "E34"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
EXTRA = ["d_log", "d_rank", "d_x_arr15", "d_x_dep15", "d_x_rdep10", "arr_minus_dep_15", "ev_rate_15"]
FEATS = NUM + EXTRA
DBINS = [0, 900, 1800, 3600, 7200, 10800, 10**9]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def enrich(df: pl.DataFrame, d_ref: np.ndarray | None):
    d = df["mvt_sched"].to_numpy().astype(float)
    d = np.maximum(d, 0.0)
    arr10 = df["sarr_15"].to_numpy().astype(float)
    dep10 = df["sdep_15"].to_numpy().astype(float)
    rdep10 = df["srdep_10"].to_numpy().astype(float)
    if d_ref is None:
        d_ref = d
    rank = np.searchsorted(np.sort(d_ref), d, side="left") / max(d_ref.size, 1)
    return df.with_columns([
        pl.Series("d_log", np.log1p(d)),
        pl.Series("d_rank", rank),
        pl.Series("d_x_arr15", d * arr10),
        pl.Series("d_x_dep15", d * dep10),
        pl.Series("d_x_rdep10", d * rdep10),
        pl.Series("arr_minus_dep_15", arr10 - dep10),
        pl.Series("ev_rate_15", (dep10 + arr10) / 15.0),
    ])


def pdf(df):
    p = df.select(FEATS + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def fit_lgb(tr, target):
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, num_leaves=31, min_child_samples=15,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def fit_cat(tr, target):
    p = tr.select(FEATS + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype(str)
    m = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3.0, loss_function="RMSE",
                          random_seed=SEED, verbose=False, thread_count=-1)
    m.fit(p, target, cat_features=CAT)
    return m


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr).filter(pl.col("airport") == "LIRF")
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & pl.col("unmatched"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        D = va["mvt_sched"].to_numpy().astype(float); y = va["y"].to_numpy().astype(float)
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        # enrich with train-derived d_ref
        tr = enrich(tr, None); va = enrich(va, tr["mvt_sched"].to_numpy().astype(float))
        # temporal inner split for calibration
        from run_e12_e9 import time_es_split
        tr_fit, tr_es = time_es_split(tr)
        out = {"n": int(va.height), "e20_rmse": float(rmse(y, e20)), "e20_sse": float(np.sum((y - e20) ** 2))}
        # G model (LGB) on tr_fit
        Df = tr_fit["mvt_sched"].to_numpy().astype(float); yf = tr_fit["y"].to_numpy().astype(float)
        De = tr_es["mvt_sched"].to_numpy().astype(float); ye = tr_es["y"].to_numpy().astype(float)
        mG = fit_lgb(tr_fit, Df - yf)
        g_tr = np.asarray(mG.predict(pdf(tr_fit)), float)   # in-sample (for alpha fit use es)
        g_es = np.asarray(mG.predict(pdf(tr_es)), float)
        g_va = np.asarray(mG.predict(pdf(va)), float)
        t_dec = np.maximum(D - g_va, 0.0)
        # global alpha
        grid = {}
        for al in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
            p = al * t_dec + (1 - al) * D   # == D - al*G
            grid[al] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        beta = min(grid, key=lambda a: grid[a]["sse"])
        out["G_global"] = {"best_alpha": beta, **grid[beta], "grid": grid}
        # conditional alpha by D bin, calibrated on tr_es
        Dbin_e = np.digitize(De, DBINS) - 1
        Dbin_v = np.digitize(D, DBINS) - 1
        alphas = {}
        for b in np.unique(Dbin_e):
            m = Dbin_e == b
            if m.sum() < 30:
                alphas[int(b)] = beta
                continue
            best = None
            for al in np.arange(0.1, 1.01, 0.05):
                p = D - al * g_es[m] if False else (al * (De[m] - g_es[m]) + (1 - al) * De[m])
                s = float(np.sum((ye[m] - p) ** 2))
                if best is None or s < best[1]:
                    best = (float(al), s)
            alphas[int(b)] = best[0]
        a_v = np.array([alphas.get(int(b), beta) for b in Dbin_v])
        p_cond = np.maximum(a_v * (D - g_va) + (1 - a_v) * D, 0.0)
        out["G_cond_alpha"] = {"rmse": float(rmse(y, p_cond)), "sse": float(np.sum((y - p_cond) ** 2)),
                               "alphas": alphas}
        # CatBoost G
        mC = fit_cat(tr_fit, Df - yf)
        pc = tr.select(FEATS + CAT).to_pandas()
        for c in CAT:
            pc[c] = pc[c].fillna("NA").astype(str)
        # predict only va rows
        pv = va.select(FEATS + CAT).to_pandas()
        for c in CAT:
            pv[c] = pv[c].fillna("NA").astype(str)
        gC = np.asarray(mC.predict(pv), float)
        cgrid = {}
        for al in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
            pp = np.maximum(D - al * gC, 0.0)
            cgrid[al] = {"rmse": float(rmse(y, pp)), "sse": float(np.sum((y - pp) ** 2))}
        bestC = min(cgrid, key=lambda a: cgrid[a]["sse"])
        pC = np.maximum(D - bestC * gC, 0.0)
        out["G_catboost"] = {"best_alpha": float(bestC), "rmse": float(rmse(y, pC)),
                             "sse": float(np.sum((y - pC) ** 2)), "grid": cgrid}
        bgrid = {}
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            gb = w * g_va + (1 - w) * gC
            for al in [0.6, 0.7, 0.8, 0.9, 1.0]:
                pp = np.maximum(D - al * gb, 0.0)
                bgrid[(w, al)] = {"rmse": float(rmse(y, pp)), "sse": float(np.sum((y - pp) ** 2))}
        bk = min(bgrid, key=lambda k: bgrid[k]["sse"])
        out["G_blend"] = {"w_lgb": bk[0], "alpha": bk[1], **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in bgrid[bk].items()},
                          "grid": {f"{k[0]}_{k[1]}": v for k, v in bgrid.items()}}
        # Q = T/D model
        mQ = fit_lgb(tr_fit, np.clip(yf / np.maximum(Df, 60), 0, 1))
        q = np.clip(np.asarray(mQ.predict(pdf(va)), float), 0, 1)
        pQ = np.maximum(D * q, 0.0)
        out["Q_model"] = {"rmse": float(rmse(y, pQ)), "sse": float(np.sum((y - pQ) ** 2))}
        payload[split] = out
        log(f"  [{split}] E20 {out['e20_rmse']:.0f} | G global a={beta} {out['G_global']['rmse']:.0f} | "
            f"G cond {out['G_cond_alpha']['rmse']:.0f} | G cat a={bestC:.1f} {out['G_catboost']['rmse']:.0f} | Q {out['Q_model']['rmse']:.0f}")
    (RES / "E34_lirf.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    save_result("E34", payload)
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
