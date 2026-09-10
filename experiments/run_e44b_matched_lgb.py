"""E44b STEP 1 (light) — LGB matched-expert tuning + seed averaging only."""
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
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import add_causal_rolling, airport_mean_fallback, fill_with_fallback, load_dep, rmse  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS, time_es_split  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402

RES = HERE / "results" / "E44"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT_COLS).to_pandas()
    for c in CAT_COLS:
        p[c] = p[c].astype("category")
    return p[cols + CAT_COLS]


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr(); dep = add_traffic(dep, arr); dep = add_queue(dep)
    dep = add_push_disruption(dep); dep = add_rolling_quantiles(dep); dep = add_arrival_delay_state(dep, load_arr_delay())
    num = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
    variants = [
        ("base", dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=80)),
        ("deep", dict(n_estimators=600, learning_rate=0.04, num_leaves=127, min_child_samples=20)),
        ("shallow", dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40)),
        ("reg", dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=100, reg_lambda=10.0)),
    ]
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, va))
        p_ap_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal_tr = fill_with_fallback(fill_with_fallback(p_ap_t, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
        um = tr["unmatched"].to_numpy().astype(bool); ap = tr["airport"].to_numpy()
        keep = ~(um & (ap == "LIRF"))
        trm = tr.filter(pl.Series(keep)).with_columns(pl.Series("p_cal", p_cal_tr[keep]))
        tr_fit, tr_es = time_es_split(trm)
        ytr = tr_fit["y"].to_numpy().astype(float) - tr_fit["p_cal"].to_numpy()
        yes = tr_es["y"].to_numpy().astype(float) - tr_es["p_cal"].to_numpy()
        Xf, Xe, Xv = pdf(tr_fit, num), pdf(tr_es, num), pdf(va, num)
        yv = va["y"].to_numpy().astype(float); umv = va["unmatched"].to_numpy().astype(bool)
        mm = ~umv & np.isfinite(yv)
        def top(p, frac):
            sse = (yv[mm] - p[mm]) ** 2; k = max(1, int(frac * mm.sum()))
            return float(np.sort(sse)[::-1][:k].sum())
        preds = {}
        for name, kw in variants:
            m = lgb.LGBMRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1, **kw)
            m.fit(Xf, ytr, eval_set=[(Xe, yes)], callbacks=[lgb.early_stopping(40, verbose=False)], categorical_feature=CAT_COLS)
            preds[name] = p_cal + np.asarray(m.predict(Xv), float)
        # seed average of base
        seedpred = [preds["base"]]
        for s in (SEED + 1, SEED + 2):
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=80,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
            m.fit(Xf, ytr, eval_set=[(Xe, yes)], callbacks=[lgb.early_stopping(40, verbose=False)], categorical_feature=CAT_COLS)
            seedpred.append(p_cal + np.asarray(m.predict(Xv), float))
        preds["seed_avg"] = np.mean(seedpred, axis=0)
        res = {k: {"matched": float(rmse(yv[mm], p[mm])), "top5": top(p, 0.05)} for k, p in preds.items()}
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        j = va.select(["MVT_ID_mvt"]).join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * j["pred_C"].to_numpy() + 0.053 * j["pred_D"].to_numpy() + 0.491 * j["pred_E"].to_numpy()
        res["E20"] = {"matched": float(rmse(yv[mm], e20[mm])), "top5": top(e20, 0.05)}
        for tag, cand in [("nnls_E20plus", [e20]), ("nnls_new_only", [])]:
            colls = [e20[mm]] + [preds[k][mm] for k in ["deep", "shallow", "reg", "seed_avg"]]
            M = np.column_stack(colls); w, _ = nnls(M, yv[mm]); w = w / max(w.sum(), 1e-9)
            p = e20.copy(); p[mm] = M @ w
            res[tag] = {"matched": float(rmse(yv[mm], p[mm])), "top5": top(p, 0.05), "w": [float(x) for x in w]}
        payload[split] = res
        log(f"  [{split}] E20 {res['E20']['matched']:.2f} | " + " ".join(f"{k}={res[k]['matched']:.1f}" for k in [v[0] for v in variants] + ["seed_avg"]) +
            f" | nnls {res['nnls_E20plus']['matched']:.2f} w={np.round(res['nnls_E20plus']['w'],3).tolist()}")
    (RES / "E44_tuning.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
