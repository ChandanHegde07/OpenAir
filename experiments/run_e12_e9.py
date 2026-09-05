"""E12 LightGBM on frozen representation; E9 residual vs direct."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
    load_dep,
    metrics_block,
    save_result,
    split_by_months,
)
from run_e2b_e3 import attach_geometry, evaluate, geometry_tables, per_airport_ols
from run_e4_e5 import add_queue, add_traffic, load_arr

NUM_COLS = [
    "mvt_aobt",
    "aobt_eobt",
    "mvt_sched",
    "geo_mean",
    "roll10_mean_mvt_aobt",
    "dep_15m",
    "dep_rwy_15m",
    "dep_60m",
    "arr_15m",
    "queue",
    "queue_rwy",
    "hour",
    "dow",
    "month",
    "unmatched",
    "type_null",
]
CAT_COLS = [
    "airport",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "ADES_mvt",
]


def to_xy(df: pl.DataFrame):
    pdf = df.select(NUM_COLS + CAT_COLS + ["y"]).to_pandas()
    pdf["unmatched"] = pdf["unmatched"].astype(np.int8)
    pdf["type_null"] = pdf["type_null"].astype(np.int8)
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype("category")
    x = pdf[NUM_COLS + CAT_COLS]
    y = pdf["y"].to_numpy(dtype=np.float64)
    return x, y


def fit_lgb(xtr, ytr, xva_es, yva_es, seed=0):
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva_es, yva_es)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def time_es_split(df: pl.DataFrame, frac=0.15):
    d = df.sort("MVT_TIME_UTC_mvt")
    n = d.height
    k = int(n * (1 - frac))
    return d.head(k), d.tail(n - k)


def main():
    out_path = Path(__file__).resolve().parent / "results" / "E12_E9.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        print("building features...", flush=True)
        dep = add_causal_rolling(load_dep())
        dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
        arr = load_arr()
        dep = add_traffic(dep, arr)
        dep = add_queue(dep)
        print(f"ready {dep.height:,}", flush=True)
        payload = {}

        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr0, va0 = split_by_months(dep, months)
            tabs = geometry_tables(tr0)
            tr = attach_geometry(tr0, tabs, 30)
            va = attach_geometry(va0, tabs, 30)
            y = va["y"].to_numpy()
            um = va["unmatched"].to_numpy()
            ap = va["airport"].to_numpy()
            fb = airport_mean_fallback(tr, va)
            geo = va["geo_mean"].to_numpy().astype(float)
            mvt_sched = va["mvt_sched"].to_numpy().astype(float)

            print(f"\n========== {split_name} ==========", flush=True)
            fh.write(f"\n========== {split_name} ==========\n")
            block = {}

            # linear core
            p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
            p_lin = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)
            block["E3_linear"] = evaluate("E3 per-airport aobt+eobt+geo", y, p_lin, um, ap, fh)

            # E11 hybrid: LIRF unmatched -> MVT-SCHED (no clip / clip 24h)
            hybrid = p_lin.copy()
            mask = um & (ap == "LIRF") & np.isfinite(mvt_sched)
            hybrid[mask] = mvt_sched[mask]
            block["E3_plus_LIRF_mvtsched"] = evaluate(
                "E3 + LIRF unmatched=MVT-SCHED", y, hybrid, um, ap, fh
            )
            hybrid24 = p_lin.copy()
            hybrid24[mask] = np.clip(mvt_sched[mask], 0, 86400)
            block["E3_plus_LIRF_mvtsched_clip24h"] = evaluate(
                "E3 + LIRF unmatched=clip(MVT-SCHED,24h)", y, hybrid24, um, ap, fh
            )

            # LightGBM direct
            print("  fitting LGB direct...", flush=True)
            tr_fit, tr_es = time_es_split(tr)
            xtr, ytr = to_xy(tr_fit)
            xes, yes = to_xy(tr_es)
            xva, _ = to_xy(va)
            model_d = fit_lgb(xtr, ytr, xes, yes)
            pred_d = model_d.predict(xva)
            block["lgb_direct"] = evaluate("LGB direct y", y, pred_d, um, ap, fh)
            block["lgb_direct_trees"] = int(model_d.best_iteration_ or model_d.n_estimators)
            # importances
            imp = sorted(
                zip(NUM_COLS + CAT_COLS, model_d.feature_importances_.tolist()),
                key=lambda t: -t[1],
            )
            block["lgb_direct_importance"] = imp[:20]
            print("  top importances", imp[:12], flush=True)
            fh.write(f"  top importances {imp[:12]}\n")

            # E9 residual: LGB on y - P_cal, P_cal = linear on train applied to all
            p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
            p_tr = fill_with_fallback(fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
            tr_res = tr.with_columns(pl.Series("p_cal", p_tr))
            va_res_p = p_lin  # already computed on val
            # residual target
            print("  fitting LGB residual...", flush=True)
            tr_fit_r, tr_es_r = time_es_split(tr_res)
            xtr_r, _ = to_xy(tr_fit_r)
            ytr_r = tr_fit_r["y"].to_numpy() - tr_fit_r["p_cal"].to_numpy()
            xes_r, _ = to_xy(tr_es_r)
            yes_r = tr_es_r["y"].to_numpy() - tr_es_r["p_cal"].to_numpy()
            model_r = fit_lgb(xtr_r, ytr_r, xes_r, yes_r, seed=1)
            pred_r = va_res_p + model_r.predict(xva)
            block["lgb_residual"] = evaluate("LGB residual + E3 P_cal", y, pred_r, um, ap, fh)
            block["lgb_residual_trees"] = int(model_r.best_iteration_ or model_r.n_estimators)
            imp_r = sorted(
                zip(NUM_COLS + CAT_COLS, model_r.feature_importances_.tolist()),
                key=lambda t: -t[1],
            )
            block["lgb_residual_importance"] = imp_r[:20]
            print("  residual top importances", imp_r[:12], flush=True)
            fh.write(f"  residual top {imp_r[:12]}\n")

            # LGB direct but unmatched LIRF overwritten by MVT-SCHED
            pred_mix = pred_d.copy()
            pred_mix[mask] = mvt_sched[mask]
            block["lgb_direct_LIRF_sched_override"] = evaluate(
                "LGB direct, LIRF unmatched overwritten MVT-SCHED", y, pred_mix, um, ap, fh
            )
            pred_mix_r = pred_r.copy()
            pred_mix_r[mask] = mvt_sched[mask]
            block["lgb_resid_LIRF_sched_override"] = evaluate(
                "LGB residual, LIRF unmatched overwritten MVT-SCHED", y, pred_mix_r, um, ap, fh
            )

            payload[split_name] = block

        save_result("E12_E9", payload)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
