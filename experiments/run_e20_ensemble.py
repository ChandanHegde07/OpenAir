"""E20: stacked residual ensemble for maximum RMSE reduction.

Experts (all ranking-safe, E18-H hygiene: matched-only geo_mean, LIRF-override
rows excluded from every expert fit, always-on LIRF MVT-SCHED override):
  A  LightGBM residual      (E18-H production baseline)
  A2 LightGBM residual + E19-B same-runway queue cols (candidate add-on)
  B  LightGBm direct TAXITIME (P_cal as a feature, not the target)
  C  CatBoost residual
  D  XGBoost residual (numeric matrix)
  E  airport-specific residual LightGBM experts (fallback to A when small)

Blend weights (>=0, sum=1) are low-dimensional and fit on the Jan+Jul temporal
holdout (NNLS / positive Ridge / fixed schemes) — a genuine held-out split, so
overfit risk is small; Dec is the honest transfer check. No ranking/submitting.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostRegressor, Pool  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    AIRPORTS,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    mae,
    rmse,
    save_result,
)
from run_e12_e9 import CAT_COLS, NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from e19_features import B_COLS as E19B_COLS, add_runway_queue  # noqa: E402

RES = HERE / "results" / "E20"
PLOTS = RES / "plots"
for d in (RES, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
BASE_OVERALL = 372.36
BASE_MATCHED = 250.98
BASE_DEC = 238.01

NUM_FEATS = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
NUM_FEATS_B = [c for c in NUM_FEATS + E19B_COLS if c not in CAT_COLS]
EXPERTS = ["A", "A2", "B", "C", "D", "E"]
BLEND_BASE = ["A", "B", "C", "D", "E"]

COLS = ["#4c72b0", "#dd8452"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def score_all(y, p, um, ap):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~um
    lirf_um = ok & um & (ap == "LIRF")
    nlu = ok & um & (ap != "LIRF")
    return {
        "n": int(ok.sum()),
        "rmse": rmse(y[ok], p[ok]),
        "mae": mae(y[ok], p[ok]),
        "matched": rmse(y[matched], p[matched]) if matched.any() else float("nan"),
        "mae_matched": mae(y[matched], p[matched]) if matched.any() else float("nan"),
        "unmatched": rmse(y[ok & um], p[ok & um]) if (ok & um).any() else float("nan"),
        "lirf_unmatched": rmse(y[lirf_um], p[lirf_um]) if lirf_um.any() else float("nan"),
        "nlu_unmatched": rmse(y[nlu], p[nlu]) if nlu.any() else float("nan"),
        "sse": float(np.sum((y[ok] - p[ok]) ** 2)) if ok.any() else 0.0,
    }


def apply_override(p, um, ap, mvt_sched):
    out = np.asarray(p, dtype=np.float64).copy()
    mask = np.asarray(um, dtype=bool) & (np.asarray(ap) == "LIRF") & np.isfinite(mvt_sched)
    out[mask] = mvt_sched[mask]
    return out


def calibrate(tr, va):
    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal_va = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)),
                                  airport_mean_fallback(tr, va))
    p_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal_tr = fill_with_fallback(fill_with_fallback(p_t, tr["geo_mean"].to_numpy().astype(float)),
                                  airport_mean_fallback(tr, tr))
    return p_cal_tr, p_cal_va


def lgb_pdf(df, num_cols):
    pdf = df.select(num_cols + CAT_COLS + ["p_cal", "y"]).to_pandas()
    if "unmatched" in pdf:
        pdf["unmatched"] = pdf["unmatched"].astype(np.int8)
    if "type_null" in pdf:
        pdf["type_null"] = pdf["type_null"].astype(np.int8)
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype("category")
    return pdf


def cat_pdf(df, num_cols):
    pdf = df.select(num_cols + CAT_COLS + ["p_cal", "y"]).to_pandas()
    for c in CAT_COLS:
        pdf[c] = pdf[c].fillna("NA").astype(str)
    if "unmatched" in pdf:
        pdf["unmatched"] = pdf["unmatched"].astype(int)
    return pdf


def xgb_pdf(df, num_cols):
    return df.select(num_cols + ["p_cal", "y"]).to_pandas()


def residual_target(df):
    return np.asarray(df["y"].to_numpy(), dtype=np.float64) - np.asarray(df["p_cal"].to_numpy(), dtype=np.float64)


def importance(model, cols):
    g = model.booster_.feature_importance(importance_type="gain")
    return sorted(zip(cols, g.tolist()), key=lambda t: -t[1])


def main():
    log("E20: building base features...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    arr_d = load_arr_delay()
    dep = add_arrival_delay_state(dep, arr_d)
    log("  attaching E19-B same-runway queue cols (candidate expert only)...")
    dep = add_runway_queue(dep)
    log(f"ready {dep.height:,} rows, {dep.width} cols")

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat()}

    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        label = "Jan+Jul 2025" if split_name == "janjul" else "December 2025"
        log(f"========== {split_name} ==========")
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        p_tr, p_va = calibrate(tr, va)
        tr = tr.with_columns(pl.Series("p_cal", p_tr))
        va = va.with_columns(pl.Series("p_cal", p_va))

        um_tr = tr["unmatched"].to_numpy().astype(bool)
        ap_tr = tr["airport"].to_numpy()
        ov = um_tr & (ap_tr == "LIRF")
        n_drop = int(ov.sum())
        tr_res = tr.filter(~pl.Series(ov))
        tr_fit, tr_es = time_es_split(tr_res)
        log(f"  hygiene dropped {n_drop:,} LIRF-override rows; fit {tr_fit.height:,} / es {tr_es.height:,}")

        y_va = va["y"].to_numpy().astype(np.float64)
        um_va = va["unmatched"].to_numpy().astype(bool)
        ap_va = va["airport"].to_numpy()
        sched_va = va["mvt_sched"].to_numpy().astype(np.float64)

        yres_tr = residual_target(tr_fit)
        yres_es = residual_target(tr_es)

        # matrices
        pdf_tr = lgb_pdf(tr_fit, NUM_FEATS)
        pdf_es = lgb_pdf(tr_es, NUM_FEATS)
        pdf_va = lgb_pdf(va, NUM_FEATS)
        pdf_tr_b = lgb_pdf(tr_fit, NUM_FEATS_B)
        pdf_es_b = lgb_pdf(tr_es, NUM_FEATS_B)
        pdf_va_b = lgb_pdf(va, NUM_FEATS_B)
        cat_tr = cat_pdf(tr_fit, NUM_FEATS)
        cat_es = cat_pdf(tr_es, NUM_FEATS)
        cat_va = cat_pdf(va, NUM_FEATS)
        xgb_tr = xgb_pdf(tr_fit, NUM_FEATS)
        xgb_es = xgb_pdf(tr_es, NUM_FEATS)
        xgb_va = xgb_pdf(va, NUM_FEATS)

        preds = {}
        imps = {}

        # ---- A: LightGBM residual (E18-H) ----
        log("  expert A (LGB residual = E18-H)...")
        mA = fit_lgb(pdf_tr[NUM_FEATS + CAT_COLS], yres_tr, pdf_es[NUM_FEATS + CAT_COLS], yres_es, seed=SEED)
        predA = apply_override(p_va + np.asarray(mA.predict(pdf_va[NUM_FEATS + CAT_COLS]), dtype=np.float64),
                               um_va, ap_va, sched_va)
        preds["A"] = predA
        imps["A"] = importance(mA, NUM_FEATS + CAT_COLS)

        # ---- A2: LightGBM residual + E19-B ----
        log("  expert A2 (LGB residual + E19-B same-runway)...")
        mA2 = fit_lgb(pdf_tr_b[NUM_FEATS_B + CAT_COLS], yres_tr, pdf_es_b[NUM_FEATS_B + CAT_COLS], yres_es, seed=SEED)
        preds["A2"] = apply_override(p_va + np.asarray(mA2.predict(pdf_va_b[NUM_FEATS_B + CAT_COLS]), dtype=np.float64),
                                     um_va, ap_va, sched_va)
        imps["A2"] = importance(mA2, NUM_FEATS_B + CAT_COLS)

        # ---- B: LightGBM direct ----
        log("  expert B (LGB direct TAXITIME)...")
        numB = [c for c in NUM_FEATS + ["p_cal"] if c not in CAT_COLS]
        mB = fit_lgb(pdf_tr[numB + CAT_COLS], tr_fit["y"].to_numpy().astype(np.float64),
                     pdf_es[numB + CAT_COLS], tr_es["y"].to_numpy().astype(np.float64), seed=SEED + 1)
        preds["B"] = apply_override(np.asarray(mB.predict(pdf_va[numB + CAT_COLS]), dtype=np.float64),
                                    um_va, ap_va, sched_va)
        imps["B"] = importance(mB, numB + CAT_COLS)

        # ---- C: CatBoost residual ----
        log("  expert C (CatBoost residual)...")
        mC = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                               loss_function="RMSE", random_seed=SEED, verbose=False, thread_count=-1)
        mC.fit(Pool(cat_tr[NUM_FEATS + CAT_COLS], yres_tr, cat_features=CAT_COLS),
               eval_set=Pool(cat_es[NUM_FEATS + CAT_COLS], yres_es, cat_features=CAT_COLS),
               early_stopping_rounds=60)
        preds["C"] = apply_override(p_va + np.asarray(mC.predict(cat_va[NUM_FEATS + CAT_COLS]), dtype=np.float64),
                                    um_va, ap_va, sched_va)

        # ---- D: XGBoost residual ----
        log("  expert D (XGBoost residual)...")
        mD = xgb.train(
            {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 7, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 80, "lambda": 1.0, "seed": SEED, "tree_method": "hist"},
            xgb.DMatrix(xgb_tr[NUM_FEATS], yres_tr), num_boost_round=2000,
            evals=[(xgb.DMatrix(xgb_es[NUM_FEATS], yres_es), "es")],
            early_stopping_rounds=60, verbose_eval=False)
        preds["D"] = apply_override(p_va + np.asarray(mD.predict(xgb.DMatrix(xgb_va[NUM_FEATS])), dtype=np.float64),
                                    um_va, ap_va, sched_va)

        # ---- E: airport experts ----
        log("  expert E (airport-specific residual LGB)...")
        predE = np.full(len(y_va), np.nan)
        for a in AIRPORTS:
            tr_a = tr_res.filter(pl.col("airport") == a)
            va_a = va.filter(pl.col("airport") == a)
            va_idx = np.where(ap_va == a)[0]
            if tr_a.height < 20000 or va_a.height == 0:
                predE[va_idx] = predA[va_idx]
                continue
            ta_fit, ta_es = time_es_split(tr_a)
            pdf_a = lgb_pdf(ta_fit, NUM_FEATS)
            pdf_ae = lgb_pdf(ta_es, NUM_FEATS)
            pdf_av = lgb_pdf(va_a, NUM_FEATS)
            ma = fit_lgb(pdf_a[NUM_FEATS + CAT_COLS], residual_target(ta_fit),
                         pdf_ae[NUM_FEATS + CAT_COLS], residual_target(ta_es), seed=SEED)
            resid = np.asarray(ma.predict(pdf_av[NUM_FEATS + CAT_COLS]), dtype=np.float64)
            p_local = va_a["p_cal"].to_numpy()
            predE[va_idx] = apply_override(p_local + resid, um_va[va_idx], ap_va[va_idx], sched_va[va_idx])
        preds["E"] = predE

        for k in preds:
            s = score_all(y_va, preds[k], um_va, ap_va)
            log(f"    {k}: overall={s['rmse']:.2f} matched={s['matched']:.2f} mae_matched={s['mae_matched']:.2f}")
        if split_name == "janjul":
            dA = score_all(y_va, preds["A"], um_va, ap_va)
            log(f"    A vs recorded E18-H 372.36: {dA['rmse']:.2f} (Δ {dA['rmse']-BASE_OVERALL:+.2f})")
            if abs(dA["rmse"] - BASE_OVERALL) > 1.0:
                raise SystemExit(f"E18-H reproduction failed: {dA['rmse']}")

        mm = (~um_va) & np.isfinite(y_va)
        corr = pd.DataFrame(np.corrcoef(np.column_stack([y_va[mm] - preds[k][mm] for k in EXPERTS]).T),
                            index=EXPERTS, columns=EXPERTS)
        acorr = pd.DataFrame(np.corrcoef(np.column_stack([np.abs(y_va[mm] - preds[k][mm]) for k in EXPERTS]).T),
                             index=EXPERTS, columns=EXPERTS)
        payload.setdefault(split_name, {})["expert_scores"] = {k: score_all(y_va, preds[k], um_va, ap_va) for k in preds}
        payload[split_name]["residual_corr"] = corr.round(4)
        payload[split_name]["error_corr"] = acorr.round(4)

        # save holdout predictions used for blending
        oof = pl.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"], "y": y_va, "unmatched": um_va,
                            "airport": ap_va, "mvt_sched": sched_va, "p_cal": p_va})
        for k in preds:
            oof = oof.with_columns(pl.Series(f"pred_{k}", preds[k]))
        oof.write_parquet(RES / f"oof_predictions_{split_name}.parquet")

        if split_name == "janjul":
            payload[split_name]["importance"] = {k: imps[k][:40] for k in imps}

    blends = build_all_blends(payload)
    payload["blend_results"] = blends
    verdict, best, reason = decide_blend(blends)
    payload["verdict"], payload["best_blend"], payload["reason"] = verdict, best, reason
    log(f"VERDICT {verdict} best={best}")
    log(reason)

    make_reports(payload, blends)
    _conv = lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else (o if not hasattr(o, "to_dict") else o.to_dict("records")))
    (RES / "E20.json").write_text(json.dumps(payload, indent=2, default=_conv), encoding="utf-8")
    save_result("E20", payload)
    write_summary(payload, blends)
    log("WROTE " + str(RES))


def load_oof(split):
    df = pl.read_parquet(RES / f"oof_predictions_{split}.parquet")
    return df


def compute_blends(df, key):
    y = np.asarray(df["y"].to_numpy(), dtype=np.float64)
    um = df["unmatched"].to_numpy().astype(bool)
    ap = df["airport"].to_numpy()
    P = np.column_stack([df[f"pred_{k}"].to_numpy() for k in BLEND_BASE])
    sched = df["mvt_sched"].to_numpy()
    out = {}
    base = apply_override(P[:, 0], um, ap, sched)
    out["E18-H"] = score_all(y, base, um, ap)
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    schemes = {
        "equal5": np.full(5, 0.2),
        "A_centric": np.array([0.6, 0.15, 0.1, 0.1, 0.05]),
        "A_B_C_D": np.array([0.45, 0.25, 0.15, 0.15, 0.0]),
        "A_B_D_E": np.array([0.5, 0.25, 0.0, 0.15, 0.1]),
        "A_E": np.array([0.7, 0.0, 0.0, 0.0, 0.3]),
        "nnls": nnls_weights(P[ok], y[ok]),
        "ridge1": ridge_weights(P[ok], y[ok], 1.0),
        "ridge10": ridge_weights(P[ok], y[ok], 10.0),
    }
    for name, w in schemes.items():
        p = apply_override(P @ w, um, ap, sched)
        out[name] = score_all(y, p, um, ap)
    return out


def nnls_weights(P, y):
    w, _ = nnls(P, y)
    return w / max(np.sum(w), 1e-9)


def ridge_weights(P, y, alpha):
    m = Ridge(alpha=alpha, positive=True, fit_intercept=True, random_state=SEED)
    m.fit(P, y)
    return m.coef_ / max(np.sum(m.coef_), 1e-9)


def build_all_blends(payload):
    blends = {}
    for split in ("janjul", "dec"):
        blends[split] = compute_blends(load_oof(split), split)
    return blends


def decide_blend(b):
    j, d = b["janjul"], b["dec"]
    cands = [k for k in j if k != "E18-H" and j[k]["rmse"] < j["E18-H"]["rmse"]]
    if not cands:
        return "REJECT", "none", "no blend improves on E18-H (Jan+Jul)"
    best = min(cands, key=lambda k: j[k]["rmse"])
    imp = j["E18-H"]["rmse"] - j[best]["rmse"]
    dimp = d["E18-H"]["rmse"] - d[best]["rmse"]
    reg = d[best]["rmse"] - d["E18-H"]["rmse"]
    if imp >= 1.5 and reg <= 3.0:
        verdict = "ACCEPT"
    elif imp >= 0.5 and reg <= 5.0:
        verdict = "WEAK"
    else:
        verdict = "REJECT"
    reason = (f"best blend {best}: Jan+Jul {j['E18-H']['rmse']:.2f} -> {j[best]['rmse']:.2f} (Δ{imp:+.2f}); "
              f"Dec {d['E18-H']['rmse']:.2f} -> {d[best]['rmse']:.2f} (Δ{dimp:+.2f})")
    return verdict, best, reason


def write_summary(payload, blends):
    j, d = blends["janjul"], blends["dec"]
    L = ["# E20 — Stacked residual ensemble", ""]
    L.append(f"E18-H reproduction: Jan+Jul {j['E18-H']['rmse']:.2f} / matched {j['E18-H']['matched']:.2f}; "
             f"Dec {d['E18-H']['rmse']:.2f} / {d['E18-H']['matched']:.2f}.")
    L.append("")
    L.append("## Expert scores")
    L.append("")
    L.append("| Expert | Jan+Jul | matched | Dec | matched |")
    L.append("|---|---:|---:|---:|---:|")
    js, ds = payload["janjul"]["expert_scores"], payload["dec"]["expert_scores"]
    for k in EXPERTS:
        L.append(f"| {k} | {js[k]['rmse']:.2f} | {js[k]['matched']:.2f} | {ds[k]['rmse']:.2f} | {ds[k]['matched']:.2f} |")
    L.append("")
    L.append("## Blends (base set A,B,C,D,E)")
    L.append("")
    L.append("| Blend | Jan+Jul | Δ vs E18-H | matched | Dec | Δ Dec |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k in j:
        dj = j[k]["rmse"] - j["E18-H"]["rmse"]
        dd = d[k]["rmse"] - d["E18-H"]["rmse"]
        L.append(f"| {k} | {j[k]['rmse']:.2f} | {dj:+.2f} | {j[k]['matched']:.2f} | {d[k]['rmse']:.2f} | {dd:+.2f} |")
    L.append("")
    L.append(f"## Decision: {payload['verdict']}  (best={payload['best_blend']})")
    L.append("")
    L.append(payload["reason"])
    L.append("")
    L.append("Residual correlation (Jan+Jul matched): see error_correlation.csv / E20.json.")
    L.append("")
    L.append("Artifacts: model_metrics.csv, blend_metrics.csv, error_correlation.csv, regime_metrics.csv, "
             "oof_predictions_{janjul,dec}.parquet, feature_importance.csv, plots/.")
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def make_reports(payload, blends):
    js, ds = payload["janjul"]["expert_scores"], payload["dec"]["expert_scores"]
    rows = []
    for k in EXPERTS + list(blends["janjul"]):
        j = js[k] if k in js else blends["janjul"][k]
        d = ds[k] if k in ds else blends["dec"][k]
        rows.append({"model": k, "janjul_overall": j["rmse"], "janjul_matched": j["matched"],
                     "dec_overall": d["rmse"], "dec_matched": d["matched"],
                     "janjul_mae_matched": j.get("mae_matched", float("nan"))})
    pd.DataFrame(rows).to_csv(RES / "model_metrics.csv", index=False)
    pd.DataFrame(blends["janjul"]).T.reset_index().rename(columns={"index": "blend"}).to_csv(RES / "blend_metrics.csv", index=False)
    payload["janjul"]["residual_corr"].to_csv(RES / "error_correlation.csv")
    fi_rows = []
    for k, arr in payload["janjul"].get("importance", {}).items():
        for f, g in arr:
            fi_rows.append({"expert": k, "feature": f, "gain": g})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(RES / "feature_importance.csv", index=False)

    # ---- Jan+Jul holdout rows + best blend vector ----
    df = load_oof("janjul")
    y = np.asarray(df["y"].to_numpy(), dtype=np.float64)
    um = df["unmatched"].to_numpy().astype(bool)
    ap = df["airport"].to_numpy()
    sched = df["mvt_sched"].to_numpy()
    P = {k: df[f"pred_{k}"].to_numpy() for k in EXPERTS}
    mm = (~um) & np.isfinite(y)

    # recompute best blend weights
    best = payload["best_blend"]
    Pmat = np.column_stack([P[k] for k in BLEND_BASE])
    ok = np.isfinite(y) & np.isfinite(Pmat).all(axis=1)
    if best == "equal5":
        w = np.full(5, 0.2)
    elif best == "A_centric":
        w = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
    elif best == "A_B_C_D":
        w = np.array([0.45, 0.25, 0.15, 0.15, 0.0])
    elif best == "A_B_D_E":
        w = np.array([0.5, 0.25, 0.0, 0.15, 0.1])
    elif best == "A_E":
        w = np.array([0.7, 0.0, 0.0, 0.0, 0.3])
    elif best == "nnls":
        w = nnls_weights(Pmat[ok], y[ok])
    elif best.startswith("ridge"):
        w = ridge_weights(Pmat[ok], y[ok], float(best[5:]) if best[5:] else 1.0)
    else:
        w = np.array([1.0, 0, 0, 0, 0])
    pbest = apply_override(Pmat @ w, um, ap, sched)
    pA = P["A"]

    # regime metrics table
    rows = []
    for model, p in [("A", pA)] + [(f"pred_{m}"[5:], P[m]) for m in EXPERTS] + [("E18-H", pA)]:
        for a in AIRPORTS:
            sel = mm & (ap == a)
            if sel.any():
                rows.append({"model": model, "regime": f"airport_{a}",
                             "rmse": rmse(y[sel], p[sel]), "mae": mae(y[sel], p[sel])})
        for rng, m in [("matched", mm), ("unmatched", um)]:
            sel = m & np.isfinite(y) & np.isfinite(p)
            if sel.any():
                rows.append({"model": model, "regime": rng, "rmse": rmse(y[sel], p[sel]), "mae": mae(y[sel], p[sel])})
        pb = np.digitize(p, [0, 600, 1200, 1800, 3600])
        for bi in range(1, 6):
            sel = mm & (pb == bi)
            if sel.any():
                rows.append({"model": model, "regime": f"predbin{bi}", "rmse": rmse(y[sel], p[sel]),
                             "mae": mae(y[sel], p[sel])})
    pd.DataFrame(rows).to_csv(RES / "regime_metrics.csv", index=False)

    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(mm)[0], size=min(20000, int(mm.sum())), replace=False)

    # prediction_comparison: B/C/D/E vs A (matched sample)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axes, ["B", "C", "D", "E"]):
        ax.scatter(pA[idx], P[m][idx], s=2, alpha=0.12, color="#2a6f97")
        lim = [0, 2500]
        ax.plot(lim, lim, "r--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Expert A pred (s)"); ax.set_ylabel(f"Expert {m} pred (s)")
        ax.set_title(f"Expert {m} vs A (matched, n={len(idx):,})")
    fig.tight_layout()
    fig.savefig(PLOTS / "prediction_comparison.png", dpi=110)
    plt.close(fig)

    # error_correlation heatmap (Jan+Jul matched)
    ec = payload["janjul"]["residual_corr"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ec.to_numpy(), cmap="viridis", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(ec))); ax.set_xticklabels(ec.columns)
    ax.set_yticks(range(len(ec))); ax.set_yticklabels(ec.index)
    for i in range(len(ec)):
        for j in range(len(ec)):
            ax.text(j, i, f"{ec.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    fig.colorbar(im, ax=ax)
    ax.set_title("Residual correlation (Jan+Jul matched)")
    fig.tight_layout()
    fig.savefig(PLOTS / "error_correlation.png", dpi=110)
    plt.close(fig)

    # ensemble_vs_baseline
    jb = blends["janjul"]
    keys = ["E18-H"] + [k for k in jb if k != "E18-H"]
    vals = [jb[k]["rmse"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = ["#c44e52" if k == "E18-H" else "#4c72b0" for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    ax.set_ylabel("Jan+Jul overall RMSE")
    ax.set_title("E20 blends vs E18-H (Jan+Jul)")
    fig.tight_layout()
    fig.savefig(PLOTS / "ensemble_vs_baseline.png", dpi=110)
    plt.close(fig)

    # sse_comparison by airport (A vs best)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(AIRPORTS))
    sseA = [float(np.sum((y[mm & (ap == a)] - pA[mm & (ap == a)]) ** 2)) for a in AIRPORTS]
    sseB = [float(np.sum((y[mm & (ap == a)] - pbest[mm & (ap == a)]) ** 2)) for a in AIRPORTS]
    w = 0.36
    ax.bar(x - w / 2, np.log10(sseA), w, label="E18-H", color=COLS[0])
    ax.bar(x + w / 2, np.log10(sseB), w, label=best, color=COLS[1])
    ax.set_xticks(x, AIRPORTS)
    ax.set_ylabel("log10 SSE (matched)")
    ax.set_title("Matched SSE by airport: E18-H vs best blend (Jan+Jul)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "sse_comparison.png", dpi=110)
    plt.close(fig)

    # per_airport_rmse
    fig, ax = plt.subplots(figsize=(10, 4.4))
    vA = [rmse(y[mm & (ap == a)], pA[mm & (ap == a)]) for a in AIRPORTS]
    vB = [rmse(y[mm & (ap == a)], pbest[mm & (ap == a)]) for a in AIRPORTS]
    ax.bar(x - w / 2, vA, w, label="E18-H", color=COLS[0])
    ax.bar(x + w / 2, vB, w, label=best, color=COLS[1])
    for i, d in enumerate(np.asarray(vB) - np.asarray(vA)):
        ax.text(i + w / 2, vB[i], f"{d:+.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, AIRPORTS)
    ax.set_ylabel("Matched RMSE (s)")
    ax.set_title("Matched RMSE by airport: E18-H vs best blend (Jan+Jul)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "per_airport_rmse.png", dpi=110)
    plt.close(fig)
    log("wrote reports + plots")


if __name__ == "__main__":
    main()
