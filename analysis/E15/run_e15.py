"""E15: tail-aware residual objectives on the frozen E9/E14 pipeline.

Question: does changing the residual *training objective* reduce tail error
without degrading the 10–20 min bulk? No new feature families.

Frozen: P_cal, NUM_COLS/CAT_COLS, train/val split, LIRF unmatched fallback,
preprocessing, LightGBM capacity (leaves, depth-equivalent, ES, seed).

Sequential:
  E15-A  Huber/robust residual objective
  E15-B  tail-weighted residual (long taxi + large positive residual)
  E15-C  two-stage P(y>30 min) + conditional tail residual, only if A/B fail

Fits use training_*.parquet only. Jan+Jul 2025 holdout is primary; December stress.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE if (HERE / "common.py").exists() else HERE.parents[1] / "experiments"
sys.path.insert(0, str(EXP_DIR))

from common import (  # noqa: E402
    ROOT,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    mae,
    rmse,
    save_result,
    split_by_months,
)
from run_e12_e9 import CAT_COLS, NUM_COLS, time_es_split, to_xy  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "E15"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    }
)

# Frozen LightGBM capacity from E9/E12/E14. Objective / weights may change.
LGB_KW = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)
SEED = 1
ES_ROUNDS = 40
TAIL_S = 1800.0
HOUR_S = 3600.0
BULK_S = 1200.0  # <20 min
BASELINE_MATCHED_RMSE = 256.46

VARIANT_ORDER = ["baseline", "A_huber", "B_tailweight", "C_twostage"]
VARIANT_LABEL = {
    "baseline": "E14 L2 residual",
    "A_huber": "E15-A Huber",
    "B_tailweight": "E15-B tail-weighted",
    "C_twostage": "E15-C two-stage",
}
VARIANT_COLOR = {
    "baseline": "#4C72B0",
    "A_huber": "#DD8452",
    "B_tailweight": "#55A868",
    "C_twostage": "#C44E52",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def savefig(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"  figure {path.name}")
    return path


def savecsv(name: str, df: pd.DataFrame) -> Path:
    path = TAB / name
    df.to_csv(path, index=False)
    log(f"  table {path.name}")
    return path


def json_conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def override_mask(unmatched: np.ndarray, airport: np.ndarray, mvt_sched: np.ndarray) -> np.ndarray:
    """Frozen E12/E14 LIRF unmatched fallback (type-null is almost the same set)."""
    return unmatched & (airport == "LIRF") & np.isfinite(mvt_sched)


def apply_override(pred: np.ndarray, mask: np.ndarray, mvt_sched: np.ndarray) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float64).copy()
    out[mask] = mvt_sched[mask]
    return out


def huber_delta_from_train(resid: np.ndarray, matched: np.ndarray) -> dict:
    """Canonical Huber delta from matched train P_cal residuals only."""
    r = np.asarray(resid, dtype=np.float64)
    m = np.asarray(matched, dtype=bool) & np.isfinite(r)
    rr = r[m]
    med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - med)))
    sigma = 1.4826 * mad
    delta = max(1.345 * sigma, 1.0)
    abs_r = np.abs(rr)
    return {
        "n": int(m.sum()),
        "median": med,
        "mad": mad,
        "sigma_mad": sigma,
        "delta": float(delta),
        "mae": float(np.mean(abs_r)),
        "p50_abs": float(np.median(abs_r)),
        "p90_abs": float(np.quantile(abs_r, 0.90)),
        "p95_abs": float(np.quantile(abs_r, 0.95)),
        "p99_abs": float(np.quantile(abs_r, 0.99)),
    }


def make_huber_objective(delta: float):
    """LightGBM Huber: e = pred - true; clip gradient at ±delta; hess = 1 (matches LGB built-in)."""

    def objective(y_true, y_pred):
        e = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        abs_e = np.abs(e)
        grad = np.where(abs_e <= delta, e, delta * np.sign(e))
        hess = np.ones_like(e)
        return grad, hess

    return objective


def make_huber_eval(delta: float):
    def feval(y_true, y_pred):
        e = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        abs_e = np.abs(e)
        loss = np.where(abs_e <= delta, 0.5 * e * e, delta * (abs_e - 0.5 * delta))
        return "huber", float(np.mean(loss)), False

    return feval


def tail_weight_scale(y: np.ndarray, p_cal: np.ndarray, matched: np.ndarray) -> float:
    """Median positive P_cal residual on matched train rows (not unmatched LIRF bombs)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p_cal, dtype=np.float64)
    m = np.asarray(matched, dtype=bool) & np.isfinite(y) & np.isfinite(p)
    pos = np.maximum(0.0, y[m] - p[m])
    pos = pos[pos > 0]
    if pos.size == 0:
        return 1800.0
    return float(np.median(pos))


def tail_weights(y, p_cal, tau: float, freeze_w1: np.ndarray | None = None) -> np.ndarray:
    """Weight grows with long taxi (y>30 min) and large positive residual vs P_cal.

    w = 1 + relu(y-1800)/1800 + relu(y-P_cal)/tau, clipped to [1, 10].
    Override rows keep weight 1 so the LIRF rule population does not dominate.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p_cal, dtype=np.float64)
    tau = max(float(tau), 1.0)
    long_term = np.maximum(0.0, y - TAIL_S) / TAIL_S
    pos_term = np.maximum(0.0, y - p) / tau
    w = 1.0 + long_term + pos_term
    w = np.clip(w, 1.0, 10.0)
    w[~np.isfinite(w)] = 1.0
    if freeze_w1 is not None:
        w[np.asarray(freeze_w1, dtype=bool)] = 1.0
    return w


def fit_residual_lgb(
    xtr,
    ytr,
    xes,
    yes,
    *,
    objective="regression",
    eval_metric=None,
    sample_weight=None,
    eval_sample_weight=None,
    seed=SEED,
):
    model = lgb.LGBMRegressor(objective=objective, random_state=seed, **LGB_KW)
    fit_kw = {
        "eval_set": [(xes, yes)],
        "callbacks": [lgb.early_stopping(ES_ROUNDS, verbose=False, first_metric_only=True)],
        "categorical_feature": CAT_COLS,
    }
    if eval_metric is not None:
        fit_kw["eval_metric"] = eval_metric
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    if eval_sample_weight is not None:
        fit_kw["eval_sample_weight"] = [eval_sample_weight]
    model.fit(xtr, ytr, **fit_kw)
    return model


def fit_tail_classifier(xtr, y_true_tr, xes, y_true_es, seed=SEED):
    ztr = (np.asarray(y_true_tr, dtype=np.float64) > TAIL_S).astype(np.int32)
    zes = (np.asarray(y_true_es, dtype=np.float64) > TAIL_S).astype(np.int32)
    n_pos = int(ztr.sum())
    n_neg = int((ztr == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0
    clf = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        class_weight="balanced",
        **LGB_KW,
    )
    clf.fit(
        xtr,
        ztr,
        eval_set=[(xes, zes)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False, first_metric_only=True)],
        categorical_feature=CAT_COLS,
    )
    return clf, {"n_pos": n_pos, "n_neg": n_neg, "scale_pos_weight": float(spw)}


def trees_used(model) -> int:
    return int(model.best_iteration_ or model.n_estimators)


def slice_df(x: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return x.iloc[np.asarray(mask, dtype=bool).nonzero()[0]]


def bin_rmse(y, p, mask) -> float:
    m = np.asarray(mask, dtype=bool) & np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return rmse(y[m], p[m])


def score_pred(y, pred, unmatched) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    um = np.asarray(unmatched, dtype=bool)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~um
    r = y - p
    sse = float(np.sum(r[matched] ** 2)) if matched.any() else 0.0
    gt30 = matched & (y > TAIL_S)
    gt60 = matched & (y > HOUR_S)
    lt20 = matched & (y < BULK_S)
    sse_gt30 = float(np.sum(r[gt30] ** 2)) if gt30.any() else 0.0
    return {
        "n": int(ok.sum()),
        "n_matched": int(matched.sum()),
        "n_unmatched": int((ok & um).sum()),
        "n_lt20_matched": int(lt20.sum()),
        "n_gt30_matched": int(gt30.sum()),
        "n_gt60_matched": int(gt60.sum()),
        "rmse_overall": rmse(y[ok], p[ok]),
        "rmse_matched": rmse(y[matched], p[matched]) if matched.any() else float("nan"),
        "mae_matched": mae(y[matched], p[matched]) if matched.any() else float("nan"),
        "rmse_lt20_matched": bin_rmse(y, p, lt20),
        "rmse_gt30_matched": bin_rmse(y, p, gt30),
        "rmse_gt60_matched": bin_rmse(y, p, gt60),
        "sse_share_gt30_matched": (sse_gt30 / sse) if sse > 0 else float("nan"),
        "resid_mean_matched": float(np.mean(r[matched])) if matched.any() else float("nan"),
        "resid_median_matched": float(np.median(r[matched])) if matched.any() else float("nan"),
        "rmse_unmatched": rmse(y[ok & um], p[ok & um]) if (ok & um).any() else float("nan"),
    }


TARGET_BINS = [0, 600, 900, 1200, 1800, 2700, 3600, np.inf]
TARGET_LABELS = ["<10m", "10-15m", "15-20m", "20-30m", "30-45m", "45-60m", ">60m"]


def target_bin_table(y, pred, unmatched, variant: str, split: str) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    matched = (~np.asarray(unmatched, dtype=bool)) & np.isfinite(y) & np.isfinite(p)
    yy, pp = y[matched], p[matched]
    idx = np.digitize(yy, TARGET_BINS[1:-1], right=True)
    rows = []
    sse_all = float(np.sum((yy - pp) ** 2))
    for i, lab in enumerate(TARGET_LABELS):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            continue
        r = yy[m] - pp[m]
        sse = float(np.sum(r ** 2))
        rows.append(
            {
                "split": split,
                "variant": variant,
                "bin": lab,
                "n": n,
                "mean_y": float(np.mean(yy[m])),
                "mean_pred": float(np.mean(pp[m])),
                "mean_resid": float(np.mean(r)),
                "mae": float(np.mean(np.abs(r))),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "sse": sse,
                "sse_share": sse / sse_all if sse_all else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ab_sufficient(base: dict, cand: dict) -> dict:
    """A/B wins only if the operational tail improves without bulk damage."""
    d_gt30 = cand["rmse_gt30_matched"] - base["rmse_gt30_matched"]
    d_gt60 = cand["rmse_gt60_matched"] - base["rmse_gt60_matched"]
    d_lt20 = cand["rmse_lt20_matched"] - base["rmse_lt20_matched"]
    d_matched = cand["rmse_matched"] - base["rmse_matched"]
    tail_ok = d_gt30 <= -10.0
    bulk_ok = d_lt20 <= 5.0
    matched_ok = d_matched <= 3.0
    ok = bool(tail_ok and bulk_ok and matched_ok)
    return {
        "sufficient": ok,
        "d_gt30": float(d_gt30),
        "d_gt60": float(d_gt60),
        "d_lt20": float(d_lt20),
        "d_matched": float(d_matched),
        "tail_ok": bool(tail_ok),
        "bulk_ok": bool(bulk_ok),
        "matched_ok": bool(matched_ok),
        "reason": (
            "tail improved >=10s RMSE and bulk/matched held"
            if ok
            else "tail not improved enough, or <20m/matched RMSE degraded"
        ),
    }


def prepare_split(dep: pl.DataFrame, months: list[int]) -> dict:
    tr0, va0 = split_by_months(dep, months)
    tabs = geometry_tables(tr0)
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)

    y_va = va["y"].to_numpy().astype(np.float64)
    um_va = va["unmatched"].to_numpy()
    ap_va = va["airport"].to_numpy()
    geo_va = va["geo_mean"].to_numpy().astype(float)
    sched_va = va["mvt_sched"].to_numpy().astype(float)
    fb_va = airport_mean_fallback(tr, va)

    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_cal_va = fill_with_fallback(fill_with_fallback(p_ap, geo_va), fb_va)

    p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_tr = fill_with_fallback(
        fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    tr_res = tr.with_columns(pl.Series("p_cal", p_tr))
    tr_fit, tr_es = time_es_split(tr_res)

    xtr, ytr_true = to_xy(tr_fit)
    xes, yes_true = to_xy(tr_es)
    xva, _ = to_xy(va)
    ytr_res = tr_fit["y"].to_numpy().astype(np.float64) - tr_fit["p_cal"].to_numpy().astype(np.float64)
    yes_res = tr_es["y"].to_numpy().astype(np.float64) - tr_es["p_cal"].to_numpy().astype(np.float64)

    um_fit = tr_fit["unmatched"].to_numpy()
    um_es = tr_es["unmatched"].to_numpy()
    ap_fit = tr_fit["airport"].to_numpy()
    ap_es = tr_es["airport"].to_numpy()
    sched_fit = tr_fit["mvt_sched"].to_numpy().astype(float)
    sched_es = tr_es["mvt_sched"].to_numpy().astype(float)
    pcal_fit = tr_fit["p_cal"].to_numpy().astype(np.float64)
    pcal_es = tr_es["p_cal"].to_numpy().astype(np.float64)
    y_fit = tr_fit["y"].to_numpy().astype(np.float64)
    y_es = tr_es["y"].to_numpy().astype(np.float64)

    ov_va = override_mask(um_va, ap_va, sched_va)
    ov_fit = override_mask(um_fit, ap_fit, sched_fit)
    ov_es = override_mask(um_es, ap_es, sched_es)

    return {
        "va": va,
        "y_va": y_va,
        "um_va": um_va,
        "ap_va": ap_va,
        "p_cal_va": p_cal_va,
        "sched_va": sched_va,
        "ov_va": ov_va,
        "xtr": xtr,
        "xes": xes,
        "xva": xva,
        "ytr_true": ytr_true,
        "yes_true": yes_true,
        "ytr_res": ytr_res,
        "yes_res": yes_res,
        "y_fit": y_fit,
        "y_es": y_es,
        "pcal_fit": pcal_fit,
        "pcal_es": pcal_es,
        "um_fit": um_fit,
        "um_es": um_es,
        "ov_fit": ov_fit,
        "ov_es": ov_es,
        "n_train_fit": int(tr_fit.height),
        "n_train_es": int(tr_es.height),
        "n_val": int(va.height),
    }


def predict_residual(model, pack) -> np.ndarray:
    resid_hat = np.asarray(model.predict(pack["xva"]), dtype=np.float64)
    return apply_override(pack["p_cal_va"] + resid_hat, pack["ov_va"], pack["sched_va"])


def fit_c_variant(pack: dict, baseline_model) -> tuple[np.ndarray, dict]:
    """P(y>30 min) gate + tail residual expert; mixture with frozen L2 residual."""
    log("  fitting E15-C classifier P(y>30 min)...")
    clf, clf_meta = fit_tail_classifier(pack["xtr"], pack["ytr_true"], pack["xes"], pack["yes_true"])
    p_hat = np.asarray(clf.predict_proba(pack["xva"])[:, 1], dtype=np.float64)
    log(
        f"    trees={trees_used(clf)} pos={clf_meta['n_pos']:,} "
        f"scale_pos_weight={clf_meta['scale_pos_weight']:.2f} "
        f"mean p_hat={float(np.mean(p_hat)):.4f}"
    )

    tail_tr = pack["y_fit"] > TAIL_S
    tail_es = pack["y_es"] > TAIL_S
    n_tail_tr = int(tail_tr.sum())
    n_tail_es = int(tail_es.sum())
    log(f"  fitting E15-C tail residual on y>30 min (n_fit={n_tail_tr:,} n_es={n_tail_es:,})...")
    if n_tail_tr < 200 or n_tail_es < 50:
        log("    insufficient tail rows; C falls back to baseline residual")
        r_tail = np.asarray(baseline_model.predict(pack["xva"]), dtype=np.float64)
        tail_trees = 0
    else:
        tail_model = fit_residual_lgb(
            slice_df(pack["xtr"], tail_tr),
            pack["ytr_res"][tail_tr],
            slice_df(pack["xes"], tail_es),
            pack["yes_res"][tail_es],
        )
        r_tail = np.asarray(tail_model.predict(pack["xva"]), dtype=np.float64)
        tail_trees = trees_used(tail_model)
        log(f"    tail trees={tail_trees}")

    r_l2 = np.asarray(baseline_model.predict(pack["xva"]), dtype=np.float64)
    mixed = pack["p_cal_va"] + (1.0 - p_hat) * r_l2 + p_hat * r_tail
    pred = apply_override(mixed, pack["ov_va"], pack["sched_va"])
    meta = {
        "trees_clf": trees_used(clf),
        "trees_tail": tail_trees,
        "mean_p_hat": float(np.mean(p_hat)),
        **clf_meta,
    }
    return pred, meta


def run_split(name: str, months: list[int], dep: pl.DataFrame) -> dict:
    log(f"===== {name} months={months} =====")
    pack = prepare_split(dep, months)
    log(f"  n_train_fit={pack['n_train_fit']:,} n_es={pack['n_train_es']:,} n_val={pack['n_val']:,}")

    models = {}
    meta = {}
    preds = {}

    log("  fitting baseline L2 residual...")
    models["baseline"] = fit_residual_lgb(pack["xtr"], pack["ytr_res"], pack["xes"], pack["yes_res"])
    preds["baseline"] = predict_residual(models["baseline"], pack)
    meta["baseline"] = {"trees": trees_used(models["baseline"])}
    log(f"    trees={meta['baseline']['trees']}")

    hstat = huber_delta_from_train(pack["ytr_res"], ~pack["um_fit"])
    delta = hstat["delta"]
    log(
        f"  E15-A Huber delta={delta:.1f}s "
        f"(MAD={hstat['mad']:.1f}, sigma={hstat['sigma_mad']:.1f}, "
        f"P90|r|={hstat['p90_abs']:.1f}, P99|r|={hstat['p99_abs']:.1f})"
    )
    log("  fitting E15-A Huber residual...")
    models["A_huber"] = fit_residual_lgb(
        pack["xtr"],
        pack["ytr_res"],
        pack["xes"],
        pack["yes_res"],
        objective=make_huber_objective(delta),
        eval_metric=make_huber_eval(delta),
    )
    preds["A_huber"] = predict_residual(models["A_huber"], pack)
    meta["A_huber"] = {"trees": trees_used(models["A_huber"]), "huber": hstat}
    log(f"    trees={meta['A_huber']['trees']}")

    tau = tail_weight_scale(pack["y_fit"], pack["pcal_fit"], ~pack["um_fit"])
    wtr = tail_weights(pack["y_fit"], pack["pcal_fit"], tau, pack["ov_fit"])
    wes = tail_weights(pack["y_es"], pack["pcal_es"], tau, pack["ov_es"])
    log(
        f"  E15-B tail weights tau={tau:.1f}s  "
        f"mean_w_fit={float(np.mean(wtr)):.3f} max={float(np.max(wtr)):.2f} "
        f"share_w>1={float(np.mean(wtr > 1.0 + 1e-12)):.3f}"
    )
    log("  fitting E15-B tail-weighted residual...")
    models["B_tailweight"] = fit_residual_lgb(
        pack["xtr"],
        pack["ytr_res"],
        pack["xes"],
        pack["yes_res"],
        sample_weight=wtr,
        eval_sample_weight=wes,
    )
    preds["B_tailweight"] = predict_residual(models["B_tailweight"], pack)
    meta["B_tailweight"] = {
        "trees": trees_used(models["B_tailweight"]),
        "tau": tau,
        "mean_weight_fit": float(np.mean(wtr)),
        "p90_weight_fit": float(np.quantile(wtr, 0.90)),
        "max_weight_fit": float(np.max(wtr)),
        "share_weight_gt1": float(np.mean(wtr > 1.0 + 1e-12)),
    }
    log(f"    trees={meta['B_tailweight']['trees']}")

    scores = {}
    bin_frames = []
    for v, pred in preds.items():
        scores[v] = score_pred(pack["y_va"], pred, pack["um_va"])
        scores[v]["trees"] = meta[v].get("trees", meta[v].get("trees_clf"))
        s = scores[v]
        log(
            f"  {v:14s} overall={s['rmse_overall']:.2f}  matched={s['rmse_matched']:.2f}  "
            f"MAE={s['mae_matched']:.2f}  <20={s['rmse_lt20_matched']:.2f}  "
            f">30={s['rmse_gt30_matched']:.2f}  >60={s['rmse_gt60_matched']:.2f}  "
            f"SSE>30={s['sse_share_gt30_matched']:.3f}  mean_r={s['resid_mean_matched']:.2f}"
        )
        bin_frames.append(target_bin_table(pack["y_va"], pred, pack["um_va"], v, name))

    return {
        "pack": pack,
        "preds": preds,
        "scores": scores,
        "meta": meta,
        "models": models,
        "bins": pd.concat(bin_frames, ignore_index=True),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for split, blob in results.items():
        for v, s in blob["scores"].items():
            rows.append({"split": split, "variant": v, **s})
    return pd.DataFrame(rows)


def plot_metric_bars(mdf: pd.DataFrame, split: str, variants: list[str]) -> None:
    sub = mdf[mdf["split"] == split]
    metrics = [
        ("rmse_overall", "Overall RMSE"),
        ("rmse_matched", "Matched RMSE"),
        ("rmse_lt20_matched", "<20 min RMSE"),
        ("rmse_gt30_matched", ">30 min RMSE"),
        ("rmse_gt60_matched", ">60 min RMSE"),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, v in enumerate(variants):
        row = sub[sub["variant"] == v]
        if row.empty:
            continue
        vals = [float(row.iloc[0][k]) for k, _ in metrics]
        ax.bar(
            x + (i - (len(variants) - 1) / 2) * width,
            vals,
            width=width,
            label=VARIANT_LABEL[v],
            color=VARIANT_COLOR[v],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], rotation=15)
    ax.set_ylabel("seconds")
    ax.set_title(f"E15 RMSE comparison — {split}")
    ax.legend()
    savefig(f"e15_rmse_comparison_{split}.png")


def plot_bin_rmse(bins: pd.DataFrame, split: str, variants: list[str]) -> None:
    sub = bins[bins["split"] == split]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(TARGET_LABELS))
    width = 0.8 / max(len(variants), 1)
    for i, v in enumerate(variants):
        sv = sub[sub["variant"] == v].set_index("bin")
        vals = [float(sv.loc[lab, "rmse"]) if lab in sv.index else np.nan for lab in TARGET_LABELS]
        ax.bar(
            x + (i - (len(variants) - 1) / 2) * width,
            vals,
            width=width,
            label=VARIANT_LABEL[v],
            color=VARIANT_COLOR[v],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(f"Matched RMSE by actual taxi bin — {split}")
    ax.legend()
    savefig(f"e15_target_bin_rmse_{split}.png")


def plot_bin_resid(bins: pd.DataFrame, split: str, variants: list[str]) -> None:
    sub = bins[bins["split"] == split]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(TARGET_LABELS))
    for v in variants:
        sv = sub[sub["variant"] == v].set_index("bin")
        vals = [float(sv.loc[lab, "mean_resid"]) if lab in sv.index else np.nan for lab in TARGET_LABELS]
        ax.plot(x, vals, marker="o", label=VARIANT_LABEL[v], color=VARIANT_COLOR[v])
    ax.axhline(0.0, color="0.4", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylabel("mean residual y − pred (s)")
    ax.set_title(f"Tail compression diagnostic — {split}")
    ax.legend()
    savefig(f"e15_target_bin_mean_resid_{split}.png")


def plot_sse_share(mdf: pd.DataFrame, split: str, variants: list[str]) -> None:
    sub = mdf[mdf["split"] == split]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    xs, ys, cs = [], [], []
    for v in variants:
        row = sub[sub["variant"] == v]
        if row.empty:
            continue
        xs.append(VARIANT_LABEL[v])
        ys.append(100.0 * float(row.iloc[0]["sse_share_gt30_matched"]))
        cs.append(VARIANT_COLOR[v])
    ax.bar(xs, ys, color=cs)
    ax.set_ylabel("% of matched SSE from y>30 min")
    ax.set_title(f"Tail SSE share — {split}")
    savefig(f"e15_sse_share_{split}.png")


def plot_scatter(y, pred_base, pred_best, unmatched, best_name: str) -> None:
    m = (~np.asarray(unmatched, dtype=bool)) & np.isfinite(y) & np.isfinite(pred_base) & np.isfinite(pred_best)
    # subsample for readability
    rng = np.random.default_rng(1)
    idx = np.where(m)[0]
    if idx.size > 40000:
        idx = rng.choice(idx, 40000, replace=False)
    yy = y[idx]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True)
    for ax, pp, title in (
        (axes[0], pred_base[idx], "E14 L2 residual"),
        (axes[1], pred_best[idx], VARIANT_LABEL[best_name]),
    ):
        ax.scatter(yy, pp, s=4, alpha=0.15, c="#4C72B0", linewidths=0)
        lim = [0, min(8000, max(float(np.quantile(yy, 0.999)), float(np.quantile(pp, 0.999))))]
        ax.plot(lim, lim, color="0.3", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("actual taxi (s)")
        ax.set_ylabel("predicted (s)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Jan+Jul matched: actual vs predicted (subsample)")
    savefig("e15_actual_vs_predicted.png")


def plot_resid_vs_actual(y, preds: dict, unmatched, variants: list[str]) -> None:
    m = ~np.asarray(unmatched, dtype=bool)
    rng = np.random.default_rng(1)
    idx = np.where(m)[0]
    if idx.size > 30000:
        idx = rng.choice(idx, 30000, replace=False)
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4.6), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        r = y[idx] - preds[v][idx]
        ax.scatter(y[idx], r, s=4, alpha=0.15, c=VARIANT_COLOR[v], linewidths=0)
        ax.axhline(0.0, color="0.3", lw=1)
        ax.set_xlabel("actual taxi (s)")
        ax.set_title(VARIANT_LABEL[v])
        ax.set_xlim(0, 8000)
        ax.set_ylim(-2500, 5000)
    axes[0].set_ylabel("residual y − pred (s)")
    fig.suptitle("Jan+Jul matched residual vs actual (subsample)")
    savefig("e15_residual_vs_actual.png")


def fmt_s(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.2f}"


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{100.0 * float(x):.1f}%"


def pick_winner(janjul_scores: dict, variants: list[str]) -> str:
    """Primary: matched RMSE, then >30 min RMSE, then <20 min RMSE (lower better)."""
    def key(v):
        s = janjul_scores[v]
        return (
            s["rmse_matched"],
            s["rmse_gt30_matched"],
            s["rmse_lt20_matched"],
            s["rmse_overall"],
        )

    return min(variants, key=key)


def write_report(results: dict, decision: dict, mdf: pd.DataFrame, bins: pd.DataFrame) -> Path:
    variants = decision["variants_run"]
    winner = decision["winner"]
    path = REP / "E15_report.md"
    j = results["janjul"]["scores"]
    d = results["dec"]["scores"]
    base_j = j["baseline"]
    win_j = j[winner]
    lines = []
    a = lines.append

    a("# E15 — Tail-aware residual objective")
    a("")
    a("**Project:** OpenAir")
    a("**Target:** `TAXITIME_SEC_mvt`")
    a("**Experiment:** E15")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress check:** December 2025")
    a("**Data:** 12 `training_*.parquet` files only. No ranking/submission.")
    a("")
    a("Script: `experiments/run_e15.py` (copied to `analysis/E15/run_e15.py`).")
    a("")
    a("---")
    a("")
    a("## Objective")
    a("")
    a("E14 showed the remaining matched error is **tail compression**: the model overpredicts")
    a("short taxis and underpredicts long taxis. Flights >30 min are ~4.5% of matched rows")
    a("but ~46% of matched SSE. Traffic, queue, finer geometry, airport intercepts, hour,")
    a("WTC, type, and missingness had little leftover signal.")
    a("")
    a("E15 tests whether **changing the residual training objective** reduces tail error")
    a("without degrading the normal bulk. **No new feature families.**")
    a("")
    a("## Frozen (not touched)")
    a("")
    a("- `P_cal` = per-airport OLS(`MVT−AOBT`, `AOBT−EOBT`, `geo_mean`) with geo/airport fallback")
    a("- Feature matrix `NUM_COLS` + `CAT_COLS` from E12")
    a("- Train / validation split (Jan+Jul holdout; December stress)")
    a("- Unmatched LIRF → `MVT−SCHED` (E12/E14 mask)")
    a("- Preprocessing: geometry tables, causal roll10, traffic, queue")
    a("- LightGBM capacity: 400 trees, lr 0.05, 63 leaves, min_child 80, colsample 0.8,")
    a("  λ=1, seed=1, early stopping 40 on last 15% of *train* by time")
    a("")
    a("## Variants")
    a("")
    a("| ID | Residual objective |")
    a("|---|---|")
    a("| baseline | Frozen E14 L2 residual LightGBM |")
    h = results["janjul"]["meta"]["A_huber"]["huber"]
    a(
        f"| E15-A | Huber; δ = 1.345 × 1.4826 × MAD(matched train P_cal residual) "
        f"= **{h['delta']:.1f} s** |"
    )
    tau = results["janjul"]["meta"]["B_tailweight"]["tau"]
    a(
        f"| E15-B | L2 with sample weight `1 + relu(y−1800)/1800 + relu(y−P_cal)/τ`, "
        f"τ = median positive matched train residual = **{tau:.1f} s**, clip [1, 10]; "
        f"override rows weight 1 |"
    )
    if "C_twostage" in variants:
        a(
            "| E15-C | Two-stage: LightGBM `P(y>30 min)` + residual model trained only on "
            "`y>30 min`; mixture `(1−p)·r_L2 + p·r_tail` added to frozen `P_cal` |"
        )
    else:
        a("| E15-C | **Not run.** A/B already met the tail-without-bulk-damage rule. |")
    a("")
    a("Huber δ and tail-weight τ are computed on the **training split only** (matched rows).")
    a("")
    a("## Decision rule for E15-C")
    a("")
    a("Run C only if neither A nor B, on Jan+Jul matched, does all of:")
    a("")
    a("- drop >30 min RMSE by at least 10 s")
    a("- keep <20 min RMSE from rising more than 5 s")
    a("- keep matched RMSE from rising more than 3 s")
    a("")
    a(f"**Decision:** {decision['c_reason']}")
    a("")
    a("---")
    a("")
    a("## Jan+Jul 2025 (primary)")
    a("")
    a("| Variant | Overall RMSE | Matched RMSE | Matched MAE | <20 min RMSE | >30 min RMSE | >60 min RMSE | SSE share >30 min | Residual mean |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        s = j[v]
        mark = " **" if v == winner else ""
        end = "**" if v == winner else ""
        a(
            f"| {VARIANT_LABEL[v]}{mark} | {fmt_s(s['rmse_overall'])} | {fmt_s(s['rmse_matched'])} | "
            f"{fmt_s(s['mae_matched'])} | {fmt_s(s['rmse_lt20_matched'])} | {fmt_s(s['rmse_gt30_matched'])} | "
            f"{fmt_s(s['rmse_gt60_matched'])} | {fmt_pct(s['sse_share_gt30_matched'])} | "
            f"{fmt_s(s['resid_mean_matched'])}{end} |"
        )
    a("")
    a(f"Baseline matched RMSE vs E14 journal 256.46: **{base_j['rmse_matched'] - BASELINE_MATCHED_RMSE:+.2f} s**.")
    a("")
    a("Deltas vs frozen L2 residual (negative = better):")
    a("")
    a("| Variant | Δ overall | Δ matched | Δ MAE | Δ <20 | Δ >30 | Δ >60 | Δ SSE share >30 (pp) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        if v == "baseline":
            continue
        s = j[v]
        a(
            f"| {VARIANT_LABEL[v]} | {s['rmse_overall']-base_j['rmse_overall']:+.2f} | "
            f"{s['rmse_matched']-base_j['rmse_matched']:+.2f} | "
            f"{s['mae_matched']-base_j['mae_matched']:+.2f} | "
            f"{s['rmse_lt20_matched']-base_j['rmse_lt20_matched']:+.2f} | "
            f"{s['rmse_gt30_matched']-base_j['rmse_gt30_matched']:+.2f} | "
            f"{s['rmse_gt60_matched']-base_j['rmse_gt60_matched']:+.2f} | "
            f"{100*(s['sse_share_gt30_matched']-base_j['sse_share_gt30_matched']):+.2f} |"
        )
    a("")
    a("A/B sufficiency vs the C gate:")
    a("")
    a("| Variant | Sufficient? | Δ >30 | Δ >60 | Δ <20 | Δ matched |")
    a("|---|---|---:|---:|---:|---:|")
    for v, chk in decision["ab_checks"].items():
        a(
            f"| {VARIANT_LABEL[v]} | {'yes' if chk['sufficient'] else 'no'} | "
            f"{chk['d_gt30']:+.2f} | {chk['d_gt60']:+.2f} | {chk['d_lt20']:+.2f} | {chk['d_matched']:+.2f} |"
        )
    a("")
    a("## December 2025 (stress)")
    a("")
    a("| Variant | Overall RMSE | Matched RMSE | Matched MAE | <20 min RMSE | >30 min RMSE | >60 min RMSE | SSE share >30 min | Residual mean |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        s = d[v]
        a(
            f"| {VARIANT_LABEL[v]} | {fmt_s(s['rmse_overall'])} | {fmt_s(s['rmse_matched'])} | "
            f"{fmt_s(s['mae_matched'])} | {fmt_s(s['rmse_lt20_matched'])} | {fmt_s(s['rmse_gt30_matched'])} | "
            f"{fmt_s(s['rmse_gt60_matched'])} | {fmt_pct(s['sse_share_gt30_matched'])} | "
            f"{fmt_s(s['resid_mean_matched'])} |"
        )
    a("")
    a("December deltas vs L2 residual:")
    a("")
    a("| Variant | Δ overall | Δ matched | Δ <20 | Δ >30 | Δ >60 |")
    a("|---|---:|---:|---:|---:|---:|")
    base_d = d["baseline"]
    for v in variants:
        if v == "baseline":
            continue
        s = d[v]
        a(
            f"| {VARIANT_LABEL[v]} | {s['rmse_overall']-base_d['rmse_overall']:+.2f} | "
            f"{s['rmse_matched']-base_d['rmse_matched']:+.2f} | "
            f"{s['rmse_lt20_matched']-base_d['rmse_lt20_matched']:+.2f} | "
            f"{s['rmse_gt30_matched']-base_d['rmse_gt30_matched']:+.2f} | "
            f"{s['rmse_gt60_matched']-base_d['rmse_gt60_matched']:+.2f} |"
        )
    a("")
    a("## Compression by actual-taxi bin (Jan+Jul matched)")
    a("")
    a("Mean residual (y − pred). Positive = still underpredicting long taxis.")
    a("")
    # wide-ish table
    a("| Bin | " + " | ".join(VARIANT_LABEL[v] for v in variants) + " |")
    a("|" + "---|" * (1 + len(variants)))
    jb = bins[bins["split"] == "janjul"]
    for lab in TARGET_LABELS:
        cells = [lab]
        for v in variants:
            sv = jb[(jb["variant"] == v) & (jb["bin"] == lab)]
            cells.append(fmt_s(float(sv.iloc[0]["mean_resid"])) if len(sv) else "NA")
        a("| " + " | ".join(cells) + " |")
    a("")
    a("RMSE by bin:")
    a("")
    a("| Bin | " + " | ".join(VARIANT_LABEL[v] for v in variants) + " |")
    a("|" + "---|" * (1 + len(variants)))
    for lab in TARGET_LABELS:
        cells = [lab]
        for v in variants:
            sv = jb[(jb["variant"] == v) & (jb["bin"] == lab)]
            cells.append(fmt_s(float(sv.iloc[0]["rmse"])) if len(sv) else "NA")
        a("| " + " | ".join(cells) + " |")
    a("")
    a("## What each variant did")
    a("")
    a("### E15-A Huber")
    a("")
    a("Canonical robust regression: quadratic inside ±δ, linear outside. δ is the")
    a("Gaussian-efficient Huber cutoff from the **matched train** P_cal residual MAD,")
    a("not from validation. Built-in LightGBM `alpha` is not a first-class sklearn 4.6")
    a("parameter, so A uses a custom objective that matches LightGBM's Huber gradient")
    a("(clip `pred−true` at ±δ, hessian 1) and early-stops on Huber loss on the train")
    a("time holdout.")
    a("")
    a(
        f"Jan+Jul: matched {fmt_s(j['A_huber']['rmse_matched'])} "
        f"(Δ {j['A_huber']['rmse_matched']-base_j['rmse_matched']:+.2f}), "
        f">30 min {fmt_s(j['A_huber']['rmse_gt30_matched'])} "
        f"(Δ {j['A_huber']['rmse_gt30_matched']-base_j['rmse_gt30_matched']:+.2f}), "
        f"<20 min {fmt_s(j['A_huber']['rmse_lt20_matched'])} "
        f"(Δ {j['A_huber']['rmse_lt20_matched']-base_j['rmse_lt20_matched']:+.2f})."
    )
    a("")
    a("### E15-B tail-weighted L2")
    a("")
    a("Same L2 residual trees, but training rows with long taxi and/or large positive")
    a("`y−P_cal` get higher weight. Unmatched LIRF override rows stay at weight 1 so the")
    a("already-solved unmatched bomb does not steal the residual fit. τ and the weight")
    a("formula use train-split statistics only.")
    a("")
    a(
        f"Jan+Jul: matched {fmt_s(j['B_tailweight']['rmse_matched'])} "
        f"(Δ {j['B_tailweight']['rmse_matched']-base_j['rmse_matched']:+.2f}), "
        f">30 min {fmt_s(j['B_tailweight']['rmse_gt30_matched'])} "
        f"(Δ {j['B_tailweight']['rmse_gt30_matched']-base_j['rmse_gt30_matched']:+.2f}), "
        f"<20 min {fmt_s(j['B_tailweight']['rmse_lt20_matched'])} "
        f"(Δ {j['B_tailweight']['rmse_lt20_matched']-base_j['rmse_lt20_matched']:+.2f})."
    )
    a("")
    if "C_twostage" in variants:
        a("### E15-C two-stage")
        a("")
        a("A/B did not satisfy the tail-without-bulk-damage rule, so C adds a probability")
        a("gate and a tail expert on the **same frozen features**:")
        a("")
        a("1. Binary LightGBM for `P(y > 30 min)` (`scale_pos_weight` from train counts).")
        a("2. Residual LightGBM trained only on train rows with `y > 30 min`.")
        a("3. `ŷ = P_cal + (1−p)·r_L2 + p·r_tail`, then the frozen LIRF unmatched override.")
        a("")
        a("This is still an objective / expert-mixture change, not a new feature family.")
        a("")
        a(
            f"Jan+Jul: matched {fmt_s(j['C_twostage']['rmse_matched'])} "
            f"(Δ {j['C_twostage']['rmse_matched']-base_j['rmse_matched']:+.2f}), "
            f">30 min {fmt_s(j['C_twostage']['rmse_gt30_matched'])} "
            f"(Δ {j['C_twostage']['rmse_gt30_matched']-base_j['rmse_gt30_matched']:+.2f}), "
            f"<20 min {fmt_s(j['C_twostage']['rmse_lt20_matched'])} "
            f"(Δ {j['C_twostage']['rmse_lt20_matched']-base_j['rmse_lt20_matched']:+.2f})."
        )
        a("")
    a("## Conclusion")
    a("")
    a(f"**Winner: {VARIANT_LABEL[winner]}** (Jan+Jul matched RMSE, then >30, then <20).")
    a("")
    a(
        f"- **>30 min:** baseline {fmt_s(base_j['rmse_gt30_matched'])} → "
        f"winner {fmt_s(win_j['rmse_gt30_matched'])} "
        f"({win_j['rmse_gt30_matched']-base_j['rmse_gt30_matched']:+.2f} s). "
        f"{'Improved.' if win_j['rmse_gt30_matched'] < base_j['rmse_gt30_matched'] else 'Did not improve.'}"
    )
    a(
        f"- **>60 min:** baseline {fmt_s(base_j['rmse_gt60_matched'])} → "
        f"winner {fmt_s(win_j['rmse_gt60_matched'])} "
        f"({win_j['rmse_gt60_matched']-base_j['rmse_gt60_matched']:+.2f} s). "
        f"{'Improved.' if win_j['rmse_gt60_matched'] < base_j['rmse_gt60_matched'] else 'Did not improve.'}"
    )
    a(
        f"- **<20 min:** baseline {fmt_s(base_j['rmse_lt20_matched'])} → "
        f"winner {fmt_s(win_j['rmse_lt20_matched'])} "
        f"({win_j['rmse_lt20_matched']-base_j['rmse_lt20_matched']:+.2f} s). "
        f"{'Degraded.' if win_j['rmse_lt20_matched'] > base_j['rmse_lt20_matched'] + 1e-6 else 'Not degraded.'}"
    )
    a(
        f"- **Matched RMSE:** baseline {fmt_s(base_j['rmse_matched'])} → "
        f"winner {fmt_s(win_j['rmse_matched'])} "
        f"({win_j['rmse_matched']-base_j['rmse_matched']:+.2f} s)."
    )
    a(
        f"- **SSE share from >30 min:** baseline {fmt_pct(base_j['sse_share_gt30_matched'])} → "
        f"winner {fmt_pct(win_j['sse_share_gt30_matched'])}."
    )
    a("")
    a(decision["final_narrative"])
    a("")
    a("December is a stress check, not the selection split. If the winner fails to")
    a("transfer, that is recorded above and weighs against promoting the variant.")
    a("")
    a("## Artifacts")
    a("")
    a("- `analysis/E15/figures/`")
    a("- `analysis/E15/tables/e15_metrics.csv`")
    a("- `analysis/E15/tables/e15_target_bin_metrics.csv`")
    a("- `analysis/E15/tables/e15_findings.json`")
    a("- `analysis/E15/tables/e15_matched_predictions_janjul.parquet`")
    a("- `experiments/results/E15.json`")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  report {path}")
    return path


def final_narrative(winner: str, j: dict, decision: dict) -> str:
    base = j["baseline"]
    win = j[winner]
    d30 = win["rmse_gt30_matched"] - base["rmse_gt30_matched"]
    d60 = win["rmse_gt60_matched"] - base["rmse_gt60_matched"]
    d20 = win["rmse_lt20_matched"] - base["rmse_lt20_matched"]
    dm = win["rmse_matched"] - base["rmse_matched"]
    tail_fixed = (d30 < -10) and (d60 <= 0 or d30 < -20)
    bulk_hurt = d20 > 5
    if winner == "baseline":
        more = (
            "E15-C was not justified as a next model-class jump: A/B already tested the "
            "objective change, and C "
            + ("was run and still did not beat L2." if "C_twostage" in j else "was not warranted.")
        )
        return (
            "Changing the residual objective did **not** beat frozen L2 on the ranking analogue. "
            f">30 min RMSE moved {d30:+.2f} s and <20 min moved {d20:+.2f} s under the selected variant "
            f"(the L2 baseline itself). Tail compression is not a loss-function artifact we can "
            f"remove with Huber or simple tail weights on this feature set. {more} "
            "Evidence does **not** justify another residual-objective experiment. Any next step "
            "needs a new hypothesis (rare-event features, disruption clocks already in the matrix "
            "used differently, or accepting irreducible tail noise) — not another loss tweak."
        )
    if tail_fixed and not bulk_hurt:
        nxt = (
            "The tail move is real but modest; it does not justify a broad model sweep. "
            "Keep the winning objective if it also holds on December. Do not open a new feature "
            "family on the back of this result unless December agrees and the >30 min SSE share "
            "actually fell."
            if abs(dm) < 8
            else "Promote the winning objective into the current-best residual trainer. "
            "A further loss-function experiment is not justified; the remaining tail is then "
            "a data/identifiability problem (E14 disruption tails), not an L2 artifact."
        )
        return (
            f"**{VARIANT_LABEL[winner]}** reduces operational tail error without wrecking the bulk "
            f"(matched Δ {dm:+.2f} s, >30 Δ {d30:+.2f}, >60 Δ {d60:+.2f}, <20 Δ {d20:+.2f}). {nxt}"
        )
    if tail_fixed and bulk_hurt:
        return (
            f"{VARIANT_LABEL[winner]} moves the tail (Δ >30 {d30:+.2f} s) but **degrades <20 min** "
            f"(Δ {d20:+.2f} s). That is not a clean fix of E14's compression. Do not replace L2 "
            "with this variant. Another residual-objective experiment is not justified; the "
            "tradeoff is the usual RMSE-vs-tail weighting, already measured."
        )
    return (
        f"{VARIANT_LABEL[winner]} is the least-bad of the variants on matched RMSE, but it does "
        f"not actually fix the E14 failure (>30 Δ {d30:+.2f} s, >60 Δ {d60:+.2f} s, <20 Δ {d20:+.2f} s). "
        "Evidence does **not** justify another residual-objective experiment. The tail remains "
        "mostly unidentified given the frozen features."
    )


def main():
    t0 = datetime.now(timezone.utc)
    log("E15: building frozen features from training_*.parquet only...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    log(f"ready {dep.height:,}")

    # Sequential: A/B on both splits, decide C from Jan+Jul, then C if needed.
    results = {}
    results["janjul"] = run_split("janjul", [1, 7], dep)
    results["dec"] = run_split("dec", [12], dep)

    ab_checks = {}
    any_ok = False
    for v in ("A_huber", "B_tailweight"):
        chk = ab_sufficient(results["janjul"]["scores"]["baseline"], results["janjul"]["scores"][v])
        ab_checks[v] = chk
        log(f"  A/B gate {v}: sufficient={chk['sufficient']} ({chk['reason']})")
        any_ok = any_ok or chk["sufficient"]

    run_c = not any_ok
    if run_c:
        log("A/B insufficient on Jan+Jul — running E15-C two-stage on both splits.")
        for split_name in ("janjul", "dec"):
            blob = results[split_name]
            pred_c, meta_c = fit_c_variant(blob["pack"], blob["models"]["baseline"])
            blob["preds"]["C_twostage"] = pred_c
            blob["scores"]["C_twostage"] = score_pred(
                blob["pack"]["y_va"], pred_c, blob["pack"]["um_va"]
            )
            blob["meta"]["C_twostage"] = meta_c
            s = blob["scores"]["C_twostage"]
            log(
                f"  C_twostage    overall={s['rmse_overall']:.2f}  matched={s['rmse_matched']:.2f}  "
                f"MAE={s['mae_matched']:.2f}  <20={s['rmse_lt20_matched']:.2f}  "
                f">30={s['rmse_gt30_matched']:.2f}  >60={s['rmse_gt60_matched']:.2f}  "
                f"SSE>30={s['sse_share_gt30_matched']:.3f}  mean_r={s['resid_mean_matched']:.2f}"
            )
            bins_c = target_bin_table(
                blob["pack"]["y_va"], pred_c, blob["pack"]["um_va"], "C_twostage", split_name
            )
            blob["bins"] = pd.concat([blob["bins"], bins_c], ignore_index=True)
    else:
        log("A/B sufficient — skipping E15-C.")

    variants = [v for v in VARIANT_ORDER if v in results["janjul"]["scores"]]
    winner = pick_winner(results["janjul"]["scores"], variants)
    log(f"winner={winner}")

    decision = {
        "run_c": run_c,
        "c_reason": (
            "Neither A nor B dropped >30 min RMSE by ≥10 s while holding <20 min and matched RMSE. Running C."
            if run_c
            else "A and/or B already improved the tail without bulk damage. C not run."
        ),
        "ab_checks": ab_checks,
        "variants_run": variants,
        "winner": winner,
    }
    decision["final_narrative"] = final_narrative(winner, results["janjul"]["scores"], decision)

    mdf = metrics_table(results)
    bins = pd.concat([results["janjul"]["bins"], results["dec"]["bins"]], ignore_index=True)
    savecsv("e15_metrics.csv", mdf)
    savecsv("e15_target_bin_metrics.csv", bins)

    for split in ("janjul", "dec"):
        plot_metric_bars(mdf, split, variants)
        plot_bin_rmse(bins, split, variants)
        plot_bin_resid(bins, split, variants)
        plot_sse_share(mdf, split, variants)

    y = results["janjul"]["pack"]["y_va"]
    um = results["janjul"]["pack"]["um_va"]
    preds = results["janjul"]["preds"]
    plot_scatter(y, preds["baseline"], preds[winner], um, winner)
    plot_resid_vs_actual(y, preds, um, variants)

    # matched predictions (Jan+Jul)
    pack = results["janjul"]["pack"]
    matched = ~pack["um_va"]
    pdf = pd.DataFrame(
        {
            "airport": pack["ap_va"][matched],
            "y": pack["y_va"][matched],
            "p_cal": pack["p_cal_va"][matched],
        }
    )
    for v in variants:
        pdf[f"pred_{v}"] = preds[v][matched]
        pdf[f"resid_{v}"] = pdf["y"] - pdf[f"pred_{v}"]
    pred_path = TAB / "e15_matched_predictions_janjul.parquet"
    pdf.to_parquet(pred_path, index=False)
    log(f"  table {pred_path.name}")

    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "janjul": {
            "scores": results["janjul"]["scores"],
            "meta": {
                k: {kk: vv for kk, vv in meta.items() if kk != "huber" or True}
                for k, meta in results["janjul"]["meta"].items()
            },
        },
        "dec": {
            "scores": results["dec"]["scores"],
            "meta": results["dec"]["meta"],
        },
        "baseline_delta_vs_e14": results["janjul"]["scores"]["baseline"]["rmse_matched"] - BASELINE_MATCHED_RMSE,
    }
    (TAB / "e15_findings.json").write_text(json.dumps(findings, indent=2, default=json_conv), encoding="utf-8")
    save_result("E15", findings)
    write_report(results, decision, mdf, bins)

    src = Path(__file__).resolve()
    dst = OUT / "run_e15.py"
    shutil.copy2(src, dst)
    log(f"copied script to {dst}")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log(f"E15 done in {elapsed/60:.1f} min. winner={winner}")
    # reprint headline
    for split in ("janjul", "dec"):
        log(f"--- {split} ---")
        for v in variants:
            s = results[split]["scores"][v]
            log(
                f"{v:14s} overall={s['rmse_overall']:.2f} matched={s['rmse_matched']:.2f} "
                f"MAE={s['mae_matched']:.2f} <20={s['rmse_lt20_matched']:.2f} "
                f">30={s['rmse_gt30_matched']:.2f} >60={s['rmse_gt60_matched']:.2f} "
                f"SSE30={s['sse_share_gt30_matched']:.3f} mean_r={s['resid_mean_matched']:.2f}"
            )


if __name__ == "__main__":
    main()
