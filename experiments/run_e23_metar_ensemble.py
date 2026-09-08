"""E23: stack E22 METAR into E20's three winning experts (not a fourth model).

E20 NNLS = 0.456 CatBoost residual + 0.491 per-airport LGB + 0.053 XGB residual.
This experiment adds the same METAR columns as E22 to C, D, and E, then:

  1. scores each expert with vs without METAR (same-run ablation)
  2. applies frozen E20 weights 0.456/0.053/0.491 to the METAR experts
  3. refits NNLS on the METAR-augmented C, D, E (Jan+Jul holdout only)

Dec is the transfer check. The risk to catch is Jan+Jul overfitting on a
12-column add-on. Training_*.parquet only. No ranking/submitting in fits.
No E15 loss work, no E19 rolling-queue revival, no fourth expert.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

import xgboost as xgb  # noqa: E402
from catboost import CatBoostRegressor, Pool  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import AIRPORTS, ROOT, add_causal_rolling, load_dep, mae, rmse, save_result  # noqa: E402
from run_e12_e9 import CAT_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    load_arr_delay,
    score_pred,
)
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_e20_ensemble import (  # noqa: E402
    NUM_FEATS,
    apply_override,
    calibrate,
    cat_pdf,
    lgb_pdf,
    residual_target,
    xgb_pdf,
)
from run_e22_metar import METAR_NUM, decode_and_join, load_metar  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "E23"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
OOF_DIR = HERE / "results" / "E20"
for d in (FIG, TAB, REP, FIG / "dec"):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
E20_OVERALL = 368.03
E20_MATCHED = 244.76
E20_DEC = 228.45
E20_DEC_MATCHED = 215.90
# published NNLS on (C, D, E); A and B were zero
W_E20 = np.array([0.456, 0.053, 0.491], dtype=np.float64)
EXPERTS = ("C", "D", "E")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def tail_block(y, p, um) -> dict:
    s = score_pred(y, p, um)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~um
    r2 = (y - p) ** 2
    sse = float(np.sum(r2[matched])) if matched.any() else 0.0
    gt60 = matched & (y > 3600.0)
    sse_gt60 = float(np.sum(r2[gt60])) if gt60.any() else 0.0
    return {
        "overall": s["rmse_overall"],
        "matched": s["rmse_matched"],
        "unmatched": s["rmse_unmatched"],
        "mae_matched": s["mae_matched"],
        "rmse_lt20": s["rmse_lt20_matched"],
        "rmse_gt30": s["rmse_gt30_matched"],
        "rmse_gt60": s["rmse_gt60_matched"],
        "sse_share_gt30": s["sse_share_gt30_matched"],
        "sse_share_gt60": (sse_gt60 / sse) if sse > 0 else float("nan"),
        "n_gt30": s["n_gt30_matched"],
        "n_gt60": s["n_gt60_matched"],
        "n_matched": s["n_matched"],
        "n_unmatched": s["n_unmatched"],
    }


def airport_matched_rmse(y, p, um, ap) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    out = {}
    for a in AIRPORTS:
        sel = (~um) & (ap == a) & np.isfinite(y) & np.isfinite(p)
        out[a] = rmse(y[sel], p[sel]) if sel.any() else float("nan")
    return out


def blend(preds: dict, w: np.ndarray) -> np.ndarray:
    return w[0] * preds["C"] + w[1] * preds["D"] + w[2] * preds["E"]


def nnls_cde(preds: dict, y: np.ndarray, um: np.ndarray) -> np.ndarray:
    P = np.column_stack([preds["C"], preds["D"], preds["E"]])
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    w, _ = nnls(P[ok], y[ok])
    s = float(np.sum(w))
    return w / s if s > 1e-9 else np.array([1.0, 0.0, 0.0])


def metar_gain_pairs(names, gain) -> list:
    gain = np.asarray(gain, dtype=float)
    if len(gain) != len(names):
        return []
    return [(n, float(g)) for n, g in zip(names, gain) if n.startswith("met_")]


def train_cde(tr, va, num_feats, label: str):
    """CatBoost / XGB / airport-LGB residual experts. Same hparams as E20."""
    p_tr, p_va = calibrate(tr, va)
    tr = tr.with_columns(pl.Series("p_cal", p_tr))
    va = va.with_columns(pl.Series("p_cal", p_va))

    um_tr = tr["unmatched"].to_numpy().astype(bool)
    ap_tr = tr["airport"].to_numpy()
    ov = um_tr & (ap_tr == "LIRF")
    tr_res = tr.filter(~pl.Series(ov))
    tr_fit, tr_es = time_es_split(tr_res)
    log(f"    {label}: drop {int(ov.sum()):,} LIRF-ov; fit {tr_fit.height:,} / es {tr_es.height:,}")

    y_va = va["y"].to_numpy().astype(np.float64)
    um_va = va["unmatched"].to_numpy().astype(bool)
    ap_va = va["airport"].to_numpy()
    sched_va = va["mvt_sched"].to_numpy().astype(np.float64)
    yres_tr = residual_target(tr_fit)
    yres_es = residual_target(tr_es)

    cat_tr = cat_pdf(tr_fit, num_feats)
    cat_es = cat_pdf(tr_es, num_feats)
    cat_va = cat_pdf(va, num_feats)
    xgb_tr = xgb_pdf(tr_fit, num_feats)
    xgb_es = xgb_pdf(tr_es, num_feats)
    xgb_va = xgb_pdf(va, num_feats)
    pdf_tr = lgb_pdf(tr_fit, num_feats)
    pdf_es = lgb_pdf(tr_es, num_feats)
    pdf_va = lgb_pdf(va, num_feats)

    cols_cat = list(num_feats) + list(CAT_COLS)

    log(f"    {label} C CatBoost residual...")
    mC = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
    )
    mC.fit(
        Pool(cat_tr[cols_cat], yres_tr, cat_features=CAT_COLS),
        eval_set=Pool(cat_es[cols_cat], yres_es, cat_features=CAT_COLS),
        early_stopping_rounds=60,
    )
    predC = apply_override(
        p_va + np.asarray(mC.predict(cat_va[cols_cat]), dtype=np.float64),
        um_va,
        ap_va,
        sched_va,
    )
    c_gain = metar_gain_pairs(cols_cat, mC.get_feature_importance())

    log(f"    {label} D XGBoost residual...")
    mD = xgb.train(
        {
            "objective": "reg:squarederror",
            "eta": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 80,
            "lambda": 1.0,
            "seed": SEED,
            "tree_method": "hist",
        },
        xgb.DMatrix(xgb_tr[num_feats], yres_tr),
        num_boost_round=2000,
        evals=[(xgb.DMatrix(xgb_es[num_feats], yres_es), "es")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    predD = apply_override(
        p_va + np.asarray(mD.predict(xgb.DMatrix(xgb_va[num_feats])), dtype=np.float64),
        um_va,
        ap_va,
        sched_va,
    )
    xgb_scores = mD.get_score(importance_type="gain")
    d_gain = [(n, float(xgb_scores.get(n, 0.0))) for n in METAR_NUM]

    log(f"    {label} E airport residual LGB...")
    # global LGB residual used only if an airport is too small (E20 fell back to A)
    mG = fit_lgb(
        pdf_tr[cols_cat],
        yres_tr,
        pdf_es[cols_cat],
        yres_es,
        seed=SEED,
    )
    predG = apply_override(
        p_va + np.asarray(mG.predict(pdf_va[cols_cat]), dtype=np.float64),
        um_va,
        ap_va,
        sched_va,
    )
    predE = np.full(len(y_va), np.nan)
    e_gain_by_ap = {}
    n_fallback = 0
    for a in AIRPORTS:
        tr_a = tr_res.filter(pl.col("airport") == a)
        va_a = va.filter(pl.col("airport") == a)
        va_idx = np.where(ap_va == a)[0]
        if tr_a.height < 20000 or va_a.height == 0:
            predE[va_idx] = predG[va_idx]
            n_fallback += 1
            continue
        ta_fit, ta_es = time_es_split(tr_a)
        pdf_a = lgb_pdf(ta_fit, num_feats)
        pdf_ae = lgb_pdf(ta_es, num_feats)
        pdf_av = lgb_pdf(va_a, num_feats)
        ma = fit_lgb(
            pdf_a[cols_cat],
            residual_target(ta_fit),
            pdf_ae[cols_cat],
            residual_target(ta_es),
            seed=SEED,
        )
        resid = np.asarray(ma.predict(pdf_av[cols_cat]), dtype=np.float64)
        predE[va_idx] = apply_override(
            va_a["p_cal"].to_numpy() + resid,
            um_va[va_idx],
            ap_va[va_idx],
            sched_va[va_idx],
        )
        booster = getattr(ma, "booster_", None)
        if booster is not None:
            g = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            e_gain_by_ap[a] = metar_gain_pairs(cols_cat, g)
    if n_fallback:
        log(f"    {label} E fallback airports: {n_fallback}")

    preds = {"C": predC, "D": predD, "E": predE}
    ids = va["MVT_ID_mvt"].to_numpy()
    return {
        "preds": preds,
        "y": y_va,
        "um": um_va,
        "ap": ap_va,
        "ids": ids,
        "c_gain": sorted(c_gain, key=lambda t: -t[1]),
        "d_gain": sorted(d_gain, key=lambda t: -t[1]),
        "e_gain_by_ap": e_gain_by_ap,
    }


def load_e20_oof(split: str, ids: np.ndarray, y: np.ndarray, um: np.ndarray, ap: np.ndarray):
    df = pl.read_parquet(OOF_DIR / f"oof_predictions_{split}.parquet")
    src = df.select(
        "MVT_ID_mvt",
        pl.col("pred_C").alias("C"),
        pl.col("pred_D").alias("D"),
        pl.col("pred_E").alias("E"),
    )
    order = pl.DataFrame({"MVT_ID_mvt": ids})
    j = order.join(src, on="MVT_ID_mvt", how="left")
    preds = {k: j[k].to_numpy().astype(np.float64) for k in EXPERTS}
    if int(np.isnan(np.column_stack(list(preds.values()))).sum()) != 0:
        raise RuntimeError(f"E20 oof align failed on {split}")
    p = blend(preds, W_E20)
    blk = tail_block(y, p, um)
    experts = {k: tail_block(y, preds[k], um) for k in EXPERTS}
    return {"preds": preds, "blend": blk, "experts": experts, "airport": airport_matched_rmse(y, p, um, ap)}


def featurize() -> pl.DataFrame:
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    dep = add_traffic(dep, load_arr())
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    return dep


def pack_scores(pack, y, um, ap) -> dict:
    out = {k: tail_block(y, pack["preds"][k], um) for k in EXPERTS}
    for k in EXPERTS:
        out[k]["airport"] = airport_matched_rmse(y, pack["preds"][k], um, ap)
    return out


def make_plots(payload: dict) -> None:
    for split_name, dest in (("janjul", FIG), ("dec", FIG / "dec")):
        s = payload[split_name]
        labels = ["Overall", "Matched", ">30 min", ">60 min"]
        keys = ["overall", "matched", "rmse_gt30", "rmse_gt60"]
        models = [
            ("E20 frozen", s["e20_blend"], "#4C78A8"),
            ("wx + frozen w", s["wx_frozen"], "#54A24B"),
            ("wx + NNLS", s["wx_nnls"], "#F58518"),
        ]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        width = 0.24
        for i, (name, blk, col) in enumerate(models):
            vals = [blk[k] for k in keys]
            ax.bar(x + (i - 1) * width, vals, width, label=name, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE (s)")
        ax.set_title(f"E23 vs E20 ({split_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "01_rmse_comparison.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        labs = ["C CatBoost", "D XGB", "E airport"]
        nowx = [s["nowx"][k]["overall"] for k in EXPERTS]
        wx = [s["wx"][k]["overall"] for k in EXPERTS]
        xx = np.arange(len(labs))
        ax.bar(xx - 0.18, nowx, 0.36, label="no METAR", color="#4C78A8")
        ax.bar(xx + 0.18, wx, 0.36, label="+ METAR", color="#F58518")
        ax.set_xticks(xx)
        ax.set_xticklabels(labs)
        ax.set_ylabel("Overall RMSE (s)")
        ax.set_title(f"Expert ablation ({split_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "02_expert_ablation.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        nowx30 = [s["nowx"][k]["rmse_gt30"] for k in EXPERTS]
        wx30 = [s["wx"][k]["rmse_gt30"] for k in EXPERTS]
        ax.bar(xx - 0.18, nowx30, 0.36, label="no METAR", color="#4C78A8")
        ax.bar(xx + 0.18, wx30, 0.36, label="+ METAR", color="#F58518")
        ax.axhline(s["e20_blend"]["rmse_gt30"], color="#B279A2", ls="--", lw=1.2, label="E20 blend")
        ax.set_xticks(xx)
        ax.set_xticklabels(labs)
        ax.set_ylabel(">30 min RMSE (s)")
        ax.set_title(f">30 min tail by expert ({split_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "03_tail_gt30.png")
        plt.close(fig)

    # per-airport E vs E_wx, January+July and December side by side
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=False)
    for ax, split_name in zip(axes, ("janjul", "dec")):
        s = payload[split_name]
        e0 = [s["nowx"]["E"]["airport"][a] for a in AIRPORTS]
        e1 = [s["wx"]["E"]["airport"][a] for a in AIRPORTS]
        xx = np.arange(len(AIRPORTS))
        ax.bar(xx - 0.18, e0, 0.36, label="E no METAR", color="#4C78A8")
        ax.bar(xx + 0.18, e1, 0.36, label="E + METAR", color="#F58518")
        ax.set_xticks(xx)
        ax.set_xticklabels(AIRPORTS, rotation=45, ha="right")
        ax.set_ylabel("Matched RMSE (s)")
        ax.set_title(split_name)
        ax.legend(fontsize=8)
    fig.suptitle("Airport expert E: METAR by airport")
    fig.tight_layout()
    fig.savefig(FIG / "04_airport_E.png")
    plt.close(fig)


def write_report(payload: dict) -> None:
    j, d = payload["janjul"], payload["dec"]
    w_nnls = payload["nnls_weights"]
    lines = []
    a = lines.append
    a("# E23 — METAR stacked into E20 experts")
    a("")
    a("**Project:** OpenAir")
    a("**Experiment:** E23")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress:** December 2025")
    a("**Data:** 12 `training_*.parquet` + Iowa Mesonet ASOS/METAR (same join as E22).")
    a("No ranking/submitting in fits.")
    a("Script: `experiments/run_e23_metar_ensemble.py`.")
    a("")
    a("## Question")
    a("")
    a("E22 added METAR to the E18-H LightGBM residual and got Jan+Jul −0.56 s")
    a("(tail unchanged) vs December −9 s / >30 min 716→651. E20's winning blend")
    a("does not use that LightGBM (weight 0). This experiment puts the **same")
    a("METAR columns** into the three experts that actually carry weight:")
    a("CatBoost residual (C, 0.456), per-airport LightGBM (E, 0.491), XGBoost")
    a("residual (D, 0.053). Not a fourth model. NNLS is refit on the")
    a("METAR-augmented C/D/E. Frozen E20 weights are the control.")
    a("")
    a("Risk to catch: 12 extra columns overfit Jan+Jul. December >30 min gain")
    a("must still be visible **inside the ensemble**, not only on a single LGB.")
    a("")
    a("## Setup")
    a("")
    a("- Join: E22 `decode_and_join` (AOBT else MVT, stale >3 h nulled). Row")
    a("  order restored so `time_es_split` ties match E20.")
    a("- Hygiene: matched-only `geo_mean`, LIRF-override rows dropped from")
    a("  every expert, always-on LIRF `MVT−SCHED`.")
    a("- Same hparams as E20 for C/D/E. A/A2/B are not trained and not blended.")
    a("- Frozen weights: C=0.456, D=0.053, E=0.491.")
    a("- NNLS fit on Jan+Jul holdout only; December is transfer.")
    a("")
    a("## Experts (with vs without METAR)")
    a("")
    a("### Jan+Jul 2025")
    a("")
    a("| Expert | no METAR overall | +METAR overall | Δ | no METAR matched | +METAR matched | >30 no | >30 wx |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, name in (("C", "C CatBoost"), ("D", "D XGB"), ("E", "E airport")):
        b, n = j["nowx"][k], j["wx"][k]
        a(
            f"| {name} | {b['overall']:.2f} | {n['overall']:.2f} | {n['overall']-b['overall']:+.2f} | "
            f"{b['matched']:.2f} | {n['matched']:.2f} | {b['rmse_gt30']:.1f} | {n['rmse_gt30']:.1f} |"
        )
    a("")
    a("### December 2025")
    a("")
    a("| Expert | no METAR overall | +METAR overall | Δ | no METAR matched | +METAR matched | >30 no | >30 wx |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, name in (("C", "C CatBoost"), ("D", "D XGB"), ("E", "E airport")):
        b, n = d["nowx"][k], d["wx"][k]
        a(
            f"| {name} | {b['overall']:.2f} | {n['overall']:.2f} | {n['overall']-b['overall']:+.2f} | "
            f"{b['matched']:.2f} | {n['matched']:.2f} | {b['rmse_gt30']:.1f} | {n['rmse_gt30']:.1f} |"
        )
    a("")
    a("Airport expert E, matched RMSE by airport (does weather land locally?):")
    a("")
    a("| Airport | Jan+Jul E | Jan+Jul E+wx | Δ | Dec E | Dec E+wx | Δ |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for ap in AIRPORTS:
        e0, e1 = j["nowx"]["E"]["airport"][ap], j["wx"]["E"]["airport"][ap]
        d0, d1 = d["nowx"]["E"]["airport"][ap], d["wx"]["E"]["airport"][ap]
        a(f"| {ap} | {e0:.2f} | {e1:.2f} | {e1-e0:+.2f} | {d0:.2f} | {d1:.2f} | {d1-d0:+.2f} |")
    a("")
    a("## Blends")
    a("")
    a("### Jan+Jul 2025")
    a("")
    a("| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE>30 | SSE>60 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")

    def row(name, blk):
        return (
            f"| {name} | {blk['overall']:.2f} | {blk['matched']:.2f} | {blk['unmatched']:.2f} | "
            f"{blk['rmse_gt30']:.1f} | {blk['rmse_gt60']:.1f} | {100*blk['sse_share_gt30']:.1f}% | "
            f"{100*blk['sse_share_gt60']:.1f}% |"
        )

    a(row("E20 frozen (oof)", j["e20_blend"]))
    a(row("C/D/E no-wx + frozen w", j["nowx_frozen"]))
    a(row("C/D/E +METAR + frozen w", j["wx_frozen"]))
    a(row("C/D/E +METAR + NNLS", j["wx_nnls"]))
    a("")
    a("### December 2025")
    a("")
    a("| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE>30 | SSE>60 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    a(row("E20 frozen (oof)", d["e20_blend"]))
    a(row("C/D/E no-wx + frozen w", d["nowx_frozen"]))
    a(row("C/D/E +METAR + frozen w", d["wx_frozen"]))
    a(row("C/D/E +METAR + NNLS", d["wx_nnls"]))
    a("")
    a(
        f"NNLS weights on METAR experts (C, D, E), fit Jan+Jul only: "
        f"**C={w_nnls[0]:.3f}, D={w_nnls[1]:.3f}, E={w_nnls[2]:.3f}** "
        f"(E20 was C=0.456, D=0.053, E=0.491)."
    )
    a("")
    a("## METAR gain (Jan+Jul)")
    a("")
    if j.get("c_gain"):
        a("CatBoost C:")
        a("")
        for n, g in j["c_gain"]:
            a(f"- `{n}`: {g:.1f}")
        a("")
    if j.get("d_gain"):
        a("XGBoost D:")
        a("")
        for n, g in j["d_gain"]:
            a(f"- `{n}`: {g:.1f}")
        a("")
    a("Airport E — `met_precip` / `met_tmpf` / `met_relh` gain by airport (top 3 shown in findings JSON).")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    a("Accept only if the METAR-augmented blend beats E20 on **both** splits")
    a("without regressing matched/overall, and without hurting Jan+Jul (the")
    a("overfit risk). Readme current-model is updated only on ACCEPT.")
    a("")
    a("Artifacts: `analysis/E23/`, `experiments/results/E23.json`.")
    a("")
    path = REP / "E23_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


def main():
    log("E23: METAR into E20 experts C/D/E (not a fourth model).")
    num_wx = list(NUM_FEATS) + [c for c in METAR_NUM if c not in NUM_FEATS]
    log("load METAR + featurize DEP...")
    met = load_metar()
    dep = featurize()
    dep = dep.with_row_index("_ord")
    dep = decode_and_join(dep, met)
    dep = dep.sort("_ord").drop("_ord")
    log(f"  DEP {dep.height:,} met_missing={int(dep['met_missing'].sum()):,}")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "e20_weights_CDE": W_E20.tolist(),
        "metar_cols": list(METAR_NUM),
    }

    for split_name, months in ("janjul", [1, 7]), ("dec", [12]):
        log(f"===== {split_name} =====")
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]

        nowx = train_cde(tr, va, list(NUM_FEATS), f"{split_name} no-wx")
        wx = train_cde(tr, va, num_wx, f"{split_name} +METAR")

        y, um, ap = wx["y"], wx["um"], wx["ap"]
        e20 = load_e20_oof(split_name, wx["ids"], y, um, ap)

        nowx_sc = pack_scores(nowx, y, um, ap)
        wx_sc = pack_scores(wx, y, um, ap)
        nowx_frozen = tail_block(y, blend(nowx["preds"], W_E20), um)
        wx_frozen = tail_block(y, blend(wx["preds"], W_E20), um)

        if split_name == "janjul":
            w_nnls = nnls_cde(wx["preds"], y, um)
            payload["nnls_weights"] = [float(x) for x in w_nnls]
            log(f"  NNLS C/D/E = {w_nnls[0]:.3f}/{w_nnls[1]:.3f}/{w_nnls[2]:.3f}")
        else:
            w_nnls = np.asarray(payload["nnls_weights"], dtype=np.float64)

        wx_nnls = tail_block(y, blend(wx["preds"], w_nnls), um)

        log(
            f"  E20 oof {e20['blend']['overall']:.2f}/{e20['blend']['matched']:.2f} "
            f">30={e20['blend']['rmse_gt30']:.1f}"
        )
        log(
            f"  nowx frozen {nowx_frozen['overall']:.2f}/{nowx_frozen['matched']:.2f} "
            f">30={nowx_frozen['rmse_gt30']:.1f}"
        )
        log(
            f"  wx frozen  {wx_frozen['overall']:.2f}/{wx_frozen['matched']:.2f} "
            f">30={wx_frozen['rmse_gt30']:.1f}"
        )
        log(
            f"  wx NNLS    {wx_nnls['overall']:.2f}/{wx_nnls['matched']:.2f} "
            f">30={wx_nnls['rmse_gt30']:.1f}"
        )
        for k in EXPERTS:
            log(
                f"    {k} {nowx_sc[k]['overall']:.2f} -> {wx_sc[k]['overall']:.2f} "
                f"(>30 {nowx_sc[k]['rmse_gt30']:.1f}->{wx_sc[k]['rmse_gt30']:.1f})"
            )

        payload[split_name] = {
            "e20_blend": e20["blend"],
            "e20_experts": e20["experts"],
            "nowx": nowx_sc,
            "wx": wx_sc,
            "nowx_frozen": nowx_frozen,
            "wx_frozen": wx_frozen,
            "wx_nnls": wx_nnls,
            "c_gain": wx["c_gain"],
            "d_gain": wx["d_gain"],
            "e_gain_by_ap": wx["e_gain_by_ap"],
        }

        oof = pl.DataFrame(
            {
                "MVT_ID_mvt": wx["ids"],
                "y": y,
                "unmatched": um,
                "airport": ap,
                "pred_C": nowx["preds"]["C"],
                "pred_D": nowx["preds"]["D"],
                "pred_E": nowx["preds"]["E"],
                "pred_C_wx": wx["preds"]["C"],
                "pred_D_wx": wx["preds"]["D"],
                "pred_E_wx": wx["preds"]["E"],
                "pred_e20": blend(e20["preds"], W_E20),
                "pred_wx_frozen": blend(wx["preds"], W_E20),
                "pred_wx_nnls": blend(wx["preds"], w_nnls),
            }
        )
        oof.write_parquet(TAB / f"e23_oof_{split_name}.parquet")

    j, d = payload["janjul"], payload["dec"]
    cand_j = j["wx_nnls"]
    cand_d = d["wx_nnls"]
    # if frozen-wx beats refit NNLS on Jan+Jul without Dec regression, mention both
    dj = cand_j["overall"] - j["e20_blend"]["overall"]
    dd = cand_d["overall"] - d["e20_blend"]["overall"]
    dj_m = cand_j["matched"] - j["e20_blend"]["matched"]
    dd_m = cand_d["matched"] - d["e20_blend"]["matched"]
    dj_f = j["wx_frozen"]["overall"] - j["e20_blend"]["overall"]
    dd_f = d["wx_frozen"]["overall"] - d["e20_blend"]["overall"]

    janjul_hurt = (dj > 0.5) or (dj_m > 0.5)
    beat_j = (dj < -0.5) and (dj_m <= 0.5)
    beat_d = (dd < -0.5) and (dd_m <= 0.5)
    dec_gt30_e20 = d["e20_blend"]["rmse_gt30"]
    dec_gt30_nnls = cand_d["rmse_gt30"]
    tail_survives = dec_gt30_nnls < dec_gt30_e20 - 1.0

    if janjul_hurt:
        verdict = "REJECTED"
    elif beat_j and beat_d:
        verdict = "ACCEPTED"
    elif (dj <= 0.5 and dj_m <= 0.5) and (dd < -0.5 or tail_survives):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECTED"

    reason = (
        f"E20→E23-NNLS Jan+Jul {j['e20_blend']['overall']:.2f}→{cand_j['overall']:.2f} ({dj:+.2f}), "
        f"matched {j['e20_blend']['matched']:.2f}→{cand_j['matched']:.2f} ({dj_m:+.2f}); "
        f">30 {j['e20_blend']['rmse_gt30']:.1f}→{cand_j['rmse_gt30']:.1f}; "
        f"SSE>30 {100*j['e20_blend']['sse_share_gt30']:.1f}%→{100*cand_j['sse_share_gt30']:.1f}%. "
        f"Dec {d['e20_blend']['overall']:.2f}→{cand_d['overall']:.2f} ({dd:+.2f}), "
        f"matched {d['e20_blend']['matched']:.2f}→{cand_d['matched']:.2f} ({dd_m:+.2f}); "
        f">30 {dec_gt30_e20:.1f}→{dec_gt30_nnls:.1f} "
        f"({'survives' if tail_survives else 'diluted/absent'} in ensemble). "
        f"Frozen-w on wx: Jan+Jul {dj_f:+.2f}, Dec {dd_f:+.2f}. "
        f"NNLS C/D/E={payload['nnls_weights'][0]:.3f}/"
        f"{payload['nnls_weights'][1]:.3f}/{payload['nnls_weights'][2]:.3f}."
    )
    payload["decision"] = {
        "verdict": verdict,
        "reason": reason,
        "d_janjul": dj,
        "d_dec": dd,
        "janjul_hurt": janjul_hurt,
        "dec_gt30_survives": tail_survives,
    }
    log(f"VERDICT {verdict}")
    log(reason)

    make_plots(payload)
    write_report(payload)
    rows = []
    for split_name in ("janjul", "dec"):
        s = payload[split_name]
        for name, blk in (
            ("e20", s["e20_blend"]),
            ("nowx_frozen", s["nowx_frozen"]),
            ("wx_frozen", s["wx_frozen"]),
            ("wx_nnls", s["wx_nnls"]),
        ):
            rows.append({"split": split_name, "model": name, **{k: blk[k] for k in blk if k != "airport"}})
        for k in EXPERTS:
            rows.append({"split": split_name, "model": f"{k}_nowx", **{kk: s["nowx"][k][kk] for kk in s["nowx"][k] if kk != "airport"}})
            rows.append({"split": split_name, "model": f"{k}_wx", **{kk: s["wx"][k][kk] for kk in s["wx"][k] if kk != "airport"}})
    pd.DataFrame(rows).to_csv(TAB / "e23_metrics.csv", index=False)
    (TAB / "e23_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    save_result("E23", payload)
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e23.py")
    log("done")


if __name__ == "__main__":
    main()
