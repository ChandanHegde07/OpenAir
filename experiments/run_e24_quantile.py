"""E24: quantile residual LightGBM (pinball), frozen E18-H features.

Independent of METAR / extra data. Each alpha is its own model on the same
frozen residual matrix (no cross-quantile data dependency). L2 is the
reproduction baseline.

Not a reopen of E15 Huber / tail-weighted L2 / two-stage P(y>30). This is
pinball loss on y − P_cal.

Training_*.parquet only. Jan+Jul primary; December stress.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, add_causal_rolling, load_dep, mae, rmse, save_result  # noqa: E402
from run_e12_e9 import CAT_COLS, time_es_split  # noqa: E402
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    apply_override,
    load_arr_delay,
    override_mask,
    score_pred,
    to_xy,
)
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from common import airport_mean_fallback, fill_with_fallback  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "E24"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP, FIG / "dec"):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
E18H_OVERALL = 372.36
E18H_MATCHED = 250.98
E20_OVERALL = 368.03
E20_MATCHED = 244.76
E20_DEC = 228.45
E20_DEC_MATCHED = 215.90
ALPHAS = (0.1, 0.5, 0.9)
E20_W = np.array([0.456, 0.053, 0.491], dtype=np.float64)
OOF_DIR = HERE / "results" / "E20"


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
        "resid_mean_matched": s["resid_mean_matched"],
        "n_gt30": s["n_gt30_matched"],
        "n_gt60": s["n_gt60_matched"],
    }


def featurize() -> pl.DataFrame:
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    dep = add_traffic(dep, load_arr())
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    return dep


def fit_lgb_obj(xtr, ytr, xes, yes, objective="regression", alpha=None, seed=SEED):
    kw = dict(
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
        objective=objective,
    )
    if alpha is not None:
        kw["alpha"] = float(alpha)
    model = lgb.LGBMRegressor(**kw)
    model.fit(
        xtr,
        ytr,
        eval_set=[(xes, yes)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def residual_pack(tr, va, extra):
    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    geo = va["geo_mean"].to_numpy().astype(float)
    fb = airport_mean_fallback(tr, va)
    p_cal = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)
    p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_tr = fill_with_fallback(
        fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    ov_tr = override_mask(
        tr["unmatched"].to_numpy(),
        tr["airport"].to_numpy(),
        tr["mvt_sched"].to_numpy().astype(float),
    )
    tr_res = tr.with_columns(pl.Series("p_cal", p_tr), pl.Series("_ov", ov_tr)).filter(~pl.col("_ov"))
    tr_fit, tr_es = time_es_split(tr_res)
    xtr, _ = to_xy(tr_fit, extra)
    xes, _ = to_xy(tr_es, extra)
    xva, _ = to_xy(va, extra)
    ytr = tr_fit["y"].to_numpy() - tr_fit["p_cal"].to_numpy()
    yes = tr_es["y"].to_numpy() - tr_es["p_cal"].to_numpy()
    ov_va = override_mask(
        va["unmatched"].to_numpy(),
        va["airport"].to_numpy(),
        va["mvt_sched"].to_numpy().astype(float),
    )
    sched = va["mvt_sched"].to_numpy().astype(float)
    return {
        "xtr": xtr,
        "ytr": ytr,
        "xes": xes,
        "yes": yes,
        "xva": xva,
        "p_cal": p_cal,
        "ov_va": ov_va,
        "sched": sched,
        "n_drop": int(ov_tr.sum()),
        "n_fit": tr_fit.height,
        "n_es": tr_es.height,
    }


def predict_resid(model, pack) -> np.ndarray:
    resid = np.asarray(model.predict(pack["xva"]), dtype=np.float64)
    return apply_override(pack["p_cal"] + resid, pack["ov_va"], pack["sched"])


def nnls_blend(preds: dict, y: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    P = np.column_stack([preds[k] for k in names])
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    w, _ = nnls(P[ok], y[ok])
    s = float(np.sum(w))
    w = w / s if s > 1e-9 else np.array([1.0] + [0.0] * (len(names) - 1))
    return w, P @ w


def featurize_dep() -> pl.DataFrame:
    log("featurize training DEP (no extra data)...")
    dep = featurize()
    log(f"  DEP {dep.height:,}")
    return dep


def make_plots(payload: dict) -> None:
    names_j = list(payload["janjul"]["models"].keys())
    for split_name, dest in (("janjul", FIG), ("dec", FIG / "dec")):
        s = payload[split_name]["models"]
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        xs = np.arange(len(names_j))
        overall = [s[k]["overall"] for k in names_j]
        matched = [s[k]["matched"] for k in names_j]
        ax.bar(xs - 0.18, overall, 0.36, label="overall", color="#4C78A8")
        ax.bar(xs + 0.18, matched, 0.36, label="matched", color="#F58518")
        ax.set_xticks(xs)
        ax.set_xticklabels(names_j, rotation=30, ha="right")
        ax.set_ylabel("RMSE (s)")
        ax.set_title(f"E24 quantile vs L2 ({split_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "01_rmse.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        gt30 = [s[k]["rmse_gt30"] for k in names_j]
        lt20 = [s[k]["rmse_lt20"] for k in names_j]
        ax.bar(xs - 0.18, lt20, 0.36, label="<20 min", color="#54A24B")
        ax.bar(xs + 0.18, gt30, 0.36, label=">30 min", color="#E45756")
        ax.set_xticks(xs)
        ax.set_xticklabels(names_j, rotation=30, ha="right")
        ax.set_ylabel("RMSE (s)")
        ax.set_title(f"Bulk vs tail ({split_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "02_bulk_tail.png")
        plt.close(fig)


def write_report(payload: dict) -> None:
    j, d = payload["janjul"], payload["dec"]
    lines = []
    a = lines.append
    a("# E24 — Quantile residual LightGBM")
    a("")
    a("**Project:** OpenAir")
    a("**Experiment:** E24")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress:** December 2025")
    a("**Data:** 12 `training_*.parquet` only. No METAR, no ranking/submitting.")
    a("Script: `experiments/run_e24_quantile.py`.")
    a("")
    a("## Question")
    a("")
    a("E15 closed Huber / tail-weighted L2 / two-stage P(y>30) — those still")
    a("fit a **conditional mean** with a different loss. Quantile regression")
    a("fits the **conditional distribution** (pinball). Median (α=0.5) and a")
    a("weighted quantile blend are the point estimates. Also tested as a")
    a("fourth NNLS expert next to E20's CatBoost / airport LGB / XGB.")
    a("")
    a("## Setup")
    a("")
    a("- Frozen: P_cal, matched-only geo_mean, LIRF-override rows dropped,")
    a("  always-on LIRF `MVT−SCHED`, E16-A disruption cols, LightGBM capacity.")
    a("- L2 reproduction must match E18-H 372.36 / 250.98.")
    a("- Quantiles α ∈ {0.1, 0.5, 0.9} on `y − P_cal` (lower / median / upper).")
    a("- Point estimates: Q50 (median residual) and NNLS of {L2, Q10, Q50, Q90}.")
    a("- Expert test: NNLS of E20 C/D/E plus Q50 (does pinball carry weight?).")
    a("")
    a("## Jan+Jul 2025")
    a("")
    a("| Model | Overall | Matched | <20 | >30 | >60 | SSE>30 | mean resid |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, blk in j["models"].items():
        a(
            f"| {name} | {blk['overall']:.2f} | {blk['matched']:.2f} | {blk['rmse_lt20']:.1f} | "
            f"{blk['rmse_gt30']:.1f} | {blk['rmse_gt60']:.1f} | {100*blk['sse_share_gt30']:.1f}% | "
            f"{blk['resid_mean_matched']:.1f} |"
        )
    a(f"| E20 ensemble (ref) | {E20_OVERALL:.2f} | {E20_MATCHED:.2f} | — | — | — | — | — |")
    a("")
    a("## December 2025")
    a("")
    a("| Model | Overall | Matched | <20 | >30 | >60 | SSE>30 | mean resid |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, blk in d["models"].items():
        a(
            f"| {name} | {blk['overall']:.2f} | {blk['matched']:.2f} | {blk['rmse_lt20']:.1f} | "
            f"{blk['rmse_gt30']:.1f} | {blk['rmse_gt60']:.1f} | {100*blk['sse_share_gt30']:.1f}% | "
            f"{blk['resid_mean_matched']:.1f} |"
        )
    a(f"| E20 ensemble (ref) | {E20_DEC:.2f} | {E20_DEC_MATCHED:.2f} | — | — | — | — | — |")
    a("")
    w = payload.get("nnls_weights") or {}
    if w:
        a("Quantile-family NNLS (Jan+Jul): " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))
        a("")
    w4 = payload.get("expert_nnls_weights") or {}
    if w4:
        a("E20+Q50 NNLS (Jan+Jul): " + ", ".join(f"{k}={v:.3f}" for k, v in w4.items()))
        a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    a("Does not have to beat E20 as a standalone to be useful. If Q50 takes")
    a("NNLS weight and does not hurt December, keep it as an expert candidate.")
    a("Readme current-model is updated only if E20+Q50 beats E20 on both splits.")
    a("")
    a("Artifacts: `analysis/E24/`.")
    a("")
    path = REP / "E24_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


def main():
    log("E24 quantile residual LGB — no METAR, frozen E18-H features.")
    extra = list(MODEL_DISRUPT_COLS)
    dep = featurize_dep()
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), "alphas": list(ALPHAS)}
    blend_names = ["L2", "Q10", "Q50", "Q90"]
    expert_names = ["C", "D", "E", "Q50"]

    for split_name, months in ("janjul", [1, 7]), ("dec", [12]):
        log(f"===== {split_name} =====")
        pack_split = prepare_split(dep, months, matched_geo=True)
        tr, va = pack_split["tr"], pack_split["va"]
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy()
        rp = residual_pack(tr, va, extra)
        log(f"  drop {rp['n_drop']:,} LIRF-ov; fit {rp['n_fit']:,} / es {rp['n_es']:,}")

        models = {}
        preds = {}

        log("  L2 residual (E18-H)...")
        m_l2 = fit_lgb_obj(rp["xtr"], rp["ytr"], rp["xes"], rp["yes"], objective="regression")
        preds["L2"] = predict_resid(m_l2, rp)
        models["L2"] = tail_block(y, preds["L2"], um)
        log(f"    L2 overall={models['L2']['overall']:.2f} matched={models['L2']['matched']:.2f} "
            f">30={models['L2']['rmse_gt30']:.1f} <20={models['L2']['rmse_lt20']:.1f}")
        if split_name == "janjul" and abs(models["L2"]["overall"] - E18H_OVERALL) > 0.5:
            raise SystemExit(f"E18-H reproduction failed: {models['L2']['overall']}")

        for a in ALPHAS:
            name = f"Q{int(round(a * 100))}"
            log(f"  {name} residual (alpha={a})...")
            mq = fit_lgb_obj(rp["xtr"], rp["ytr"], rp["xes"], rp["yes"], objective="quantile", alpha=a)
            preds[name] = predict_resid(mq, rp)
            models[name] = tail_block(y, preds[name], um)
            log(
                f"    {name} overall={models[name]['overall']:.2f} matched={models[name]['matched']:.2f} "
                f">30={models[name]['rmse_gt30']:.1f} <20={models[name]['rmse_lt20']:.1f} "
                f"mean_r={models[name]['resid_mean_matched']:.1f}"
            )

        oof = pl.read_parquet(OOF_DIR / f"oof_predictions_{split_name}.parquet")
        ids = va["MVT_ID_mvt"].to_numpy()
        aligned = pl.DataFrame({"MVT_ID_mvt": ids}).join(
            oof.select("MVT_ID_mvt", "pred_C", "pred_D", "pred_E"), on="MVT_ID_mvt", how="left"
        )
        e20_preds = {
            "C": aligned["pred_C"].to_numpy().astype(np.float64),
            "D": aligned["pred_D"].to_numpy().astype(np.float64),
            "E": aligned["pred_E"].to_numpy().astype(np.float64),
        }
        if int(np.isnan(np.column_stack(list(e20_preds.values()))).sum()):
            raise RuntimeError(f"E20 oof align failed on {split_name}")
        p_e20 = E20_W[0] * e20_preds["C"] + E20_W[1] * e20_preds["D"] + E20_W[2] * e20_preds["E"]
        models["E20"] = tail_block(y, p_e20, um)
        log(f"    E20 oof overall={models['E20']['overall']:.2f} matched={models['E20']['matched']:.2f}")

        mix = {**e20_preds, "Q50": preds["Q50"]}
        if split_name == "janjul":
            w, p_blend = nnls_blend(preds, y, blend_names)
            payload["nnls_weights"] = {k: float(v) for k, v in zip(blend_names, w)}
            log(f"  quantile-NNLS {payload['nnls_weights']}")
            w4, p_ex = nnls_blend(mix, y, expert_names)
            payload["expert_nnls_weights"] = {k: float(v) for k, v in zip(expert_names, w4)}
            log(f"  E20+Q50 NNLS {payload['expert_nnls_weights']}")
        else:
            w = np.array([payload["nnls_weights"][k] for k in blend_names], dtype=float)
            p_blend = np.column_stack([preds[k] for k in blend_names]) @ w
            w4 = np.array([payload["expert_nnls_weights"][k] for k in expert_names], dtype=float)
            p_ex = np.column_stack([mix[k] for k in expert_names]) @ w4
        models["Qblend"] = tail_block(y, p_blend, um)
        models["E20+Q50"] = tail_block(y, p_ex, um)
        log(f"    Qblend overall={models['Qblend']['overall']:.2f} matched={models['Qblend']['matched']:.2f}")
        log(f"    E20+Q50 overall={models['E20+Q50']['overall']:.2f} matched={models['E20+Q50']['matched']:.2f}")

        payload[split_name] = {"models": models}

    j, d = payload["janjul"]["models"], payload["dec"]["models"]
    q_w = float((payload.get("expert_nnls_weights") or {}).get("Q50", 0.0))
    e20q_j, e20q_d = j["E20+Q50"], d["E20+Q50"]
    med_j, med_d = j["Q50"], d["Q50"]
    # current-model bar: beat E20 both splits. Expert bar: Q50 gets weight and doesn't hurt Dec.
    beat_j = e20q_j["overall"] < j["E20"]["overall"] - 0.5 and e20q_j["matched"] <= j["E20"]["matched"] + 0.5
    beat_d = e20q_d["overall"] < d["E20"]["overall"] - 0.5
    if beat_j and beat_d:
        verdict = "ACCEPTED"
    elif q_w >= 0.05 and e20q_d["overall"] <= d["E20"]["overall"] + 0.5 and e20q_j["overall"] <= j["E20"]["overall"] + 0.5:
        verdict = "INCONCLUSIVE"
    elif med_j["overall"] < j["L2"]["overall"] - 0.5 and med_d["overall"] <= d["L2"]["overall"] + 0.5:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
    reason = (
        f"Q50 (median) Jan+Jul {med_j['overall']:.2f}/{med_j['matched']:.2f} vs L2 "
        f"{j['L2']['overall']:.2f}/{j['L2']['matched']:.2f}; >30 {j['L2']['rmse_gt30']:.1f}→{med_j['rmse_gt30']:.1f}; "
        f"<20 {j['L2']['rmse_lt20']:.1f}→{med_j['rmse_lt20']:.1f}. "
        f"E20+Q50 NNLS Q50-weight={q_w:.3f}: Jan+Jul {e20q_j['overall']:.2f} vs E20 {j['E20']['overall']:.2f}; "
        f"Dec {e20q_d['overall']:.2f} vs E20 {d['E20']['overall']:.2f}. "
        f"Qblend {j['Qblend']['overall']:.2f}/{d['Qblend']['overall']:.2f}."
    )
    payload["decision"] = {"verdict": verdict, "reason": reason}
    log(f"VERDICT {verdict}")
    log(reason)
    make_plots(payload)
    write_report(payload)
    (TAB / "e24_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    rows = []
    for split_name in ("janjul", "dec"):
        for name, blk in payload[split_name]["models"].items():
            rows.append({"split": split_name, "model": name, **blk})
    pd.DataFrame(rows).to_csv(TAB / "e24_metrics.csv", index=False)
    save_result("E24", payload)
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e24.py")
    log("done")


if __name__ == "__main__":
    main()
