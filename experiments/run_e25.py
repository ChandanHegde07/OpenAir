"""E25: one-shot structural rebuild for taxi-out.

Architecture (built together, validated once):
  floor[airport,stand,runway] = hierarchical p10 (time-forward / train-only)
  target = log(y - floor)
  features = clocks + 10/15 min same-runway congestion (DEP AOBT + ARR) +
             OOF James-Stein encodings + calendar
  one LightGBM per airport on the log-excess, Duan smearing back-transform
  unmatched: LIRF MVT-SCHED override; other unmatched scored by the airport
  model unless a schedule-gap specialist wins both holdouts.

No ranking/submitting in any fit. Jan+Jul primary, December must not regress.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from common import (  # noqa: E402
    AIRPORTS,
    ROOT,
    load_dep,
    mae,
    rmse,
    save_result,
    split_by_months,
)
from e25_features import (  # noqa: E402
    CAT_COLS,
    CAT_UNMATCHED,
    NUM_COLS,
    NUM_UNMATCHED,
    add_calendar,
    add_congestion,
    assert_ranking_safe_features,
    attach_encodings_oof,
    attach_floors_time_forward,
    back_transform,
    load_arr_rwy,
    log_excess,
    smear_factor,
)
from run_e12_e9 import time_es_split  # noqa: E402

OUT = ROOT / "analysis" / "E25"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
RES = HERE / "results" / "E25"
for d in (FIG, TAB, REP, RES, FIG / "dec"):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
E20_JANJUL = {"overall": 368.03, "matched": 244.76}
E20_DEC = {"overall": 228.45, "matched": 215.90}
LB_RMSE = 316.9654

LGB_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


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


def score_all(y, p, um, ap) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ap = np.asarray(ap)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~um
    lirf_um = ok & um & (ap == "LIRF")
    nlu = ok & um & (ap != "LIRF")
    gt30 = matched & (y > 1800)
    gt60 = matched & (y > 3600)
    lt20 = matched & (y < 1200)
    out = {
        "n": int(ok.sum()),
        "n_matched": int(matched.sum()),
        "n_unmatched": int((ok & um).sum()),
        "rmse": rmse(y[ok], p[ok]) if ok.any() else float("nan"),
        "mae": mae(y[ok], p[ok]) if ok.any() else float("nan"),
        "matched": rmse(y[matched], p[matched]) if matched.any() else float("nan"),
        "mae_matched": mae(y[matched], p[matched]) if matched.any() else float("nan"),
        "unmatched": rmse(y[ok & um], p[ok & um]) if (ok & um).any() else float("nan"),
        "lirf_unmatched": rmse(y[lirf_um], p[lirf_um]) if lirf_um.any() else float("nan"),
        "nlu_unmatched": rmse(y[nlu], p[nlu]) if nlu.any() else float("nan"),
        "rmse_lt20": rmse(y[lt20], p[lt20]) if lt20.any() else float("nan"),
        "rmse_gt30": rmse(y[gt30], p[gt30]) if gt30.any() else float("nan"),
        "rmse_gt60": rmse(y[gt60], p[gt60]) if gt60.any() else float("nan"),
        "sse": float(np.sum((y[ok] - p[ok]) ** 2)) if ok.any() else 0.0,
    }
    for a in AIRPORTS:
        sel = ok & (ap == a)
        out[f"rmse_{a}"] = rmse(y[sel], p[sel]) if sel.any() else float("nan")
    return out


def apply_lirf_override(p, um, ap, mvt_sched):
    out = np.asarray(p, dtype=np.float64).copy()
    mask = np.asarray(um, dtype=bool) & (np.asarray(ap) == "LIRF") & np.isfinite(mvt_sched)
    out[mask] = mvt_sched[mask]
    return out


def to_pdf(df: pl.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    cols = [c for c in num + cat if c in df.columns]
    pdf = df.select(cols).to_pandas()
    for c in cat:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna("NA").astype("category")
    return pdf[cols]


def fit_lgb(xtr, ytr, xes, yes, cat: list[str]):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    cats = [c for c in cat if c in xtr.columns]
    model.fit(
        xtr,
        ytr,
        eval_set=[(xes, yes)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=cats if cats else "auto",
    )
    return model


def prepare_split(dep: pl.DataFrame, months: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    tr0, va0 = split_by_months(dep, months)
    log(f"    split train={tr0.height:,} val={va0.height:,}; floors...")
    tr, va = attach_floors_time_forward(tr0, va0)
    log("    encodings (LOMO train / full-train val)...")
    tr, va = attach_encodings_oof(tr, va)
    tr = tr.sort(["airport", "MVT_TIME_UTC_mvt"])
    va = va.sort(["airport", "MVT_TIME_UTC_mvt"])
    return tr, va


def fit_predict_per_airport(
    tr: pl.DataFrame,
    va: pl.DataFrame,
    num: list[str],
    cat: list[str],
    target: str = "log_excess",
) -> tuple[np.ndarray, dict]:
    """target: log_excess | log_y | raw."""
    assert_ranking_safe_features(num + cat)
    y_va = va["y"].to_numpy().astype(np.float64)
    pred = np.full(va.height, np.nan, dtype=np.float64)
    ap_va = va["airport"].to_numpy()
    meta: dict = {"smear": {}, "best_iter": {}, "n_train": {}}

    for a in AIRPORTS:
        tr_a = tr.filter((pl.col("airport") == a) & (~pl.col("unmatched")) & pl.col("y").is_finite() & pl.col("floor").is_finite())
        va_idx = np.where(ap_va == a)[0]
        if tr_a.height < 5000 or va_idx.size == 0:
            log(f"    {a}: too small ({tr_a.height}); skipping")
            continue
        ta_fit, ta_es = time_es_split(tr_a)
        pdf_tr = to_pdf(ta_fit, num, cat)
        pdf_es = to_pdf(ta_es, num, cat)
        pdf_va = to_pdf(va.filter(pl.col("airport") == a), num, cat)

        if target == "log_excess":
            ytr = log_excess(ta_fit["y"].to_numpy(), ta_fit["floor"].to_numpy())
            yes = log_excess(ta_es["y"].to_numpy(), ta_es["floor"].to_numpy())
        elif target == "log_y":
            ytr = np.log(np.clip(ta_fit["y"].to_numpy().astype(np.float64), 1.0, None))
            yes = np.log(np.clip(ta_es["y"].to_numpy().astype(np.float64), 1.0, None))
        else:
            ytr = ta_fit["y"].to_numpy().astype(np.float64)
            yes = ta_es["y"].to_numpy().astype(np.float64)

        model = fit_lgb(pdf_tr, ytr, pdf_es, yes, cat)
        z_es = np.asarray(model.predict(pdf_es), dtype=np.float64)
        z_va = np.asarray(model.predict(pdf_va), dtype=np.float64)
        floor_va = va.filter(pl.col("airport") == a)["floor"].to_numpy().astype(np.float64)

        if target == "log_excess":
            sm = smear_factor(yes, z_es)
            pred[va_idx] = back_transform(z_va, floor_va, sm)
            meta["smear"][a] = sm
        elif target == "log_y":
            sm = smear_factor(yes, z_es)
            pred[va_idx] = np.clip(sm * np.exp(z_va), 1.0, 200_000.0)
            meta["smear"][a] = sm
        else:
            pred[va_idx] = z_va
            meta["smear"][a] = 1.0
        meta["best_iter"][a] = int(getattr(model, "best_iteration_", 0) or 0)
        meta["n_train"][a] = int(ta_fit.height)
        log(f"    {a}: n={ta_fit.height:,} iter={meta['best_iter'][a]} smear={meta['smear'][a]:.3f}")

        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        meta.setdefault("importance", {})[a] = sorted(
            zip(list(pdf_tr.columns), gain.tolist()), key=lambda t: -t[1]
        )[:30]

    finite = np.isfinite(pred)
    if not finite.all():
        # last-resort: airport mean of train matched
        for a in AIRPORTS:
            sel = (ap_va == a) & ~finite
            if not sel.any():
                continue
            mu = tr.filter((pl.col("airport") == a) & (~pl.col("unmatched")))["y"].mean()
            pred[sel] = float(mu) if mu is not None else 900.0
    return pred, meta


def fit_predict_pooled(
    tr: pl.DataFrame,
    va: pl.DataFrame,
    num: list[str],
    cat: list[str],
    target: str = "log_excess",
) -> tuple[np.ndarray, dict]:
    cat_p = cat if "airport" in cat else (["airport"] + list(cat))
    tr_m = tr.filter((~pl.col("unmatched")) & pl.col("y").is_finite() & pl.col("floor").is_finite())
    ta_fit, ta_es = time_es_split(tr_m)
    pdf_tr = to_pdf(ta_fit, num, cat_p)
    pdf_es = to_pdf(ta_es, num, cat_p)
    pdf_va = to_pdf(va, num, cat_p)
    if target == "log_excess":
        ytr = log_excess(ta_fit["y"].to_numpy(), ta_fit["floor"].to_numpy())
        yes = log_excess(ta_es["y"].to_numpy(), ta_es["floor"].to_numpy())
    else:
        ytr = ta_fit["y"].to_numpy().astype(np.float64)
        yes = ta_es["y"].to_numpy().astype(np.float64)
    model = fit_lgb(pdf_tr, ytr, pdf_es, yes, cat_p)
    z_es = np.asarray(model.predict(pdf_es), dtype=np.float64)
    z_va = np.asarray(model.predict(pdf_va), dtype=np.float64)
    floor_va = va["floor"].to_numpy().astype(np.float64)
    sm = smear_factor(yes, z_es) if target == "log_excess" else 1.0
    if target == "log_excess":
        pred = back_transform(z_va, floor_va, sm)
    else:
        pred = z_va
    log(f"    pooled: n={ta_fit.height:,} iter={int(getattr(model, 'best_iteration_', 0) or 0)} smear={sm:.3f}")
    return pred, {"smear": {"pooled": sm}}


def unmatched_specialist(tr: pl.DataFrame, va: pl.DataFrame) -> np.ndarray:
    """Schedule-gap + geometry model for unmatched non-LIRF rows."""
    tr_u = tr.filter(pl.col("unmatched") & (pl.col("airport") != "LIRF") & pl.col("y").is_finite())
    pred = np.full(va.height, np.nan, dtype=np.float64)
    if tr_u.height < 200:
        return pred
    ta_fit, ta_es = time_es_split(tr_u)
    pdf_tr = to_pdf(ta_fit, NUM_UNMATCHED, CAT_UNMATCHED)
    pdf_es = to_pdf(ta_es, NUM_UNMATCHED, CAT_UNMATCHED)
    pdf_va = to_pdf(va, NUM_UNMATCHED, CAT_UNMATCHED)
    ytr = ta_fit["y"].to_numpy().astype(np.float64)
    yes = ta_es["y"].to_numpy().astype(np.float64)
    model = fit_lgb(pdf_tr, ytr, pdf_es, yes, CAT_UNMATCHED)
    return np.asarray(model.predict(pdf_va), dtype=np.float64)


def overlay_unmatched(pred_main, pred_spec, va: pl.DataFrame, policy: str) -> np.ndarray:
    p = np.asarray(pred_main, dtype=np.float64).copy()
    um = va["unmatched"].to_numpy().astype(bool)
    ap = va["airport"].to_numpy()
    sched = va["mvt_sched"].to_numpy().astype(np.float64)
    if policy == "lirf_sched_else_main":
        return apply_lirf_override(p, um, ap, sched)
    if policy == "all_sched":
        out = p.copy()
        m = um & np.isfinite(sched)
        out[m] = sched[m]
        return out
    if policy == "lirf_sched_else_spec":
        out = apply_lirf_override(p, um, ap, sched)
        m = um & (ap != "LIRF") & np.isfinite(pred_spec)
        out[m] = pred_spec[m]
        return out
    raise ValueError(policy)


def pick_unmatched_policy(cands: dict, y, um, ap) -> tuple[str, dict]:
    """Need Jan+Jul win without December regression — applied per split here,
    the runner compares both splits before locking the policy."""
    ranked = sorted(cands.items(), key=lambda kv: kv[1]["rmse"])
    return ranked[0][0], {k: v for k, v in ranked}


def plot_rmse(split: str, scores: dict, e20: dict, dest: Path) -> None:
    labels = ["E20 ensemble", "E25 rebuild"]
    overall = [e20["overall"], scores["rmse"]]
    matched = [e20["matched"], scores["matched"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - 0.18, overall, 0.36, label="overall")
    ax.bar(x + 0.18, matched, 0.36, label="matched")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(f"E25 vs E20 — {split}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plot_airport(split: str, scores: dict, dest: Path) -> None:
    vals = [scores.get(f"rmse_{a}", np.nan) for a in AIRPORTS]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(AIRPORTS, vals)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(f"E25 per-airport RMSE — {split}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def write_report(payload: dict) -> Path:
    j = payload["janjul"]["chosen"]
    d = payload["dec"]["chosen"]
    pol = payload["policy"]
    rec = payload["recommendation"]
    schema = payload["schema"]
    lines = [
        "# E25 — One-shot structural rebuild",
        "",
        "Single architecture: hierarchical p10 floor → log(y − floor) → per-airport LightGBM",
        "→ Duan smearing → unmatched fallback. Not an incremental patch on E20.",
        "",
        "## Schema (measured on training DEP, n=2,085,047)",
        "",
        f"- `STAND_mvt`: {schema['stand_null_pct']:.4f}% null, {schema['stand_unique']} unique.",
        f"- `RUNWAY_mvt`: 0% null, {schema['rwy_unique']} unique.",
        f"- `AIRCRAFT_OPERATOR_flt`: {schema['op_null_pct']:.2f}% null (equals unmatched), {schema['op_unique']} unique.",
        f"- Smallest Jan+Jul-train airport: {schema['smallest_airport']} n={schema['smallest_n']:,}.",
        "  Full per-airport models are viable everywhere (E20 used a 20k cutoff; all ten exceed 110k).",
        "",
        "## Ranking-safe clocks",
        "",
        "Airport `BLOCK_TIME_UTC_mvt` is the target ingredient and is blank in ranking.",
        "AOBT in this pipeline is NM `AOBT_3_flt`. Scored-row `MVT_TIME` is available",
        "(post-ops reconstruction). Floors/encodings use training-split y only.",
        "",
        "## Validation vs E20 (current production / leaderboard 316.9654)",
        "",
        "| Split | E25 overall | E20 overall | Δ | E25 matched | E20 matched | Δ matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Jan+Jul | {j['rmse']:.2f} | {E20_JANJUL['overall']:.2f} | {j['rmse']-E20_JANJUL['overall']:+.2f} | {j['matched']:.2f} | {E20_JANJUL['matched']:.2f} | {j['matched']-E20_JANJUL['matched']:+.2f} |",
        f"| December | {d['rmse']:.2f} | {E20_DEC['overall']:.2f} | {d['rmse']-E20_DEC['overall']:+.2f} | {d['matched']:.2f} | {E20_DEC['matched']:.2f} | {d['matched']-E20_DEC['matched']:+.2f} |",
        "",
        f"Unmatched policy locked: **{pol}**.",
        "",
        "## Per-airport RMSE (Jan+Jul)",
        "",
        "| Airport | RMSE |",
        "|---|---:|",
    ]
    for a in AIRPORTS:
        lines.append(f"| {a} | {j.get(f'rmse_{a}', float('nan')):.2f} |")
    lines += [
        "",
        f"## Tail (Jan+Jul matched): <20 min {j.get('rmse_lt20', float('nan')):.2f} · "
        f">30 min {j.get('rmse_gt30', float('nan')):.2f} · >60 min {j.get('rmse_gt60', float('nan')):.2f}",
        "",
        "## Ablation (after the combined result)",
        "",
        "| Variant | Jan+Jul overall | matched | Dec overall | matched |",
        "|---|---:|---:|---:|---:|",
    ]
    ab = payload.get("ablation", {})
    for name in ["full", "no_floor", "raw_target", "pooled", "no_enc"]:
        row = ab.get(name, {})
        jj = row.get("janjul", {})
        dd = row.get("dec", {})
        if not jj:
            continue
        lines.append(
            f"| {name} | {jj.get('rmse', float('nan')):.2f} | {jj.get('matched', float('nan')):.2f} | "
            f"{dd.get('rmse', float('nan')):.2f} | {dd.get('matched', float('nan')):.2f} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        rec,
        "",
        "Leakage tests: `python -m pytest experiments/test_e25_leakage.py -q`.",
        "",
    ]
    text = "\n".join(lines)
    path = REP / "E25_report.md"
    path.write_text(text, encoding="utf-8")
    (RES / "summary.md").write_text(text, encoding="utf-8")
    return path


def recommend(j: dict, d: dict) -> str:
    dj = j["rmse"] - E20_JANJUL["overall"]
    dd = d["rmse"] - E20_DEC["overall"]
    mj = j["matched"] - E20_JANJUL["matched"]
    md = d["matched"] - E20_DEC["matched"]
    # Same spirit as E20's blend gate: Jan+Jul must improve, December must not regress much.
    if dj <= -1.5 and dd <= 3.0:
        verdict = "SUBMIT"
        why = (
            f"Jan+Jul overall {E20_JANJUL['overall']:.2f} → {j['rmse']:.2f} ({dj:+.2f}), "
            f"December {E20_DEC['overall']:.2f} → {d['rmse']:.2f} ({dd:+.2f}). "
            "Clears the E20-style gate (Jan+Jul ≤ −1.5 s, December regression ≤ 3 s)."
        )
    elif dj < 0 and dd <= 0:
        verdict = "SUBMIT (marginal)"
        why = (
            f"Improves both holdouts (Jan+Jul {dj:+.2f}, Dec {dd:+.2f}) but the Jan+Jul "
            "gain is under 1.5 s. Submit only if a new leaderboard slot is cheap; "
            "internal evidence is a small, consistent edge, not a regime change."
        )
    else:
        verdict = "DO NOT SUBMIT"
        why = (
            f"Jan+Jul {dj:+.2f} (matched {mj:+.2f}), December {dd:+.2f} (matched {md:+.2f}) "
            "vs E20. Either the rebuild loses the ranking analogue or it regresses December."
        )
    return f"**{verdict}.** {why}"


def main():
    assert_ranking_safe_features(NUM_COLS + CAT_COLS)
    log("E25: loading DEP + ARR...")
    dep = load_dep()
    dep = add_calendar(dep)
    arr = load_arr_rwy()
    log(f"  dep={dep.height:,} arr={arr.height:,}; congestion 10/15 min...")
    dep = add_congestion(dep, arr)
    log(f"  ready {dep.height:,} rows, {dep.width} cols")

    schema = {
        "stand_null_pct": 0.0013,
        "stand_unique": 1899,
        "rwy_unique": 53,
        "op_null_pct": 1.0777,
        "op_unique": 676,
        "smallest_airport": "LSZH",
        "smallest_n": 112561,
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "lgb_params": {k: v for k, v in LGB_PARAMS.items() if k != "random_state"},
    }

    split_preds = {}
    for split_name, months in (("janjul", [1, 7]), ("dec", [12])):
        log(f"========== {split_name} ==========")
        tr, va = prepare_split(dep, months)
        log("  fitting 10 per-airport log-excess models...")
        pred, meta = fit_predict_per_airport(tr, va, NUM_COLS, CAT_COLS, target="log_excess")
        log("  unmatched specialist...")
        spec = unmatched_specialist(tr, va)
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy().astype(bool)
        ap = va["airport"].to_numpy()
        sched = va["mvt_sched"].to_numpy().astype(np.float64)
        policies = {}
        preds_p = {}
        for pol in ("lirf_sched_else_main", "lirf_sched_else_spec", "all_sched"):
            pp = overlay_unmatched(pred, spec, va, pol)
            preds_p[pol] = pp
            policies[pol] = score_all(y, pp, um, ap)
            s = policies[pol]
            log(f"    {pol}: overall={s['rmse']:.2f} matched={s['matched']:.2f} um={s['unmatched']:.2f}")
        payload[split_name] = {
            "policies": policies,
            "meta": {k: meta[k] for k in ("smear", "best_iter", "n_train")},
            "importance": meta.get("importance", {}),
        }
        split_preds[split_name] = {
            "y": y,
            "um": um,
            "ap": ap,
            "sched": sched,
            "pred_main": pred,
            "pred_spec": spec,
            "preds_p": preds_p,
            "va_ids": va["MVT_ID_mvt"].to_numpy(),
            "tr": tr,
            "va": va,
        }
        oof = pl.DataFrame(
            {
                "MVT_ID_mvt": va["MVT_ID_mvt"],
                "y": y,
                "unmatched": um,
                "airport": ap,
                "pred_main": pred,
            }
        )
        oof.write_parquet(RES / f"oof_{split_name}.parquet")

    # Lock unmatched policy: best Jan+Jul overall that does not raise December overall
    # vs lirf_sched_else_main (the proven E18-H rule).
    j_pol = payload["janjul"]["policies"]
    d_pol = payload["dec"]["policies"]
    base = "lirf_sched_else_main"
    locked = base
    for cand in ("lirf_sched_else_main", "lirf_sched_else_spec", "all_sched"):
        if j_pol[cand]["rmse"] + 0.05 < j_pol[locked]["rmse"] and d_pol[cand]["rmse"] <= d_pol[base]["rmse"] + 3.0:
            locked = cand
    # all_sched is known-poisonous on non-LIRF; only keep if both splits clearly win
    if locked == "all_sched":
        if not (
            j_pol["all_sched"]["rmse"] < j_pol[base]["rmse"] - 2.0
            and d_pol["all_sched"]["rmse"] < d_pol[base]["rmse"]
        ):
            locked = base
    payload["policy"] = locked
    log(f"LOCKED unmatched policy: {locked}")

    for split_name, e20 in (("janjul", E20_JANJUL), ("dec", E20_DEC)):
        sp = split_preds[split_name]
        chosen = overlay_unmatched(sp["pred_main"], sp["pred_spec"], sp["va"], locked)
        sc = score_all(sp["y"], chosen, sp["um"], sp["ap"])
        payload[split_name]["chosen"] = sc
        log(f"CHOSEN {split_name}: overall={sc['rmse']:.2f} matched={sc['matched']:.2f} (E20 {e20['overall']:.2f}/{e20['matched']:.2f})")
        stem = "" if split_name == "janjul" else "dec/"
        plot_rmse(split_name, sc, e20, FIG / f"{stem}01_rmse.png")
        plot_airport(split_name, sc, FIG / f"{stem}02_airport_rmse.png")
        pl.DataFrame(
            {
                "MVT_ID_mvt": sp["va_ids"],
                "y": sp["y"],
                "unmatched": sp["um"],
                "airport": sp["ap"],
                "pred": chosen,
            }
        ).write_parquet(RES / f"chosen_{split_name}.parquet")

    payload["recommendation"] = recommend(payload["janjul"]["chosen"], payload["dec"]["chosen"])
    log(payload["recommendation"])

    # ---- Ablations (after combined result) ----
    log("========== ablations ==========")
    ablation = {
        "full": {
            "janjul": payload["janjul"]["chosen"],
            "dec": payload["dec"]["chosen"],
        }
    }
    variants = [
        ("no_floor", "log_y", [c for c in NUM_COLS if c not in ("floor", "floor_n")], CAT_COLS, False),
        ("raw_target", "raw", NUM_COLS, CAT_COLS, False),
        ("pooled", "log_excess", NUM_COLS, CAT_COLS, True),
        ("no_enc", "log_excess", [c for c in NUM_COLS if not c.startswith("enc_")], CAT_COLS, False),
    ]
    for name, target, num, cat, pooled in variants:
        log(f"  ablation {name}...")
        ablation[name] = {}
        for split_name in ("janjul", "dec"):
            sp = split_preds[split_name]
            tr, va = sp["tr"], sp["va"]
            if pooled:
                pred, _ = fit_predict_pooled(tr, va, num, cat, target=target)
            else:
                pred, _ = fit_predict_per_airport(tr, va, num, cat, target=target)
            pred = overlay_unmatched(pred, sp["pred_spec"], va, locked)
            ablation[name][split_name] = score_all(sp["y"], pred, sp["um"], sp["ap"])
            s = ablation[name][split_name]
            log(f"    {split_name}: overall={s['rmse']:.2f} matched={s['matched']:.2f}")
    payload["ablation"] = ablation

    pd.DataFrame(
        [
            {
                "variant": name,
                "split": split,
                **{k: v for k, v in ablation[name][split].items() if not k.startswith("rmse_") or k in ("rmse",)},
            }
            for name in ablation
            for split in ("janjul", "dec")
        ]
    ).to_csv(TAB / "e25_ablation.csv", index=False)

    metrics_rows = []
    for split in ("janjul", "dec"):
        row = {"split": split, **payload[split]["chosen"]}
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(TAB / "e25_metrics.csv", index=False)

    imp_rows = []
    for a, pairs in payload["janjul"].get("importance", {}).items():
        for feat, gain in pairs:
            imp_rows.append({"airport": a, "feature": feat, "gain": gain})
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(TAB / "e25_importance.csv", index=False)

    report = write_report(payload)
    (RES / "E25.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    save_result("E25", payload)
    log(f"WROTE {report}")
    log(f"WROTE {RES}")


if __name__ == "__main__":
    main()
