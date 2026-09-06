"""E16-A: causal airport/network disruption-state features.

Hypothesis: AOBT−EOBT says this flight is late; it does not say whether the
airport is in a broader push-delay disruption where long taxi becomes likely.

Phase 1 — small causal feature family (other flights only, known by MVT).
Phase 2 — does it distinguish the E14 tail after controlling AOBT−EOBT?
Phase 3 — only if Phase 2 has signal: add features to frozen E14 L2 residual.

Frozen: P_cal, train/val split, LIRF unmatched fallback, LightGBM capacity.
Fits: training_*.parquet only. Jan+Jul primary; December stress.
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
    AIRPORTS,
    ROOT,
    TRAIN_FILES,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    mae,
    rmse,
    save_result,
    sec,
    split_by_months,
)
from run_e12_e9 import CAT_COLS, NUM_COLS as BASE_NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402

OUT = ROOT / "analysis" / "E16A"
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

SEED = 1
TAIL_S = 1800.0
HOUR_S = 3600.0
BULK_S = 1200.0
BASELINE_MATCHED = 256.46

# Small family actually fed to LightGBM if Phase 2 passes.
MODEL_DISRUPT_COLS = [
    "dis_state_30m",  # airport_recent_push_delay_state
    "dis_frac20_30m",
    "dis_frac30_30m",
    "dis_n_dly20_60m",
    "dis_state_vs_hour",
    "arr_delay_mean_30m",
]


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


def override_mask(unmatched, airport, mvt_sched):
    return unmatched & (airport == "LIRF") & np.isfinite(mvt_sched)


def apply_override(pred, mask, mvt_sched):
    out = np.asarray(pred, dtype=np.float64).copy()
    out[mask] = mvt_sched[mask]
    return out


def to_xy(df: pl.DataFrame, extra: list[str] | tuple[str, ...] = ()):
    num = list(BASE_NUM_COLS) + [c for c in extra if c not in BASE_NUM_COLS]
    pdf = df.select(num + CAT_COLS + ["y"]).to_pandas()
    pdf["unmatched"] = pdf["unmatched"].astype(np.int8)
    pdf["type_null"] = pdf["type_null"].astype(np.int8)
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype("category")
    return pdf[num + CAT_COLS], pdf["y"].to_numpy(dtype=np.float64)


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
        "rmse_lt20_matched": rmse(y[lt20], p[lt20]) if lt20.any() else float("nan"),
        "rmse_gt30_matched": rmse(y[gt30], p[gt30]) if gt30.any() else float("nan"),
        "rmse_gt60_matched": rmse(y[gt60], p[gt60]) if gt60.any() else float("nan"),
        "sse_share_gt30_matched": (sse_gt30 / sse) if sse > 0 else float("nan"),
        "resid_mean_matched": float(np.mean(r[matched])) if matched.any() else float("nan"),
        "rmse_unmatched": rmse(y[ok & um], p[ok & um]) if (ok & um).any() else float("nan"),
    }


def corr_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def ns_array(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype("datetime64[ns]").astype(np.int64)


# ---------------------------------------------------------------------------
# Phase 1: causal disruption features
# ---------------------------------------------------------------------------

def _window_from_prefix(push_t, push_v, query_t, own_t, own_v, own_ok, w_ns: int) -> dict:
    """Stats of other pushes with AOBT in (T−w, T]. Prefix sums; self excluded."""
    v = np.asarray(push_v, dtype=np.float64)
    finite = np.isfinite(v)
    vz = np.where(finite, v, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(vz)])
    cn = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    c10 = np.concatenate([[0], np.cumsum((finite & (v > 600)).astype(np.int64))])
    c20 = np.concatenate([[0], np.cumsum((finite & (v > 1200)).astype(np.int64))])
    c30 = np.concatenate([[0], np.cumsum((finite & (v > 1800)).astype(np.int64))])
    hi = np.searchsorted(push_t, query_t, side="right")
    lo = np.searchsorted(push_t, query_t - w_ns, side="right")
    n = (cn[hi] - cn[lo]).astype(np.float64)
    s = cs[hi] - cs[lo]
    n10 = (c10[hi] - c10[lo]).astype(np.float64)
    n20 = (c20[hi] - c20[lo]).astype(np.float64)
    n30 = (c30[hi] - c30[lo]).astype(np.float64)
    in_win = (
        own_ok
        & np.isfinite(own_t)
        & np.isfinite(own_v)
        & (own_t <= query_t)
        & (own_t > query_t - w_ns)
    )
    n = n - in_win.astype(np.float64)
    s = s - np.where(in_win, own_v, 0.0)
    n10 = n10 - (in_win & (own_v > 600)).astype(np.float64)
    n20 = n20 - (in_win & (own_v > 1200)).astype(np.float64)
    n30 = n30 - (in_win & (own_v > 1800)).astype(np.float64)
    n = np.maximum(n, 0.0)
    mean = np.divide(s, n, out=np.full_like(s, np.nan), where=n >= 1)
    f10 = np.divide(n10, n, out=np.full_like(s, np.nan), where=n >= 1)
    f20 = np.divide(n20, n, out=np.full_like(s, np.nan), where=n >= 1)
    f30 = np.divide(n30, n, out=np.full_like(s, np.nan), where=n >= 1)
    return {"n": n, "mean": mean, "f10": f10, "f20": f20, "f30": f30, "n20": n20, "n30": n30}


def add_push_disruption(dep: pl.DataFrame) -> pl.DataFrame:
    """Airport push-delay state from OTHER flights whose AOBT is already known at MVT.

    Does not use BLOCK, TAXITIME, or any future event of the scored flight.
    Other flights contribute AOBT−EOBT only (known at their AOBT).
    """
    n = dep.height
    ap = dep["airport"].to_numpy()
    q_t = ns_array(dep["MVT_TIME_UTC_mvt"])
    own_t = ns_array(dep["AOBT_3_flt"])
    own_v = dep["aobt_eobt"].to_numpy().astype(np.float64)
    own_ok = (~dep["unmatched"].to_numpy()) & np.isfinite(own_v)

    push = dep.filter((~pl.col("unmatched")) & pl.col("aobt_eobt").is_finite())
    p_ap = push["airport"].to_numpy()
    p_t = ns_array(push["AOBT_3_flt"])
    p_v = push["aobt_eobt"].to_numpy().astype(np.float64)

    windows = {"15": 15 * 60 * 10**9, "30": 30 * 60 * 10**9, "60": 60 * 60 * 10**9}
    out = {f"{k}_{w}": np.full(n, np.nan) for w in windows for k in ("n", "mean", "f10", "f20", "f30", "n20")}

    for a in AIRPORTS:
        qidx = np.where(ap == a)[0]
        pidx = np.where(p_ap == a)[0]
        if qidx.size == 0 or pidx.size == 0:
            continue
        order = np.argsort(p_t[pidx], kind="mergesort")
        pt = p_t[pidx][order]
        pv = p_v[pidx][order]
        qt = q_t[qidx]
        ot = own_t[qidx]
        ov = own_v[qidx]
        ok = own_ok[qidx]
        for wname, wns in windows.items():
            st = _window_from_prefix(pt, pv, qt, ot, ov, ok, wns)
            out[f"n_{wname}"][qidx] = st["n"]
            out[f"mean_{wname}"][qidx] = st["mean"]
            out[f"f10_{wname}"][qidx] = st["f10"]
            out[f"f20_{wname}"][qidx] = st["f20"]
            out[f"f30_{wname}"][qidx] = st["f30"]
            out[f"n20_{wname}"][qidx] = st["n20"]

    extra = pl.DataFrame(
        {
            "MVT_ID_mvt": dep["MVT_ID_mvt"],
            "dis_n_15m": out["n_15"],
            "dis_n_30m": out["n_30"],
            "dis_n_60m": out["n_60"],
            "dis_state_15m": out["mean_15"],
            "dis_state_30m": out["mean_30"],  # airport_recent_push_delay_state
            "dis_state_60m": out["mean_60"],
            "dis_frac10_30m": out["f10_30"],
            "dis_frac20_15m": out["f20_15"],
            "dis_frac20_30m": out["f20_30"],
            "dis_frac20_60m": out["f20_60"],
            "dis_frac30_30m": out["f30_30"],
            "dis_frac30_60m": out["f30_60"],
            "dis_n_dly20_30m": out["n20_30"],
            "dis_n_dly20_60m": out["n20_60"],
        }
    )
    return dep.join(extra, on="MVT_ID_mvt", how="left")


def add_rolling_quantiles(dep: pl.DataFrame) -> pl.DataFrame:
    """Median/p90 of recent pushes via event-time rolling + asof (last push at or before MVT).

    One-point self contamination is possible; mean/fractions above are exact.
    """
    pushes = (
        dep.filter((~pl.col("unmatched")) & pl.col("aobt_eobt").is_finite())
        .select("airport", pl.col("AOBT_3_flt").alias("aobt"), "aobt_eobt")
        .sort(["airport", "aobt"])
    )
    rolled = pushes.rolling(index_column="aobt", period="30m", group_by="airport").agg(
        pl.col("aobt_eobt").median().alias("dis_med_ae_30m"),
        pl.col("aobt_eobt").quantile(0.75).alias("dis_p75_ae_30m"),
        pl.col("aobt_eobt").quantile(0.90).alias("dis_p90_ae_30m"),
    )
    keys = dep.select("MVT_ID_mvt", "airport", "MVT_TIME_UTC_mvt").sort(["airport", "MVT_TIME_UTC_mvt"])
    j = keys.join_asof(
        rolled.sort(["airport", "aobt"]),
        left_on="MVT_TIME_UTC_mvt",
        right_on="aobt",
        by="airport",
        strategy="backward",
    )
    return dep.join(j.select("MVT_ID_mvt", "dis_med_ae_30m", "dis_p75_ae_30m", "dis_p90_ae_30m"), on="MVT_ID_mvt", how="left")


def load_arr_delay() -> pl.DataFrame:
    cols = ["ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    return (
        pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .with_columns(
            pl.col("ADES_mvt").alias("airport"),
            sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("arr_sched_delay"),
        )
        .select("airport", "MVT_TIME_UTC_mvt", "arr_sched_delay")
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )


def add_arrival_delay_state(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Mean landing-vs-schedule delay of arrivals already landed by scored MVT."""
    n = dep.height
    ap = dep["airport"].to_numpy()
    q_t = ns_array(dep["MVT_TIME_UTC_mvt"])
    a_ap = arr["airport"].to_numpy()
    a_t = ns_array(arr["MVT_TIME_UTC_mvt"])
    a_v = arr["arr_sched_delay"].to_numpy().astype(np.float64)
    mean30 = np.full(n, np.nan)
    n30 = np.full(n, np.nan)
    w = 30 * 60 * 10**9
    for a in AIRPORTS:
        qidx = np.where(ap == a)[0]
        aidx = np.where(a_ap == a)[0]
        if qidx.size == 0 or aidx.size == 0:
            continue
        order = np.argsort(a_t[aidx], kind="mergesort")
        tt = a_t[aidx][order]
        vv = a_v[aidx][order]
        finite = np.isfinite(vv)
        vz = np.where(finite, vv, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(vz)])
        cn = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
        hi = np.searchsorted(tt, q_t[qidx], side="right")
        lo = np.searchsorted(tt, q_t[qidx] - w, side="right")
        nn = (cn[hi] - cn[lo]).astype(np.float64)
        ss = cs[hi] - cs[lo]
        n30[qidx] = nn
        mean30[qidx] = np.divide(ss, nn, out=np.full_like(ss, np.nan), where=nn >= 1)
    extra = pl.DataFrame(
        {
            "MVT_ID_mvt": dep["MVT_ID_mvt"],
            "arr_delay_mean_30m": mean30,
            "arr_n_30m": n30,
        }
    )
    return dep.join(extra, on="MVT_ID_mvt", how="left")


def attach_hour_baseline(train: pl.DataFrame, val: pl.DataFrame, col="dis_state_30m") -> pl.DataFrame:
    """Airport×hour climatology of disruption state, train-split only."""
    hist = train.group_by(["airport", "hour"]).agg(
        pl.col(col).mean().alias("_dis_hour_mean"),
        pl.col(col).len().alias("_dis_hour_n"),
    )
    out = val.join(hist, on=["airport", "hour"], how="left")
    # Anomaly vs train airport×hour mean (difference, not ratio: AOBT−EOBT can be ≤ 0).
    return out.with_columns((pl.col(col) - pl.col("_dis_hour_mean")).alias("dis_state_vs_hour"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def fit_residual(tr: pl.DataFrame, va: pl.DataFrame, extra: list[str]):
    p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    geo = va["geo_mean"].to_numpy().astype(float)
    fb = airport_mean_fallback(tr, va)
    p_cal = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)

    p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
    p_tr = fill_with_fallback(
        fill_with_fallback(p_tr, tr["geo_mean"].to_numpy().astype(float)),
        airport_mean_fallback(tr, tr),
    )
    tr_res = tr.with_columns(pl.Series("p_cal", p_tr))
    tr_fit, tr_es = time_es_split(tr_res)
    xtr, _ = to_xy(tr_fit, extra)
    ytr = tr_fit["y"].to_numpy() - tr_fit["p_cal"].to_numpy()
    xes, _ = to_xy(tr_es, extra)
    yes = tr_es["y"].to_numpy() - tr_es["p_cal"].to_numpy()
    model = fit_lgb(xtr, ytr, xes, yes, seed=SEED)
    xva, _ = to_xy(va, extra)
    resid_hat = np.asarray(model.predict(xva), dtype=np.float64)
    pred = apply_override(
        p_cal + resid_hat,
        override_mask(va["unmatched"].to_numpy(), va["airport"].to_numpy(), va["mvt_sched"].to_numpy().astype(float)),
        va["mvt_sched"].to_numpy().astype(float),
    )
    return p_cal, pred, model


def prepare_split(dep: pl.DataFrame, months: list[int]) -> dict:
    tr0, va0 = split_by_months(dep, months)
    tabs = geometry_tables(tr0)
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)
    tr = attach_hour_baseline(tr, tr)
    va = attach_hour_baseline(tr, va)
    return {"tr": tr, "va": va}


def qcut_labels(x: np.ndarray, q=8, prefix="Q"):
    s = pd.Series(x)
    try:
        cat = pd.qcut(s, q=q, labels=[f"{prefix}{i}" for i in range(1, q + 1)], duplicates="drop")
    except ValueError:
        cat = pd.qcut(s.rank(method="first"), q=q, labels=[f"{prefix}{i}" for i in range(1, q + 1)])
    return cat.astype(str).to_numpy()


def group_diag(pdf: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    total_sse = float((pdf["resid"] ** 2).sum()) if len(pdf) else 0.0
    for g, sub in pdf.groupby(key, dropna=False, observed=False):
        y = sub["y"].to_numpy()
        p = sub["pred"].to_numpy()
        r = sub["resid"].to_numpy()
        n = len(sub)
        sse = float(np.sum(r ** 2))
        rows.append(
            {
                key: g,
                "n": n,
                "mean_y": float(np.mean(y)),
                "mean_pred": float(np.mean(p)),
                "mean_aobt_eobt": float(np.nanmean(sub["aobt_eobt"].to_numpy())),
                "mean_dis": float(np.nanmean(sub["dis_state_30m"].to_numpy())),
                "frac_gt30": float(np.mean(y > TAIL_S)),
                "frac_gt60": float(np.mean(y > HOUR_S)),
                "resid_mean": float(np.mean(r)),
                "resid_rmse": float(np.sqrt(np.mean(r ** 2))),
                "y_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "sse": sse,
                "sse_share": sse / total_sse if total_sse else 0.0,
            }
        )
    return pd.DataFrame(rows)


def phase2_tables(va: pl.DataFrame, y, pred, unmatched) -> dict:
    m = (~np.asarray(unmatched, dtype=bool)) & np.isfinite(y) & np.isfinite(pred)
    pdf = pd.DataFrame(
        {
            "y": y[m],
            "pred": pred[m],
            "resid": y[m] - pred[m],
            "aobt_eobt": va["aobt_eobt"].to_numpy()[m],
            "dis_state_30m": va["dis_state_30m"].to_numpy()[m],
            "dis_frac20_30m": va["dis_frac20_30m"].to_numpy()[m],
            "dis_frac30_30m": va["dis_frac30_30m"].to_numpy()[m],
            "dis_n_dly20_60m": va["dis_n_dly20_60m"].to_numpy()[m],
            "dis_med_ae_30m": va["dis_med_ae_30m"].to_numpy()[m],
            "arr_delay_mean_30m": va["arr_delay_mean_30m"].to_numpy()[m],
            "airport": va["airport"].to_numpy()[m],
        }
    )
    pdf["dis_q"] = qcut_labels(pdf["dis_state_30m"].to_numpy(), 8, "Q")
    pdf["ae_q"] = qcut_labels(pdf["aobt_eobt"].to_numpy(), 4, "AE")
    pdf["frac_q"] = qcut_labels(pdf["dis_frac20_30m"].to_numpy(), 8, "Q")

    by_dis = group_diag(pdf, "dis_q").sort_values("dis_q")
    by_frac = group_diag(pdf, "frac_q").sort_values("frac_q")
    by_ae = group_diag(pdf, "ae_q").sort_values("ae_q")

    two = []
    for (ae, dq), sub in pdf.groupby(["ae_q", "dis_q"], dropna=False):
        yv = sub["y"].to_numpy()
        r = sub["resid"].to_numpy()
        two.append(
            {
                "ae_q": ae,
                "dis_q": dq,
                "n": len(sub),
                "mean_y": float(np.mean(yv)),
                "frac_gt30": float(np.mean(yv > TAIL_S)),
                "frac_gt60": float(np.mean(yv > HOUR_S)),
                "resid_mean": float(np.mean(r)),
                "resid_rmse": float(np.sqrt(np.mean(r ** 2))),
                "mean_aobt_eobt": float(np.nanmean(sub["aobt_eobt"])),
                "mean_dis": float(np.nanmean(sub["dis_state_30m"])),
            }
        )
    two_way = pd.DataFrame(two).sort_values(["ae_q", "dis_q"])

    # high vs low disruption
    q1 = pdf[pdf["dis_q"] == "Q1"]
    q8 = pdf[pdf["dis_q"] == "Q8"]
    high_low = {
        "n_q1": int(len(q1)),
        "n_q8": int(len(q8)),
        "mean_y_q1": float(q1["y"].mean()) if len(q1) else float("nan"),
        "mean_y_q8": float(q8["y"].mean()) if len(q8) else float("nan"),
        "frac_gt30_q1": float((q1["y"] > TAIL_S).mean()) if len(q1) else float("nan"),
        "frac_gt30_q8": float((q8["y"] > TAIL_S).mean()) if len(q8) else float("nan"),
        "frac_gt60_q1": float((q1["y"] > HOUR_S).mean()) if len(q1) else float("nan"),
        "frac_gt60_q8": float((q8["y"] > HOUR_S).mean()) if len(q8) else float("nan"),
        "resid_mean_q1": float(q1["resid"].mean()) if len(q1) else float("nan"),
        "resid_mean_q8": float(q8["resid"].mean()) if len(q8) else float("nan"),
        "resid_rmse_q1": float(np.sqrt((q1["resid"] ** 2).mean())) if len(q1) else float("nan"),
        "resid_rmse_q8": float(np.sqrt((q8["resid"] ** 2).mean())) if len(q8) else float("nan"),
        "mean_ae_q1": float(np.nanmean(q1["aobt_eobt"])) if len(q1) else float("nan"),
        "mean_ae_q8": float(np.nanmean(q8["aobt_eobt"])) if len(q8) else float("nan"),
    }
    if high_low["frac_gt30_q1"] and np.isfinite(high_low["frac_gt30_q1"]):
        high_low["gt30_rate_ratio_q8q1"] = high_low["frac_gt30_q8"] / max(high_low["frac_gt30_q1"], 1e-9)
    else:
        high_low["gt30_rate_ratio_q8q1"] = float("nan")
    high_low["resid_rmse_gap_q8q1"] = high_low["resid_rmse_q8"] - high_low["resid_rmse_q1"]

    # within each AOBT−EOBT quartile: dis Q8 vs Q1 (or max vs min available)
    within = []
    for ae, sub in pdf.groupby("ae_q"):
        d1 = sub[sub["dis_q"] == "Q1"]
        d8 = sub[sub["dis_q"] == "Q8"]
        if len(d1) < 50 or len(d8) < 50:
            # fall back to lowest/highest present labels
            labs = sorted(sub["dis_q"].dropna().unique())
            if len(labs) < 2:
                continue
            d1 = sub[sub["dis_q"] == labs[0]]
            d8 = sub[sub["dis_q"] == labs[-1]]
        r1 = float((d1["y"] > TAIL_S).mean()) if len(d1) else float("nan")
        r8 = float((d8["y"] > TAIL_S).mean()) if len(d8) else float("nan")
        within.append(
            {
                "ae_q": ae,
                "n_low": int(len(d1)),
                "n_high": int(len(d8)),
                "frac_gt30_low": r1,
                "frac_gt30_high": r8,
                "ratio": r8 / r1 if r1 and r1 > 0 else float("nan"),
                "resid_rmse_low": float(np.sqrt((d1["resid"] ** 2).mean())) if len(d1) else float("nan"),
                "resid_rmse_high": float(np.sqrt((d8["resid"] ** 2).mean())) if len(d8) else float("nan"),
                "resid_mean_low": float(d1["resid"].mean()) if len(d1) else float("nan"),
                "resid_mean_high": float(d8["resid"].mean()) if len(d8) else float("nan"),
            }
        )
    within_df = pd.DataFrame(within)

    corrs = {
        "corr_dis_aobt_eobt": corr_safe(pdf["dis_state_30m"], pdf["aobt_eobt"]),
        "corr_dis_y": corr_safe(pdf["dis_state_30m"], pdf["y"]),
        "corr_dis_resid": corr_safe(pdf["dis_state_30m"], pdf["resid"]),
        "corr_frac20_resid": corr_safe(pdf["dis_frac20_30m"], pdf["resid"]),
        "corr_frac30_resid": corr_safe(pdf["dis_frac30_30m"], pdf["resid"]),
        "corr_ndly60_resid": corr_safe(pdf["dis_n_dly20_60m"], pdf["resid"]),
        "corr_arr_resid": corr_safe(pdf["arr_delay_mean_30m"], pdf["resid"]),
        "corr_med_resid": corr_safe(pdf["dis_med_ae_30m"], pdf["resid"]),
        "corr_aobt_eobt_resid": corr_safe(pdf["aobt_eobt"], pdf["resid"]),
        "corr_dis_abs_resid": corr_safe(pdf["dis_state_30m"], np.abs(pdf["resid"].to_numpy())),
    }
    return {
        "pdf": pdf,
        "by_dis": by_dis,
        "by_frac": by_frac,
        "by_ae": by_ae,
        "two_way": two_way,
        "high_low": high_low,
        "within_ae": within_df,
        "corrs": corrs,
    }


def decide_phase3(p2: dict) -> dict:
    hl = p2["high_low"]
    w = p2["within_ae"]
    corrs = p2["corrs"]
    ratio = hl.get("gt30_rate_ratio_q8q1", np.nan)
    gap = hl.get("resid_rmse_gap_q8q1", np.nan)
    overall_tail = bool(np.isfinite(ratio) and ratio >= 1.4) or bool(np.isfinite(gap) and gap >= 20)
    n_within = 0
    if len(w):
        for _, row in w.iterrows():
            if np.isfinite(row["ratio"]) and row["ratio"] >= 1.25 and row["n_low"] >= 50 and row["n_high"] >= 50:
                n_within += 1
            elif (
                np.isfinite(row["resid_rmse_high"])
                and np.isfinite(row["resid_rmse_low"])
                and (row["resid_rmse_high"] - row["resid_rmse_low"]) >= 20
                and row["n_low"] >= 50
            ):
                n_within += 1
    beyond_ae = n_within >= 2
    weak_resid_corr = abs(corrs.get("corr_dis_resid", 0.0) or 0.0) < 0.02 and abs(
        corrs.get("corr_dis_abs_resid", 0.0) or 0.0
    ) < 0.05
    run = bool(overall_tail and beyond_ae and not weak_resid_corr)
    if not overall_tail:
        reason = "High vs low disruption does not concentrate the >30 min tail or residual RMSE."
    elif not beyond_ae:
        reason = "Disruption state is redundant with own AOBT−EOBT (within-bin tail rates do not rise)."
    elif weak_resid_corr:
        reason = "Almost no correlation with frozen-model residual; leftover tail not identified."
    else:
        reason = (
            f"High disruption concentrates the tail (Q8/Q1 >30m rate {ratio:.2f}, "
            f"resid RMSE gap {gap:.1f}s) and this remains in {n_within} AOBT−EOBT bins."
        )
    return {"run_phase3": run, "reason": reason, "n_within_ae_signal": n_within, "overall_tail": overall_tail}


def airport_rmse_table(va, y, pred_base, pred_new, unmatched) -> pd.DataFrame:
    ap = va["airport"].to_numpy()
    um = np.asarray(unmatched, dtype=bool)
    rows = []
    for a in AIRPORTS:
        m = (~um) & (ap == a) & np.isfinite(y) & np.isfinite(pred_base)
        if m.sum() == 0:
            continue
        rec = {
            "airport": a,
            "n": int(m.sum()),
            "rmse_base": rmse(y[m], pred_base[m]),
            "mae_base": mae(y[m], pred_base[m]),
        }
        if pred_new is not None:
            rec["rmse_e16a"] = rmse(y[m], pred_new[m])
            rec["mae_e16a"] = mae(y[m], pred_new[m])
            rec["d_rmse"] = rec["rmse_e16a"] - rec["rmse_base"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("rmse_base", ascending=False)


def plot_quantile_bars(by_dis: pd.DataFrame, col: str, ylabel: str, title: str, name: str, pct=False):
    df = by_dis.sort_values("dis_q")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    vals = df[col].to_numpy()
    if pct:
        vals = 100.0 * vals
    ax.bar(df["dis_q"], vals, color="#4C72B0")
    ax.set_xlabel("dis_state_30m quantile (matched Jan+Jul)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    savefig(name)


def plot_two_way_heatmap(two: pd.DataFrame, value: str, title: str, name: str, pct=False):
    pv = two.pivot(index="ae_q", columns="dis_q", values=value)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    mat = pv.to_numpy().astype(float)
    if pct:
        mat = 100.0 * mat
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pv.shape[1]))
    ax.set_xticklabels(list(pv.columns))
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(list(pv.index))
    ax.set_xlabel("disruption quantile")
    ax.set_ylabel("own AOBT−EOBT quartile")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    savefig(name)


def plot_metric_compare(base: dict, new: dict | None):
    metrics = [
        ("rmse_overall", "Overall RMSE"),
        ("rmse_matched", "Matched RMSE"),
        ("rmse_lt20_matched", "<20 min"),
        ("rmse_gt30_matched", ">30 min"),
        ("rmse_gt60_matched", ">60 min"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(metrics))
    b = [base[k] for k, _ in metrics]
    ax.bar(x - 0.18, b, 0.36, label="E14 L2 baseline", color="#4C72B0")
    if new is not None:
        n = [new[k] for k, _ in metrics]
        ax.bar(x + 0.18, n, 0.36, label="E16-A + disruption", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], rotation=15)
    ax.set_ylabel("seconds")
    ax.set_title("Jan+Jul RMSE — baseline vs E16-A")
    ax.legend()
    savefig("e16a_rmse_comparison_janjul.png")


def fmt_s(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.2f}"


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def write_report(payload: dict) -> Path:
    j = payload["janjul"]
    d = payload.get("dec")
    p2 = payload["phase2_janjul"]
    decision = payload["decision"]
    base_j = j["score_base"]
    new_j = j.get("score_e16a")
    path = REP / "E16A_report.md"
    lines = []
    a = lines.append
    a("# E16-A — Airport / network disruption-state features")
    a("")
    a("**Project:** OpenAir")
    a("**Target:** `TAXITIME_SEC_mvt`")
    a("**Experiment:** E16-A")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress check:** December 2025")
    a("**Data:** 12 `training_*.parquet` files only. No ranking/submission.")
    a("")
    a("Script: `experiments/run_e16a.py` (copied to `analysis/E16A/run_e16a.py`).")
    a("")
    a("---")
    a("")
    a("## Objective")
    a("")
    a("E14/E15: matched RMSE sits at **256.46 s** because the frozen-feature L2 residual")
    a("compresses disruption tails. Huber / tail weights / two-stage mixture all lost to L2.")
    a("")
    a("Hypothesis: own `AOBT−EOBT` says this flight pushed late. It does not say whether")
    a("the **airport** is in a broader push-delay disruption, where very long taxi becomes")
    a("more likely. A causal airport-hour disruption state might separate a normal late")
    a("push from a systemic event.")
    a("")
    a("This is one feature-family test. No loss change, no HP search, no callsign features.")
    a("")
    a("## Frozen baseline")
    a("")
    a("```")
    a("P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)")
    a("y_hat = P_cal + LGB_L2_residual(...)")
    a("if unmatched and airport==LIRF: y_hat = MVT - SCHED")
    a("```")
    a("")
    a(f"Reproduction: Jan+Jul overall {fmt_s(base_j['rmse_overall'])}, matched **{fmt_s(base_j['rmse_matched'])}**")
    a(f"(E14 256.46, delta {base_j['rmse_matched']-BASELINE_MATCHED:+.2f} s).")
    a("")
    a("## Phase 1 — causal feature family")
    a("")
    a("Prediction time for a scored departure is `MVT_TIME_UTC_mvt`.")
    a("A neighbouring departure contributes `AOBT−EOBT` only if its **AOBT ≤ scored MVT**")
    a("(the push delay is then known). The scored flight is subtracted from the window.")
    a("No BLOCK, no TAXITIME, no future ARVT_3, no other flight's taxi outcome.")
    a("")
    a("| Feature | Definition |")
    a("|---|---|")
    a("| `dis_state_30m` (`airport_recent_push_delay_state`) | Mean `AOBT−EOBT` of **other** pushes in the previous 30 min |")
    a("| `dis_frac20_30m` / `dis_frac30_30m` | Fraction of those pushes with delay >20 / >30 min |")
    a("| `dis_n_dly20_60m` | Count of other pushes with delay >20 min in the previous 60 min |")
    a("| `dis_n_*m` | Count of other pushes in 15/30/60 min (AOBT-indexed, not takeoff-indexed) |")
    a("| `dis_med_ae_30m` / `dis_p90_ae_30m` | Event-time rolling median/p90 (asof last push ≤ MVT; ~1-pt self contamination) |")
    a("| `dis_state_vs_hour` | `dis_state_30m` / train-split airport×hour mean |")
    a("| `arr_delay_mean_30m` | Mean `MVT−SCHED` of arrivals **already landed** in the previous 30 min |")
    a("")
    a("Airport×hour climatology is fit on the **training split only**, same protocol as `geo_mean`.")
    a("")
    a("## Phase 2 — does it identify the E14 tail?")
    a("")
    a("Jan+Jul matched, frozen L2 residual. Quantiles of `dis_state_30m`.")
    a("")
    a("| Q | N | Mean y | Mean AOBT−EOBT | Mean dis | >30m rate | >60m rate | Resid mean | Resid RMSE | SSE share |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in p2["by_dis"].sort_values("dis_q").iterrows():
        a(
            f"| {r['dis_q']} | {int(r['n']):,} | {r['mean_y']:.0f} | {r['mean_aobt_eobt']:.0f} | "
            f"{r['mean_dis']:.0f} | {100*r['frac_gt30']:.2f}% | {100*r['frac_gt60']:.2f}% | "
            f"{r['resid_mean']:.1f} | {r['resid_rmse']:.1f} | {100*r['sse_share']:.1f}% |"
        )
    hl = p2["high_low"]
    a("")
    a(
        f"Q8 vs Q1: >30m rate {fmt_pct(hl['frac_gt30_q1'])} → {fmt_pct(hl['frac_gt30_q8'])} "
        f"(ratio {hl['gt30_rate_ratio_q8q1']:.2f}); residual RMSE {fmt_s(hl['resid_rmse_q1'])} → "
        f"{fmt_s(hl['resid_rmse_q8'])} (gap {hl['resid_rmse_gap_q8q1']:+.1f} s); "
        f"mean residual {fmt_s(hl['resid_mean_q1'])} → {fmt_s(hl['resid_mean_q8'])}."
    )
    a("")
    a("Correlations on matched Jan+Jul:")
    a("")
    a("| Pair | corr |")
    a("|---|---:|")
    for k, v in p2["corrs"].items():
        a(f"| {k} | {v:.4f} |")
    a("")
    a("### Controlled for own AOBT−EOBT")
    a("")
    a("The hypothesis fails if disruption only restates this flight's lateness.")
    a("Within each own-`AOBT−EOBT` quartile, high vs low disruption:")
    a("")
    a("| AOBT−EOBT Q | N low | N high | >30m low | >30m high | ratio | Resid RMSE low | Resid RMSE high | Resid mean low | Resid mean high |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in p2["within_ae"].iterrows():
        a(
            f"| {r['ae_q']} | {int(r['n_low']):,} | {int(r['n_high']):,} | "
            f"{100*r['frac_gt30_low']:.2f}% | {100*r['frac_gt30_high']:.2f}% | {r['ratio']:.2f} | "
            f"{r['resid_rmse_low']:.1f} | {r['resid_rmse_high']:.1f} | "
            f"{r['resid_mean_low']:.1f} | {r['resid_mean_high']:.1f} |"
        )
    a("")
    a(f"**Phase 2 decision:** {decision['reason']}")
    a(f"Run Phase 3 model? **{'yes' if decision['run_phase3'] else 'no'}**")
    a("")
    a("## Phase 3 — residual LightGBM with disruption features")
    a("")
    if new_j is None:
        a("Not trained. Phase 2 did not show disruption information beyond `AOBT−EOBT`")
        a("that would justify adding the family to the frozen matrix.")
        a("")
        a("| Metric | Baseline | E16-A |")
        a("|---|---:|---:|")
        for lab, k in [
            ("Overall RMSE", "rmse_overall"),
            ("Matched RMSE", "rmse_matched"),
            ("Matched MAE", "mae_matched"),
            ("<20m RMSE", "rmse_lt20_matched"),
            (">30m RMSE", "rmse_gt30_matched"),
            (">60m RMSE", "rmse_gt60_matched"),
            ("SSE share >30m", "sse_share_gt30_matched"),
            ("Mean residual", "resid_mean_matched"),
        ]:
            a(f"| {lab} | {fmt_s(base_j[k]) if k != 'sse_share_gt30_matched' else fmt_pct(base_j[k])} | — |")
    else:
        a("Same L2 residual LightGBM; extra numeric columns listed above; everything else frozen.")
        a("")
        a("### Jan+Jul 2025")
        a("")
        a("| Metric | Baseline | E16-A | Δ |")
        a("|---|---:|---:|---:|")
        for lab, k, pct in [
            ("Overall RMSE", "rmse_overall", False),
            ("Matched RMSE", "rmse_matched", False),
            ("Matched MAE", "mae_matched", False),
            ("<20m RMSE", "rmse_lt20_matched", False),
            (">30m RMSE", "rmse_gt30_matched", False),
            (">60m RMSE", "rmse_gt60_matched", False),
            ("SSE share >30m", "sse_share_gt30_matched", True),
            ("Mean residual", "resid_mean_matched", False),
        ]:
            b, n = base_j[k], new_j[k]
            if pct:
                a(f"| {lab} | {fmt_pct(b)} | {fmt_pct(n)} | {100*(n-b):+.2f} pp |")
            else:
                a(f"| {lab} | {fmt_s(b)} | {fmt_s(n)} | {n-b:+.2f} |")
        if d is not None and d.get("score_e16a") is not None:
            a("")
            a("### December 2025")
            a("")
            a("| Metric | Baseline | E16-A | Δ |")
            a("|---|---:|---:|---:|")
            db, dn = d["score_base"], d["score_e16a"]
            for lab, k, pct in [
                ("Overall RMSE", "rmse_overall", False),
                ("Matched RMSE", "rmse_matched", False),
                ("Matched MAE", "mae_matched", False),
                ("<20m RMSE", "rmse_lt20_matched", False),
                (">30m RMSE", "rmse_gt30_matched", False),
                (">60m RMSE", "rmse_gt60_matched", False),
                ("SSE share >30m", "sse_share_gt30_matched", True),
                ("Mean residual", "resid_mean_matched", False),
            ]:
                b, n = db[k], dn[k]
                if pct:
                    a(f"| {lab} | {fmt_pct(b)} | {fmt_pct(n)} | {100*(n-b):+.2f} pp |")
                else:
                    a(f"| {lab} | {fmt_s(b)} | {fmt_s(n)} | {n-b:+.2f} |")
        imp = j.get("importance") or []
        if imp:
            a("")
            a("### New-feature gain")
            a("")
            a("| Feature | Gain | Rank among all features |")
            a("|---|---:|---:|")
            for rec in imp:
                a(f"| {rec['feature']} | {rec['gain']} | {rec['rank']} |")
        apt = j.get("airport")
        if apt is not None:
            a("")
            a("### Airport matched RMSE")
            a("")
            a("| Airport | N | Baseline | E16-A | Δ |")
            a("|---|---:|---:|---:|---:|")
            for _, r in apt.iterrows():
                a(
                    f"| {r['airport']} | {int(r['n']):,} | {r['rmse_base']:.2f} | "
                    f"{r['rmse_e16a']:.2f} | {r['d_rmse']:+.2f} |"
                )
    a("")
    a("## Answers")
    a("")
    a(payload["answers"]["beyond_ae"])
    a("")
    a(payload["answers"]["tail"])
    a("")
    a(payload["answers"]["overall"])
    a("")
    a(payload["answers"]["bulk"])
    a("")
    a(payload["answers"]["which"])
    a("")
    a(payload["answers"]["keep"])
    a("")
    a("## Artifacts")
    a("")
    a("- `analysis/E16A/figures/`")
    a("- `analysis/E16A/tables/`")
    a("- `experiments/results/E16A.json`")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  report {path}")
    return path


def build_answers(payload) -> dict:
    p2 = payload["phase2_janjul"]
    dec = payload["decision"]
    j = payload["janjul"]
    base = j["score_base"]
    new = j.get("score_e16a")
    corrs = p2["corrs"]
    hl = p2["high_low"]
    beyond = (
        f"**Does disruption contain information not already in AOBT−EOBT?** "
        f"corr(dis_state, AOBT−EOBT) = {corrs['corr_dis_aobt_eobt']:.3f}; "
        f"corr(dis_state, residual) = {corrs['corr_dis_resid']:.3f}. "
        f"Within-bin AOBT−EOBT signal in {dec['n_within_ae_signal']} quartiles. "
        + (
            "Yes, some leftover association remains after controlling own lateness."
            if dec["run_phase3"]
            else "No meaningful leftover: the airport state largely restates that late flights cluster, which own AOBT−EOBT already measures."
        )
    )
    if new is None:
        tail = (
            f"**Does it reduce matched tail error?** Not tested in-model. Diagnostically, Q8 vs Q1 "
            f">30m rate ratio {hl['gt30_rate_ratio_q8q1']:.2f}, residual RMSE gap "
            f"{hl['resid_rmse_gap_q8q1']:+.1f} s under the frozen model — a concentration of the "
            f"E14 failure, not yet a modelling gain."
        )
        overall = "**Does it improve overall RMSE?** No model was trained; baseline remains 378.28 / 256.46."
        bulk = "**Does it damage the <20m bulk?** Not applicable (no model)."
        which = (
            "**Which feature is useful?** None earned a place in the residual model. "
            f"`dis_state_30m` tracks AOBT−EOBT (corr {corrs['corr_dis_aobt_eobt']:.3f}). "
            f"Arrival delay corr with residual = {corrs['corr_arr_resid']:.3f}."
        )
        keep = (
            "**Keep the family?** **No.** Phase 2 did not show a disruption state that identifies "
            "systemic long-taxi risk beyond own AOBT−EOBT. Do not add it to the frozen matrix."
        )
    else:
        d30 = new["rmse_gt30_matched"] - base["rmse_gt30_matched"]
        d60 = new["rmse_gt60_matched"] - base["rmse_gt60_matched"]
        d20 = new["rmse_lt20_matched"] - base["rmse_lt20_matched"]
        dm = new["rmse_matched"] - base["rmse_matched"]
        do = new["rmse_overall"] - base["rmse_overall"]
        tail_ok = d30 < -5 or d60 < -10
        tail = (
            f"**Does it reduce matched tail error?** >30 min {base['rmse_gt30_matched']:.2f} → "
            f"{new['rmse_gt30_matched']:.2f} ({d30:+.2f} s); >60 min {base['rmse_gt60_matched']:.2f} → "
            f"{new['rmse_gt60_matched']:.2f} ({d60:+.2f} s). "
            + ("Yes, modestly." if tail_ok else "No meaningful tail reduction.")
        )
        overall = (
            f"**Does it improve overall RMSE?** {base['rmse_overall']:.2f} → {new['rmse_overall']:.2f} "
            f"({do:+.2f}); matched {base['rmse_matched']:.2f} → {new['rmse_matched']:.2f} ({dm:+.2f})."
        )
        bulk = (
            f"**Does it damage the <20m bulk?** {base['rmse_lt20_matched']:.2f} → "
            f"{new['rmse_lt20_matched']:.2f} ({d20:+.2f} s). "
            + ("Yes, bulk worsened." if d20 > 3 else "No material bulk damage.")
        )
        imps = j.get("importance") or []
        if imps:
            top = ", ".join(f"{r['feature']} (rank {r['rank']}, gain {r['gain']})" for r in imps[:4])
            which = f"**Which disruption feature is useful?** {top}."
        else:
            which = "**Which disruption feature is useful?** See gain table."
        keep = (
            "**Keep the family?** "
            + (
                "**Yes, provisionally** — tail and/or matched RMSE improved without wrecking <20 min."
                if (dm < -1 or tail_ok) and d20 <= 5
                else "**No.** Incremental signal did not translate into a cleaner tail fit on the ranking analogue."
            )
        )
    return {"beyond_ae": beyond, "tail": tail, "overall": overall, "bulk": bulk, "which": which, "keep": keep}


def main():
    t0 = datetime.now(timezone.utc)
    log("E16-A: load training_*.parquet only...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    log(f"base features ready {dep.height:,}")

    log("Phase 1: causal push-delay disruption state...")
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    log("Phase 1: arrival delay state (landed only)...")
    arr_d = load_arr_delay()
    dep = add_arrival_delay_state(dep, arr_d)
    n_state = int(np.isfinite(dep["dis_state_30m"].to_numpy()).sum())
    log(f"  dis_state_30m finite {n_state:,}/{dep.height:,}  mean={float(np.nanmean(dep['dis_state_30m'].to_numpy())):.1f}")

    payload = {"janjul": {}, "dec": {}}

    # ----- Jan+Jul: baseline then Phase 2 -----
    log("===== janjul baseline L2 =====")
    pack = prepare_split(dep, [1, 7])
    log("  fitting frozen residual LGB...")
    p_cal, pred_b, model_b = fit_residual(pack["tr"], pack["va"], extra=[])
    y = pack["va"]["y"].to_numpy().astype(np.float64)
    um = pack["va"]["unmatched"].to_numpy()
    score_b = score_pred(y, pred_b, um)
    log(
        f"  baseline overall={score_b['rmse_overall']:.2f} matched={score_b['rmse_matched']:.2f} "
        f"delta vs 256.46={score_b['rmse_matched']-BASELINE_MATCHED:+.2f}"
    )
    if abs(score_b["rmse_matched"] - BASELINE_MATCHED) > 8:
        raise SystemExit(f"baseline failed to reproduce: {score_b['rmse_matched']}")

    log("Phase 2 diagnostics on Jan+Jul matched...")
    p2 = phase2_tables(pack["va"], y, pred_b, um)
    savecsv("e16a_disruption_quantile_janjul.csv", p2["by_dis"])
    savecsv("e16a_frac20_quantile_janjul.csv", p2["by_frac"])
    savecsv("e16a_aobt_eobt_quartile_janjul.csv", p2["by_ae"])
    savecsv("e16a_twoway_ae_dis_janjul.csv", p2["two_way"])
    savecsv("e16a_within_ae_janjul.csv", p2["within_ae"])
    decision = decide_phase3(p2)
    log(f"  Phase 3 gate: run={decision['run_phase3']}  {decision['reason']}")
    log(f"  corr dis vs aobt_eobt={p2['corrs']['corr_dis_aobt_eobt']:.3f}  dis vs resid={p2['corrs']['corr_dis_resid']:.3f}")

    plot_quantile_bars(p2["by_dis"], "frac_gt30", ">30 min rate (%)", "Taxi >30 min rate by disruption quantile", "e16a_gt30_rate_by_dis_q.png", pct=True)
    plot_quantile_bars(p2["by_dis"], "frac_gt60", ">60 min rate (%)", "Taxi >60 min rate by disruption quantile", "e16a_gt60_rate_by_dis_q.png", pct=True)
    plot_quantile_bars(p2["by_dis"], "resid_rmse", "Residual RMSE (s)", "Frozen-model residual RMSE by disruption quantile", "e16a_resid_rmse_by_dis_q.png")
    plot_quantile_bars(p2["by_dis"], "resid_mean", "Mean residual (s)", "Frozen-model mean residual by disruption quantile", "e16a_resid_mean_by_dis_q.png")
    plot_quantile_bars(p2["by_dis"], "mean_y", "Mean taxi (s)", "Mean TAXITIME by disruption quantile", "e16a_mean_y_by_dis_q.png")
    plot_two_way_heatmap(p2["two_way"], "frac_gt30", ">30 min rate (%) by AOBT−EOBT × disruption", "e16a_twoway_gt30.png", pct=True)
    plot_two_way_heatmap(p2["two_way"], "resid_rmse", "Residual RMSE by AOBT−EOBT × disruption", "e16a_twoway_resid_rmse.png")
    plot_two_way_heatmap(p2["two_way"], "resid_mean", "Mean residual by AOBT−EOBT × disruption", "e16a_twoway_resid_mean.png")

    payload["janjul"]["score_base"] = score_b
    payload["janjul"]["pred_base"] = pred_b
    payload["phase2_janjul"] = {k: v for k, v in p2.items() if k != "pdf"}
    payload["decision"] = decision

    pred_new = None
    model_new = None
    if decision["run_phase3"]:
        log("Phase 3: fitting L2 residual + disruption features (janjul)...")
        _, pred_new, model_new = fit_residual(pack["tr"], pack["va"], extra=MODEL_DISRUPT_COLS)
        score_n = score_pred(y, pred_new, um)
        payload["janjul"]["score_e16a"] = score_n
        log(
            f"  E16-A overall={score_n['rmse_overall']:.2f} matched={score_n['rmse_matched']:.2f} "
            f"<20={score_n['rmse_lt20_matched']:.2f} >30={score_n['rmse_gt30_matched']:.2f} "
            f">60={score_n['rmse_gt60_matched']:.2f}"
        )
        names = list(BASE_NUM_COLS) + MODEL_DISRUPT_COLS + CAT_COLS
        gain = model_new.booster_.feature_importance(importance_type="gain")
        order = np.argsort(-gain)
        rank_of = {names[i]: int(r + 1) for r, i in enumerate(order)}
        imp_rows = []
        for c in MODEL_DISRUPT_COLS:
            if c not in names:
                continue
            idx = names.index(c)
            imp_rows.append({"feature": c, "gain": int(gain[idx]), "rank": rank_of[c], "n_features": len(names)})
        imp_rows.sort(key=lambda r: r["rank"])
        payload["janjul"]["importance"] = imp_rows
        savecsv("e16a_disruption_importance_janjul.csv", pd.DataFrame(imp_rows))
        top = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False).head(25)
        savecsv("e16a_top_importance_janjul.csv", top)

        apt = airport_rmse_table(pack["va"], y, pred_b, pred_new, um)
        payload["janjul"]["airport"] = apt
        savecsv("e16a_airport_rmse_janjul.csv", apt)

        # quantile RMSE of new model
        pdf = p2["pdf"].copy()
        pdf["pred_e16a"] = pred_new[(~um) & np.isfinite(y) & np.isfinite(pred_b)]
        pdf["resid_e16a"] = pdf["y"] - pdf["pred_e16a"]
        qnew = group_diag(pdf.assign(pred=pdf["pred_e16a"], resid=pdf["resid_e16a"]), "dis_q").sort_values("dis_q")
        savecsv("e16a_disruption_quantile_model_janjul.csv", qnew)

        plot_metric_compare(score_b, score_n)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.barh([r["feature"] for r in imp_rows][::-1], [r["gain"] for r in imp_rows][::-1], color="#DD8452")
        ax.set_xlabel("LightGBM gain")
        ax.set_title("E16-A disruption feature gain (Jan+Jul residual model)")
        savefig("e16a_disruption_gain.png")
    else:
        plot_metric_compare(score_b, None)
        apt = airport_rmse_table(pack["va"], y, pred_b, None, um)
        savecsv("e16a_airport_rmse_janjul.csv", apt)

    # ----- December stress (baseline always; E16-A if Phase 3 ran) -----
    log("===== december =====")
    pack_d = prepare_split(dep, [12])
    log("  fitting frozen residual LGB (dec)...")
    _, pred_bd, _ = fit_residual(pack_d["tr"], pack_d["va"], extra=[])
    yd = pack_d["va"]["y"].to_numpy().astype(np.float64)
    umd = pack_d["va"]["unmatched"].to_numpy()
    score_bd = score_pred(yd, pred_bd, umd)
    payload["dec"]["score_base"] = score_bd
    log(f"  dec baseline overall={score_bd['rmse_overall']:.2f} matched={score_bd['rmse_matched']:.2f}")
    p2d = phase2_tables(pack_d["va"], yd, pred_bd, umd)
    savecsv("e16a_disruption_quantile_dec.csv", p2d["by_dis"])
    savecsv("e16a_twoway_ae_dis_dec.csv", p2d["two_way"])
    payload["phase2_dec"] = {k: v for k, v in p2d.items() if k != "pdf"}

    if decision["run_phase3"]:
        log("  fitting E16-A residual (dec)...")
        _, pred_nd, model_nd = fit_residual(pack_d["tr"], pack_d["va"], extra=MODEL_DISRUPT_COLS)
        score_nd = score_pred(yd, pred_nd, umd)
        payload["dec"]["score_e16a"] = score_nd
        log(
            f"  dec E16-A overall={score_nd['rmse_overall']:.2f} matched={score_nd['rmse_matched']:.2f} "
            f">30={score_nd['rmse_gt30_matched']:.2f}"
        )
        apt_d = airport_rmse_table(pack_d["va"], yd, pred_bd, pred_nd, umd)
        savecsv("e16a_airport_rmse_dec.csv", apt_d)
        payload["dec"]["airport"] = apt_d

        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        metrics = [
            ("rmse_overall", "Overall"),
            ("rmse_matched", "Matched"),
            ("rmse_lt20_matched", "<20"),
            ("rmse_gt30_matched", ">30"),
            ("rmse_gt60_matched", ">60"),
        ]
        x = np.arange(len(metrics))
        ax.bar(x - 0.18, [score_bd[k] for k, _ in metrics], 0.36, label="baseline", color="#4C72B0")
        ax.bar(x + 0.18, [score_nd[k] for k, _ in metrics], 0.36, label="E16-A", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in metrics])
        ax.set_ylabel("RMSE (s)")
        ax.set_title("December RMSE — baseline vs E16-A")
        ax.legend()
        savefig("e16a_rmse_comparison_dec.png")

    metrics_rows = []
    for split, blob in (("janjul", payload["janjul"]), ("dec", payload["dec"])):
        for name, key in (("baseline", "score_base"), ("E16A", "score_e16a")):
            if key in blob:
                metrics_rows.append({"split": split, "variant": name, **blob[key]})
    savecsv("e16a_metrics.csv", pd.DataFrame(metrics_rows))

    payload["answers"] = build_answers(payload)
    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "janjul_scores": {"baseline": payload["janjul"]["score_base"], "e16a": payload["janjul"].get("score_e16a")},
        "dec_scores": {"baseline": payload["dec"]["score_base"], "e16a": payload["dec"].get("score_e16a")},
        "phase2_corrs_janjul": p2["corrs"],
        "phase2_high_low_janjul": p2["high_low"],
        "importance": payload["janjul"].get("importance"),
        "answers": payload["answers"],
    }
    (TAB / "e16a_findings.json").write_text(json.dumps(findings, indent=2, default=json_conv), encoding="utf-8")
    save_result("E16A", findings)
    write_report(payload)

    src = Path(__file__).resolve()
    shutil.copy2(src, OUT / "run_e16a.py")
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log(f"E16-A done in {elapsed/60:.1f} min. phase3={decision['run_phase3']}")


if __name__ == "__main__":
    main()
