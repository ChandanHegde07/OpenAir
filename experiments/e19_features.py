"""E19: causal local queue-state features for RMSE tail reduction.

All features are strictly causal (event time < t of the scored departure).
NONE uses TAXITIME, BLOCK, or any future observation. Ranking-safe.

Families (labels used in feature_importance.csv):
  A  neighbour state       e19_*_gap/ae/aobt/sched/n_dly/same_*   (airport stream)
  B  same-runway queue     e19_rwy_*                                (runway stream)
  C  delay shock           e19_shock_* (needs train climatology)
  D  queue/arrival pressure e19_dep_* / e19_arr_* / e19_inv_* / e19_gap_compression
  E  interactions          e19_qp* / e19_*_pressure / e19_*_ratio
  F  tail-aware            e19_consec_* / e19_max_ae_* / e19_rwy_ndly_* /
                           e19_tailprod*

Family C climatology is fit on the TRAINING split only (runner calls
attach_shock(train, val) like attach_hour_baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AIRPORTS  # noqa: E402

DLY = 1200.0
DLY_STRONG = 1800.0
ARR_DLY = 900.0

A_COLS = [
    "e19_gap1", "e19_gap2", "e19_gap3", "e19_gap_mean3", "e19_gap_med5", "e19_gap_min5",
    "e19_ae_p1", "e19_ae_mean3", "e19_ae_med5", "e19_ae_p90_5", "e19_ae_std5",
    "e19_aobt_p1", "e19_aobt_mean3", "e19_aobt_p90_5",
    "e19_sched_p1", "e19_sched_mean3", "e19_sched_p90_5",
    "e19_n_dly_3", "e19_n_dly_5", "e19_n_dly_10", "e19_n_dly_str_5",
    "e19_max_ae_5", "e19_same_rwy_p1", "e19_same_al_p1", "e19_same_type_p1",
    "e19_frac_same_rwy_5", "e19_frac_same_al_5", "e19_frac_same_type_5",
]

B_COLS = [
    "e19_rwy_gap1", "e19_rwy_gap2", "e19_rwy_gap3", "e19_rwy_gap_mean3", "e19_rwy_gap_min5",
    "e19_rwy_ae_p1", "e19_rwy_ae_mean3", "e19_rwy_ae_med3", "e19_rwy_ae_p90_5",
    "e19_rwy_aobt_p1", "e19_rwy_aobt_mean3", "e19_rwy_aobt_p90_5",
    "e19_rwy_sched_p1", "e19_rwy_sched_mean3", "e19_rwy_sched_p90_5",
]

C_COLS = ["e19_shock_p1", "e19_shock_mean3", "e19_shock_med5", "e19_shock_p90_5", "e19_shock_rwy"]

D_COLS = [
    "e19_dep_2m", "e19_dep_5m", "e19_dep_10m", "e19_dep_20m", "e19_dep_30m",
    "e19_rwy_dep_2m", "e19_rwy_dep_5m", "e19_rwy_dep_10m", "e19_rwy_dep_20m", "e19_rwy_dep_30m",
    "e19_arr_2m", "e19_arr_5m", "e19_arr_10m", "e19_arr_20m", "e19_arr_30m",
    "e19_arr_dly_5m", "e19_arr_dly_10m",
    "e19_inv_gap1", "e19_inv_rwy_gap1", "e19_gap_compression",
]

E_COLS = [
    "e19_qp", "e19_qp2", "e19_qp_dis", "e19_qp_rwyd", "e19_qp_dens",
    "e19_arr_dep_pressure", "e19_arr_rwy_pressure", "e19_arr_dly_pressure",
    "e19_rwy_dis_pressure", "e19_arr_dep_ratio",
]

F_COLS = [
    "e19_max_ae_10", "e19_consec_dly", "e19_consec_rwy_dly",
    "e19_rwy_ndly_5", "e19_rwy_max_ae_5",
    "e19_tailprod1", "e19_tailprod2", "e19_tailprod3", "e19_tailprod4",
]

EXP_COLS = ["e19_exp_hr", "e19_exp_rwy"]


def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.full(arr.size, np.nan)
    if arr.size > k:
        out[k:] = arr[:-k]
    return out


def _lag_obj(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(arr.size, dtype=object)
    out[:] = np.nan
    if arr.size > k:
        out[k:] = arr[:-k]
    return out


def _roll(arr: np.ndarray, k: int, stat: str) -> np.ndarray:
    s = pd.Series(arr)
    r = s.rolling(k, min_periods=1)
    fns = {
        "mean": r.mean,
        "med": r.median,
        "p90": lambda: r.quantile(0.9),
        "std": r.std,
        "max": r.max,
    }
    return fns[stat]().shift(1).to_numpy()


def _roll_sum(arr: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(arr).rolling(k, min_periods=1).sum().shift(1).to_numpy()


def _consec(flag: np.ndarray) -> np.ndarray:
    s = pd.Series(flag.astype(float))
    grp = (~s.astype(bool)).cumsum()
    return s.groupby(grp).cumsum().shift(1).to_numpy()


def add_neighbor(dep: pl.DataFrame) -> pl.DataFrame:
    """Family A + the two airport-level tail cols (max_ae_10, consec_dly)."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan, dtype=np.float64) for c in A_COLS + ["e19_max_ae_10", "e19_consec_dly"]}

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    ae = dep["aobt_eobt"].to_numpy().astype(np.float64)
    aobt = dep["mvt_aobt"].to_numpy().astype(np.float64)
    sched = dep["mvt_sched"].to_numpy().astype(np.float64)
    rwy = _str(dep["RUNWAY_mvt"])
    al = _str(dep["AIRCRAFT_OPERATOR_flt"])
    ty = _str(dep["AIRCRAFT_TYPE_mvt"])

    for a in AIRPORTS:
        sel = ap == a
        m = int(sel.sum())
        if m == 0:
            continue
        t = (mvt[sel] - mvt[sel][0]).astype(np.float64) / 1e9
        ae_v, aobt_v, sched_v = ae[sel], aobt[sel], sched[sel]
        rwy_v, al_v, ty_v = rwy[sel], al[sel], ty[sel]

        t1, t2, t3 = (_lag(t, k) for k in (1, 2, 3))
        g1, g2, g3 = t - t1, t - t2, t - t3
        g5 = np.column_stack([t - _lag(t, k) for k in range(1, 6)])
        cols["e19_gap1"][sel] = g1
        cols["e19_gap2"][sel] = g2
        cols["e19_gap3"][sel] = g3
        cols["e19_gap_mean3"][sel] = np.nanmean(np.column_stack([g1, g2, g3]), axis=1)
        cols["e19_gap_med5"][sel] = np.nanmedian(g5, axis=1)
        cols["e19_gap_min5"][sel] = np.nanmin(g5, axis=1)

        cols["e19_ae_p1"][sel] = _lag(ae_v, 1)
        cols["e19_ae_mean3"][sel] = _roll(ae_v, 3, "mean")
        cols["e19_ae_med5"][sel] = _roll(ae_v, 5, "med")
        cols["e19_ae_p90_5"][sel] = _roll(ae_v, 5, "p90")
        cols["e19_ae_std5"][sel] = _roll(ae_v, 5, "std")
        cols["e19_aobt_p1"][sel] = _lag(aobt_v, 1)
        cols["e19_aobt_mean3"][sel] = _roll(aobt_v, 3, "mean")
        cols["e19_aobt_p90_5"][sel] = _roll(aobt_v, 5, "p90")
        cols["e19_sched_p1"][sel] = _lag(sched_v, 1)
        cols["e19_sched_mean3"][sel] = _roll(sched_v, 3, "mean")
        cols["e19_sched_p90_5"][sel] = _roll(sched_v, 5, "p90")

        ae0 = np.where(np.isfinite(ae_v), ae_v, 0.0)
        dly = (ae0 > DLY).astype(np.float64)
        dly_s = (ae0 > DLY_STRONG).astype(np.float64)
        cols["e19_n_dly_3"][sel] = _roll_sum(dly, 3)
        cols["e19_n_dly_5"][sel] = _roll_sum(dly, 5)
        cols["e19_n_dly_10"][sel] = _roll_sum(dly, 10)
        cols["e19_n_dly_str_5"][sel] = _roll_sum(dly_s, 5)
        cols["e19_max_ae_5"][sel] = _roll(ae_v, 5, "max")
        cols["e19_max_ae_10"][sel] = _roll(ae_v, 10, "max")
        cols["e19_consec_dly"][sel] = _consec((ae0 > DLY).astype(np.int8))

        cols["e19_same_rwy_p1"][sel] = np.where(pd.isna(_lag_obj(rwy_v, 1)), np.nan, (_lag_obj(rwy_v, 1) == rwy_v)).astype(np.float64)
        cols["e19_same_al_p1"][sel] = np.where(pd.isna(_lag_obj(al_v, 1)), np.nan, (_lag_obj(al_v, 1) == al_v)).astype(np.float64)
        cols["e19_same_type_p1"][sel] = np.where(pd.isna(_lag_obj(ty_v, 1)), np.nan, (_lag_obj(ty_v, 1) == ty_v)).astype(np.float64)

        def frac_same(v):
            stacked = []
            for kk in range(1, 6):
                lag = _lag_obj(v, kk)
                eq = np.where(pd.isna(lag), np.nan, (lag == v).astype(np.float64))
                stacked.append(eq)
            return np.nanmean(np.column_stack(stacked), axis=1)

        cols["e19_frac_same_rwy_5"][sel] = frac_same(rwy_v)
        cols["e19_frac_same_al_5"][sel] = frac_same(al_v)
        cols["e19_frac_same_type_5"][sel] = frac_same(ty_v)

    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])


def add_runway_queue(dep: pl.DataFrame) -> pl.DataFrame:
    """Family B + runway tail cols (rwy_ndly_5, rwy_max_ae_5, consec_rwy_dly)."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan, dtype=np.float64) for c in
            B_COLS + ["e19_rwy_ndly_5", "e19_rwy_max_ae_5", "e19_consec_rwy_dly"]}

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = _str(dep["RUNWAY_mvt"])
    ae = dep["aobt_eobt"].to_numpy().astype(np.float64)
    aobt = dep["mvt_aobt"].to_numpy().astype(np.float64)
    sched = dep["mvt_sched"].to_numpy().astype(np.float64)

    groups = set(zip(ap.tolist(), rwy.tolist()))
    for (a, r) in sorted(groups):
        sel = (ap == a) & (rwy == r)
        m = int(sel.sum())
        if m < 1:
            continue
        t = (mvt[sel] - mvt[sel][0]).astype(np.float64) / 1e9
        ae_v, aobt_v, sched_v = ae[sel], aobt[sel], sched[sel]
        t1, t2, t3 = (_lag(t, k) for k in (1, 2, 3))
        g5 = np.column_stack([t - _lag(t, k) for k in range(1, 6)])
        cols["e19_rwy_gap1"][sel] = t - t1
        cols["e19_rwy_gap2"][sel] = t - t2
        cols["e19_rwy_gap3"][sel] = t - t3
        cols["e19_rwy_gap_mean3"][sel] = np.nanmean(np.column_stack([t - t1, t - t2, t - t3]), axis=1)
        cols["e19_rwy_gap_min5"][sel] = np.nanmin(g5, axis=1)
        cols["e19_rwy_ae_p1"][sel] = _lag(ae_v, 1)
        cols["e19_rwy_ae_mean3"][sel] = _roll(ae_v, 3, "mean")
        cols["e19_rwy_ae_med3"][sel] = _roll(ae_v, 3, "med")
        cols["e19_rwy_ae_p90_5"][sel] = _roll(ae_v, 5, "p90")
        cols["e19_rwy_aobt_p1"][sel] = _lag(aobt_v, 1)
        cols["e19_rwy_aobt_mean3"][sel] = _roll(aobt_v, 3, "mean")
        cols["e19_rwy_aobt_p90_5"][sel] = _roll(aobt_v, 5, "p90")
        cols["e19_rwy_sched_p1"][sel] = _lag(sched_v, 1)
        cols["e19_rwy_sched_mean3"][sel] = _roll(sched_v, 3, "mean")
        cols["e19_rwy_sched_p90_5"][sel] = _roll(sched_v, 5, "p90")
        ae0 = np.where(np.isfinite(ae_v), ae_v, 0.0)
        dly = (ae0 > DLY).astype(np.float64)
        cols["e19_rwy_ndly_5"][sel] = _roll_sum(dly, 5)
        cols["e19_rwy_max_ae_5"][sel] = _roll(ae_v, 5, "max")
        cols["e19_consec_rwy_dly"][sel] = _consec((ae0 > DLY).astype(np.int8))

    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])


def _count_windows(times: np.ndarray, query: np.ndarray, wins) -> dict:
    out = {}
    for w in wins:
        out[w] = np.searchsorted(times, query, side="left") - np.searchsorted(
            times, query - w * 60 * 10**9, side="left"
        )
    return out


def add_pressure(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Family D: counts, arrival-delay counts, spacing."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan, dtype=np.float64) for c in D_COLS}
    wins = [2, 5, 10, 20, 30]

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = _str(dep["RUNWAY_mvt"])

    arr = arr.sort(["airport", "MVT_TIME_UTC_mvt"])
    a_ap = arr["airport"].to_numpy()
    a_t = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    a_dly = arr["arr_sched_delay"].to_numpy().astype(np.float64)
    d15 = (a_dly > ARR_DLY).astype(np.float64)
    dc = np.concatenate([[0.0], np.cumsum(d15)])

    for a in AIRPORTS:
        qidx = np.where(ap == a)[0]
        if qidx.size == 0:
            continue
        t = mvt[qidx]
        for w, cnt in _count_windows(t, t, wins).items():
            cols[f"e19_dep_{w}m"][qidx] = cnt
        rw = rwy[qidx]
        for r in np.unique(rw):
            m = rw == r
            t_r = t[m]
            for w in wins:
                lo = np.searchsorted(t_r, t - w * 60 * 10**9, side="left")
                hi = np.searchsorted(t_r, t, side="left")
                cols[f"e19_rwy_dep_{w}m"][qidx] = np.where(m, (hi - lo).astype(np.float64), cols[f"e19_rwy_dep_{w}m"][qidx])
        aidx = np.where(a_ap == a)[0]
        if aidx.size:
            at = a_t[aidx]
            adlo = {w: np.searchsorted(at, t - w * 60 * 10**9, side="left") for w in wins}
            ahi = np.searchsorted(at, t, side="left")
            for w in wins:
                cols[f"e19_arr_{w}m"][qidx] = ahi - adlo[w]
            dlo5 = np.searchsorted(at, t - 5 * 60 * 10**9, side="left")
            dlo10 = np.searchsorted(at, t - 10 * 60 * 10**9, side="left")
            cols["e19_arr_dly_5m"][qidx] = dc[ahi] - dc[dlo5]
            cols["e19_arr_dly_10m"][qidx] = dc[ahi] - dc[dlo10]

    g1 = dep["e19_gap1"].to_numpy()
    rg1 = dep["e19_rwy_gap1"].to_numpy()
    gmed5 = dep["e19_gap_med5"].to_numpy()
    cols["e19_inv_gap1"] = np.where(np.isfinite(g1) & (g1 > 0), 60.0 / g1, 0.0)
    cols["e19_inv_rwy_gap1"] = np.where(np.isfinite(rg1) & (rg1 > 0), 60.0 / rg1, 0.0)
    cols["e19_gap_compression"] = np.where(np.isfinite(g1) & np.isfinite(gmed5) & (gmed5 > 0), g1 / gmed5, 1.0)

    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])


def attach_shock(train: pl.DataFrame, val: pl.DataFrame) -> pl.DataFrame:
    """Family C: fit expected AOBT-EOBT on train (matched), apply to val rows."""
    matched = train.filter(~pl.col("unmatched"))
    hr = matched.group_by(["airport", "hour"]).agg(pl.col("aobt_eobt").mean().alias("_m"))
    rwy_t = matched.filter(pl.col("RUNWAY_mvt").is_not_null()).group_by(["airport", "RUNWAY_mvt"]).agg(
        pl.col("aobt_eobt").mean().alias("_m")
    )
    ap_mean = matched.group_by("airport").agg(pl.col("aobt_eobt").mean().alias("_m"))

    v = val.join(hr, on=["airport", "hour"], how="left").join(ap_mean, on="airport", how="left")
    exp_hr = np.where(np.isfinite(v["_m"].to_numpy()), v["_m"].to_numpy(), v["_m_right"].to_numpy())
    v = v.drop("_m").drop("_m_right")
    v = v.join(rwy_t, on=["airport", "RUNWAY_mvt"], how="left").join(ap_mean, on="airport", how="left")
    exp_rwy = np.where(np.isfinite(v["_m"].to_numpy()), v["_m"].to_numpy(), v["_m_right"].to_numpy())

    ae1 = v["e19_ae_p1"].to_numpy()
    ae3 = v["e19_ae_mean3"].to_numpy()
    ae5 = v["e19_ae_med5"].to_numpy()
    ae90 = v["e19_ae_p90_5"].to_numpy()
    rwy_ae1 = v["e19_rwy_ae_p1"].to_numpy()
    out = {
        "e19_exp_hr": exp_hr,
        "e19_exp_rwy": exp_rwy,
        "e19_shock_p1": ae1 - exp_hr,
        "e19_shock_mean3": ae3 - exp_hr,
        "e19_shock_med5": ae5 - exp_hr,
        "e19_shock_p90_5": ae90 - exp_hr,
        "e19_shock_rwy": rwy_ae1 - exp_rwy,
    }
    return v.with_columns([pl.Series(k, x) for k, x in out.items()])


def add_interactions(dep: pl.DataFrame) -> pl.DataFrame:
    """Families E + tail products."""
    g = dep
    rwy10 = g["e19_rwy_dep_10m"].to_numpy()
    arr10 = g["e19_arr_10m"].to_numpy()
    dep10 = g["e19_dep_10m"].to_numpy()
    shock5 = g["e19_shock_med5"].to_numpy()
    shock90 = g["e19_shock_p90_5"].to_numpy()
    dis = g["dis_state_30m"].to_numpy()
    rwyd = g["e19_rwy_ae_mean3"].to_numpy()
    rwy30 = g["e19_rwy_dep_30m"].to_numpy()
    arr_dly10 = g["e19_arr_dly_10m"].to_numpy()

    qp = rwy10 * shock5
    out = {
        "e19_qp": qp,
        "e19_qp2": qp * qp,
        "e19_qp_dis": qp * dis,
        "e19_qp_rwyd": qp * rwyd,
        "e19_qp_dens": qp * dep10,
        "e19_arr_dep_pressure": arr10 * dep10,
        "e19_arr_rwy_pressure": arr10 * rwy10,
        "e19_arr_dly_pressure": arr_dly10 * arr10,
        "e19_rwy_dis_pressure": rwy30 * dis,
        "e19_arr_dep_ratio": arr10 / (dep10 + 1.0),
        "e19_tailprod1": qp * shock90,
        "e19_tailprod2": rwy30 * dis,
        "e19_tailprod3": arr10 * rwy30,
        "e19_tailprod4": g["e19_n_dly_str_5"].to_numpy() * g["e19_consec_dly"].to_numpy(),
    }
    return dep.with_columns([pl.Series(k, v) for k, v in out.items()])


def _str(s: pl.Series) -> np.ndarray:
    return s.fill_null("").to_numpy().astype(str)
