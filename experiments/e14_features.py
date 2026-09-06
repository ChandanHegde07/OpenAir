"""E14: leakage-safe dynamic airport surface-state features.

All features are strictly causal: for a departure at time t they use only
movements with timestamps strictly earlier than t (< t).

Counts, time-since, and rates are computed exactly via numpy searchsorted.
Recent-taxi-behavior statistics use pandas time-based rolling windows over
previous movements (shift(1) excludes the current row), matching the
convention of common.add_causal_rolling.

Note: the taxi-behavior rolling stats consume the TAXITIME of previous
TRAINING movements. They are valid for research fits on training holdouts,
but are NOT ranking-safe at submission time (ranking blanks other DEP
TAXITIME / BLOCK). This is flagged in the E14 result package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AIRPORTS  # noqa: E402

WINDOWS = [5, 10, 15, 30, 60]
RWY_WINDOWS = [10, 30]
NS_MIN = 60 * 10**9

SURFACE_FEATURES = [
    "s_dep_5m",
    "s_dep_10m",
    "s_dep_15m",
    "s_dep_30m",
    "s_dep_60m",
    "s_arr_5m",
    "s_arr_10m",
    "s_arr_15m",
    "s_arr_30m",
    "s_arr_60m",
    "s_rwy_dep_10m",
    "s_rwy_dep_30m",
    "s_rwy_arr_10m",
    "s_rwy_arr_30m",
    "s_ts_dep",
    "s_ts_arr",
    "s_dep_rate_10m",
    "s_dep_rate_30m",
    "s_arr_rate_10m",
    "s_arr_rate_30m",
    "s_taxi_mean_10m",
    "s_taxi_mean_30m",
    "s_taxi_med_30m",
    "s_taxi_p90_30m",
    "s_taxi_std_30m",
    "s_traffic_pressure",
    "s_traffic_acceleration",
    "s_arrival_pressure",
    "s_runway_pressure",
    "s_dep_burst",
    "s_arr_burst",
]


def _counts_past(times: np.ndarray, ts: np.ndarray, window_min: int) -> np.ndarray:
    """Count of events in [t - window, t) for every t in ts (strictly < t)."""
    lo = np.searchsorted(times, ts - window_min * 60 * 10**9, side="left")
    hi = np.searchsorted(times, ts, side="left")
    return hi - lo


def _rolling_taxi(mvt_ns: np.ndarray, taxi: np.ndarray) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(mvt_ns.astype("datetime64[ns]"))
    s = pd.Series(taxi, index=idx)
    r10 = s.rolling("10min")
    r30 = s.rolling("30min")
    return {
        "s_taxi_mean_10m": r10.mean().shift(1).to_numpy(),
        "s_taxi_mean_30m": r30.mean().shift(1).to_numpy(),
        "s_taxi_med_30m": r30.median().shift(1).to_numpy(),
        "s_taxi_p90_30m": r30.quantile(0.9).shift(1).to_numpy(),
        "s_taxi_std_30m": r30.std().shift(1).to_numpy(),
    }


def add_surface_state(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Attach strictly-causal surface-state features to the DEP frame."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols: dict[str, np.ndarray] = {}

    arr = arr.sort(["airport", "MVT_TIME_UTC_mvt"])
    arr_mvt = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    arr_ap = arr["airport"].to_numpy()

    mvt_all = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap_all = dep["airport"].to_numpy()
    rwy_all = dep["RUNWAY_mvt"].to_numpy()
    taxi_all = np.asarray(dep["TAXITIME_SEC_mvt"].to_numpy(), dtype=np.float64)

    for col in SURFACE_FEATURES:
        cols[col] = np.full(n, np.nan, dtype=np.float64)

    for ap in AIRPORTS:
        sel = ap_all == ap
        t = mvt_all[sel]
        if t.size == 0:
            continue
        ar = np.arange(t.size)

        # airport traffic counts (strictly < t)
        for w in WINDOWS:
            cols[f"s_dep_{w}m"][sel] = _counts_past(t, t, w)
        a_times = arr_mvt[arr_ap == ap]
        if a_times.size:
            for w in WINDOWS:
                cols[f"s_arr_{w}m"][sel] = _counts_past(a_times, t, w)
            prev_arr = np.searchsorted(a_times, t, side="left") - 1
            cols["s_ts_arr"][sel] = np.where(prev_arr >= 0, (t - a_times[prev_arr]).astype(np.float64) / 1e9, np.nan)

        # runway activity
        rwy = rwy_all[sel].astype(str)
        for r in np.unique(rwy):
            t_r = t[rwy == r]
            m_dep_r = rwy == r
            for w in RWY_WINDOWS:
                lo = np.searchsorted(t_r, t - w * 60 * 10**9, side="left")
                hi = np.searchsorted(t_r, t, side="left")
                cnt = (hi - lo).astype(np.float64)
                cols[f"s_rwy_dep_{w}m"][sel] = np.where(m_dep_r, cnt, cols[f"s_rwy_dep_{w}m"][sel])
        a_rwy_all = arr.filter(pl.col("airport") == ap)["RUNWAY_mvt"].to_numpy().astype(str)
        a_t = arr_mvt[arr_ap == ap]
        if a_t.size:
            for r in np.unique(a_rwy_all):
                m_r = a_rwy_all == r
                t_r = a_t[m_r]
                m_dep_r = rwy == r
                for w in RWY_WINDOWS:
                    ahi = np.searchsorted(t_r, t, side="left")
                    alo = np.searchsorted(t_r, t - w * 60 * 10**9, side="left")
                    cnt = (ahi - alo).astype(np.float64)
                    cols[f"s_rwy_arr_{w}m"][sel] = np.where(m_dep_r, cnt, cols[f"s_rwy_arr_{w}m"][sel])

        # temporal pressure
        prev_dep = np.searchsorted(t, t, side="left") - 1
        cols["s_ts_dep"][sel] = np.where(prev_dep >= 0, (t - t[prev_dep]).astype(np.float64) / 1e9, np.nan)

        # recent taxi behaviour (previous movements only)
        for k, v in _rolling_taxi(t, taxi_all[sel]).items():
            cols[k][sel] = v

    for k in ["s_dep_rate_10m", "s_dep_rate_30m", "s_arr_rate_10m", "s_arr_rate_30m"]:
        cols[k] = cols[k.replace("rate_", "")] / 10.0 if "10m" in k else cols[k.replace("rate_", "")] / 30.0

    cols["s_traffic_pressure"] = cols["s_dep_10m"] + cols["s_arr_10m"]
    cols["s_traffic_acceleration"] = cols["s_dep_10m"] - cols["s_dep_30m"] / 3.0
    cols["s_arrival_pressure"] = cols["s_arr_10m"] - cols["s_arr_30m"] / 3.0
    cols["s_runway_pressure"] = cols["s_rwy_dep_10m"] + cols["s_rwy_arr_10m"]
    cols["s_dep_burst"] = cols["s_dep_5m"] - cols["s_dep_30m"] / 6.0
    cols["s_arr_burst"] = cols["s_arr_5m"] - cols["s_arr_30m"] / 6.0

    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])
