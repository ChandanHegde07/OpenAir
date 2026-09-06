"""E17-A: ranking-safe operational-memory features.

All features are strictly causal: for a departure at time t they use only
movements with event time < t. NONE of them uses any TAXITIME (neither the
current flight's nor any other flight's) — TAXITIME memory is FORBIDDEN here
(E14 used it and is ranking-unsafe).

Feature classes (every feature is RANKING_SAFE):
  airport memory : causal rolling stats of MVT-AOBT, AOBT-EOBT, MVT-SCHED
                   over previous departures at the airport
  runway memory  : same-runway departure activity, gaps, rates, bursts and
                   rolling MVT-AOBT stats for previous same-runway departures

Counting and gaps use exact numpy searchsorted (strictly < t). Rolling
mean/std/median/P90 use pandas time-based windows over previous rows
(shift(1) excludes the current row), matching the add_causal_rolling
convention already used by the frozen baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AIRPORTS  # noqa: E402

RWY_COUNT_WINDOWS = [5, 10, 15, 30, 60]
ROLL_WINDOWS = [5, 10, 15, 30, 60]

AIRPORT_MEMORY_COLS = [
    "rm_aobt_mean_5m",
    "rm_aobt_mean_10m",
    "rm_aobt_mean_15m",
    "rm_aobt_mean_30m",
    "rm_aobt_mean_60m",
    "rm_aobt_med_30m",
    "rm_aobt_p90_30m",
    "rm_aobt_std_30m",
    "rm_ae_mean_5m",
    "rm_ae_mean_10m",
    "rm_ae_mean_15m",
    "rm_ae_mean_30m",
    "rm_ae_mean_60m",
    "rm_ae_med_30m",
    "rm_ae_p90_30m",
    "rm_ae_std_30m",
    "rm_sched_mean_10m",
    "rm_sched_mean_30m",
    "rm_sched_med_30m",
    "rm_sched_p90_30m",
    "rm_sched_std_30m",
]

RUNWAY_MEMORY_COLS = [
    "rm_rwy_dep_5m",
    "rm_rwy_dep_10m",
    "rm_rwy_dep_15m",
    "rm_rwy_dep_30m",
    "rm_rwy_dep_60m",
    "rm_rwy_ts_dep",
    "rm_rwy_rate_30m",
    "rm_rwy_burst",
    "rm_rwy_accel",
    "rm_rwy_aobt_mean_10m",
    "rm_rwy_aobt_mean_30m",
    "rm_rwy_aobt_med_30m",
    "rm_rwy_aobt_p90_30m",
    "rm_rwy_aobt_std_30m",
]


def _counts_past(times: np.ndarray, ts: np.ndarray, window_min: int) -> np.ndarray:
    lo = np.searchsorted(times, ts - window_min * 60 * 10**9, side="left")
    hi = np.searchsorted(times, ts, side="left")
    return hi - lo


def _rolling_past(ts_ns: np.ndarray, v: np.ndarray, window_min: int) -> dict:
    """Rolling mean/std/median/P90 over the previous window of rows, shift(1)."""
    idx = pd.DatetimeIndex(ts_ns.astype("datetime64[ns]"))
    s = pd.Series(v, index=idx)
    r = s.rolling(f"{window_min}min")
    out = {
        "mean": r.mean().shift(1).to_numpy(),
        "med": r.median().shift(1).to_numpy(),
        "p90": r.quantile(0.9).shift(1).to_numpy(),
        "std": r.std().shift(1).to_numpy(),
    }
    return out


def add_airport_memory(dep: pl.DataFrame) -> pl.DataFrame:
    """Causal airport-level rolling memory of MVT-AOBT / AOBT-EOBT / MVT-SCHED."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols: dict[str, np.ndarray] = {}
    for c in AIRPORT_MEMORY_COLS:
        cols[c] = np.full(n, np.nan, dtype=np.float64)

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    aobt = dep["mvt_aobt"].to_numpy()
    ae = dep["aobt_eobt"].to_numpy()
    sched = dep["mvt_sched"].to_numpy()

    for a in AIRPORTS:
        sel = ap == a
        t = mvt[sel]
        if t.size == 0:
            continue
        for var, prefix in [(aobt, "rm_aobt"), (ae, "rm_ae"), (sched, "rm_sched")]:
            v = var[sel]
            for w in ROLL_WINDOWS:
                if prefix == "rm_sched" and w not in (10, 30):
                    continue
                for stat in ("mean", "med", "p90", "std"):
                    key = f"{prefix}_{stat}_{w}m"
                    if key not in cols:
                        continue
                    cols[key][sel] = _rolling_past(t, v, w)[stat]
    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])


def add_runway_memory(dep: pl.DataFrame) -> pl.DataFrame:
    """Causal same-runway operational memory (activity, gaps, MVT-AOBT stats)."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols: dict[str, np.ndarray] = {}
    for c in RUNWAY_MEMORY_COLS:
        cols[c] = np.full(n, np.nan, dtype=np.float64)

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].to_numpy()
    aobt = dep["mvt_aobt"].to_numpy()

    # activity / gaps / rates via per (airport, runway) groups
    groups = set(zip(ap, rwy))
    for (a, r) in sorted(groups):
        sel = (ap == a) & (rwy == r)
        t = mvt[sel]
        if t.size == 0:
            continue
        ar = np.arange(t.size)
        for w in RWY_COUNT_WINDOWS:
            lo = np.searchsorted(t, t - w * 60 * 10**9, side="left")
            cols[f"rm_rwy_dep_{w}m"][sel] = ar - lo
        prev = np.searchsorted(t, t, side="left") - 1
        cols["rm_rwy_ts_dep"][sel] = np.where(prev >= 0, (t - t[prev]).astype(np.float64) / 1e9, np.nan)
        cols["rm_rwy_rate_30m"][sel] = cols["rm_rwy_dep_30m"][sel] / 30.0
        cols["rm_rwy_burst"][sel] = cols["rm_rwy_dep_5m"][sel] - cols["rm_rwy_dep_30m"][sel] / 6.0
        cols["rm_rwy_accel"][sel] = cols["rm_rwy_dep_10m"][sel] - cols["rm_rwy_dep_30m"][sel] / 3.0

        # rolling MVT-AOBT stats for previous same-runway departures
        v = aobt[sel]
        for w in (10, 30):
            idx = pd.DatetimeIndex(t.astype("datetime64[ns]"))
            s = pd.Series(v, index=idx)
            r = s.rolling(f"{w}min")
            cols[f"rm_rwy_aobt_mean_{w}m"][sel] = r.mean().shift(1).to_numpy()
        idx = pd.DatetimeIndex(t.astype("datetime64[ns]"))
        s = pd.Series(v, index=idx)
        r30 = s.rolling("30min")
        cols["rm_rwy_aobt_med_30m"][sel] = r30.median().shift(1).to_numpy()
        cols["rm_rwy_aobt_p90_30m"][sel] = r30.quantile(0.9).shift(1).to_numpy()
        cols["rm_rwy_aobt_std_30m"][sel] = r30.std().shift(1).to_numpy()

    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])
