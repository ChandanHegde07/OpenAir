"""Shared loading, splits, metrics, and helpers for taxi-out experiments."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Training-only corpus. Ranking/submitting must never enter loaders, stats, or fits.
FORBIDDEN_STEMS = ("ranking", "submitting", "submission", "test", "eval")
TRAIN_FILES = sorted(glob.glob(str(DATA / "training_*.parquet")))


def _assert_training_path(path: str | Path) -> Path:
    p = Path(path)
    stem = p.stem.lower()
    if not p.name.startswith("training_"):
        raise RuntimeError(
            f"Refusing non-training parquet: {p}. "
            "Research loaders may only read data/training_*.parquet."
        )
    if any(s in stem for s in FORBIDDEN_STEMS):
        raise RuntimeError(f"Refusing forbidden parquet: {p}")
    return p


if not TRAIN_FILES:
    raise RuntimeError(f"No training_*.parquet files found under {DATA}")
TRAIN_FILES = [str(_assert_training_path(p)) for p in TRAIN_FILES]

AIRPORTS = [
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LSZH",
    "LTFM",
]
FOCUS_AIRPORTS = ["LIRF", "EGLL", "LFPG", "LTFM"]

DEP_COLS = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "AOBT_3_flt",
    "ARVT_1_flt",
    "ARVT_3_flt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_TYPE_flt",
    "ADES_FILED_flt",
]


def sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def load_dep() -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(p).select(DEP_COLS) for p in TRAIN_FILES])
    df = (
        lf.filter(pl.col("PHASE_mvt") == "DEP")
        .with_columns(
            pl.col("ADEP_mvt").alias("airport"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_aobt"),
            sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_eobt"),
            sec("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_iobt"),
            sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_lobt"),
            sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_sched"),
            sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt_eobt"),
            sec("AOBT_3_flt", "IOBT_flt").alias("aobt_iobt"),
            sec("AOBT_3_flt", "LOBT_flt").alias("aobt_lobt"),
            sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt_sched"),
            sec("EOBT_1_flt", "IOBT_flt").alias("eobt_iobt"),
            sec("EOBT_1_flt", "LOBT_flt").alias("eobt_lobt"),
            sec("IOBT_flt", "LOBT_flt").alias("iobt_lobt"),
            pl.col("AOBT_3_flt").is_null().alias("unmatched"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
            pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("sched_hour"),
            pl.col("AIRCRAFT_TYPE_mvt").str.slice(0, 3).alias("ac_family"),
        )
        .collect()
    )
    # clock spread across available off-block clocks (seconds)
    clocks = df.select(["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"])
    arr = []
    for c in clocks.columns:
        arr.append(clocks[c].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[ns]").astype(np.float64))
    stack = np.stack(arr, axis=1) / 1e9  # seconds since epoch
    with np.errstate(all="ignore"):
        clock_std = np.nanstd(stack, axis=1, ddof=0)
        clock_range = np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)
        n_clocks = np.sum(np.isfinite(stack), axis=1)
    clock_std[~np.isfinite(clock_std)] = np.nan
    clock_range[~np.isfinite(clock_range)] = np.nan
    df = df.with_columns(
        pl.Series("clock_std", clock_std),
        pl.Series("clock_range", clock_range),
        pl.Series("n_clocks", n_clocks.astype(np.int32)),
        pl.col("mvt_aobt").abs().alias("abs_mvt_aobt"),
        pl.col("aobt_eobt").abs().alias("abs_aobt_eobt"),
        pl.col("aobt_iobt").abs().alias("abs_aobt_iobt"),
        pl.col("aobt_lobt").abs().alias("abs_aobt_lobt"),
    )
    return df.sort(["airport", "MVT_TIME_UTC_mvt"])


def add_causal_rolling(df: pl.DataFrame) -> pl.DataFrame:
    """Previous-completed-flight rolling of MVT-AOBT. Current row excluded via shift(1)."""
    return df.sort(["airport", "MVT_TIME_UTC_mvt"]).with_columns(
        pl.col("mvt_aobt").shift(1).over("airport").alias("lag1_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_mean(5).over("airport").alias("roll5_mean_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_mean(10).over("airport").alias("roll10_mean_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_mean(20).over("airport").alias("roll20_mean_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_median(10).over("airport").alias("roll10_med_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_std(10).over("airport").alias("roll10_std_mvt_aobt"),
        pl.col("mvt_aobt")
        .shift(1)
        .rolling_quantile(quantile=0.75, window_size=10, interpolation="nearest")
        .over("airport")
        .alias("roll10_p75_mvt_aobt"),
        pl.col("mvt_aobt")
        .shift(1)
        .rolling_quantile(quantile=0.90, window_size=10, interpolation="nearest")
        .over("airport")
        .alias("roll10_p90_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).over(["airport", "RUNWAY_mvt"]).alias("lag1_rwy_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_mean(10).over(["airport", "RUNWAY_mvt"]).alias("roll10_rwy_mean_mvt_aobt"),
        pl.col("mvt_aobt").shift(1).rolling_median(10).over(["airport", "RUNWAY_mvt"]).alias("roll10_rwy_med_mvt_aobt"),
    )


def split_by_months(df: pl.DataFrame, val_months: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    va = df.filter(pl.col("month").is_in(val_months))
    tr = df.filter(~pl.col("month").is_in(val_months))
    return tr, va


SPLITS = {
    "janjul": [1, 7],
    "dec": [12],
    "jan": [1],
    "jul": [7],
}


def rmse(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def mae(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[m] - p[m])))


def n_finite(y, p) -> int:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return int((np.isfinite(y) & np.isfinite(p)).sum())


def metrics_block(y, p, unmatched, airport) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    unmatched = np.asarray(unmatched, dtype=bool)
    airport = np.asarray(airport)
    out = {
        "n": n_finite(y, p),
        "rmse": rmse(y, p),
        "mae": mae(y, p),
    }
    m_ok = np.isfinite(y) & np.isfinite(p)
    matched = m_ok & ~unmatched
    un = m_ok & unmatched
    out["n_matched"] = int(matched.sum())
    out["rmse_matched"] = rmse(y[matched], p[matched]) if matched.any() else float("nan")
    out["mae_matched"] = mae(y[matched], p[matched]) if matched.any() else float("nan")
    out["n_unmatched"] = int(un.sum())
    out["rmse_unmatched"] = rmse(y[un], p[un]) if un.any() else float("nan")
    out["mae_unmatched"] = mae(y[un], p[un]) if un.any() else float("nan")
    gt30 = m_ok & (y > 1800)
    gt1h = m_ok & (y > 3600)
    out["n_gt30m"] = int(gt30.sum())
    out["rmse_gt30m"] = rmse(y[gt30], p[gt30]) if gt30.any() else float("nan")
    out["n_gt1h"] = int(gt1h.sum())
    out["rmse_gt1h"] = rmse(y[gt1h], p[gt1h]) if gt1h.any() else float("nan")
    for ap in AIRPORTS:
        sel = m_ok & (airport == ap)
        out[f"n_{ap}"] = int(sel.sum())
        out[f"rmse_{ap}"] = rmse(y[sel], p[sel]) if sel.any() else float("nan")
        out[f"mae_{ap}"] = mae(y[sel], p[sel]) if sel.any() else float("nan")
    return out


def fill_with_fallback(primary, fallback):
    primary = np.asarray(primary, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    return np.where(np.isfinite(primary), primary, fallback)


def group_mean_predictor(train: pl.DataFrame, val: pl.DataFrame, keys: list[str], ycol: str = "y") -> np.ndarray:
    g = train.group_by(keys).agg(pl.col(ycol).mean().alias("_p"))
    j = val.join(g, on=keys, how="left")
    return j["_p"].to_numpy().astype(np.float64)


def airport_mean_fallback(train: pl.DataFrame, val: pl.DataFrame) -> np.ndarray:
    ap = group_mean_predictor(train, val, ["airport"])
    glob = float(train["y"].mean())
    return np.where(np.isfinite(ap), ap, glob)


def ols_predict(x_train, y_train, x_val):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_val = x_val.reshape(-1, 1)
    m = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    X = np.column_stack([np.ones(m.sum()), x_train[m]])
    coef, *_ = np.linalg.lstsq(X, y_train[m], rcond=None)
    mv = np.all(np.isfinite(x_val), axis=1)
    pred = np.full(len(x_val), np.nan)
    pred[mv] = coef[0] + x_val[mv] @ coef[1:]
    return pred, coef


def fmt(d: dict, keys=None) -> str:
    if keys is None:
        keys = [
            "n",
            "rmse",
            "mae",
            "rmse_matched",
            "rmse_unmatched",
            "rmse_gt30m",
            "rmse_gt1h",
            "rmse_LIRF",
            "rmse_EGLL",
            "rmse_LFPG",
            "rmse_LTFM",
        ]
    parts = []
    for k in keys:
        v = d.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            parts.append(f"{k}=NA")
        elif k == "n" or k.startswith("n_"):
            parts.append(f"{k}={int(v):,}")
        else:
            parts.append(f"{k}={v:.2f}")
    return "  ".join(parts)


def save_result(exp_id: str, payload: dict) -> Path:
    path = RESULTS / f"{exp_id}.json"

    def _conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    path.write_text(json.dumps(payload, indent=2, default=_conv), encoding="utf-8")
    return path
