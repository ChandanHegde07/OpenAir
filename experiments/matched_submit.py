"""Shared matched-tail features, LGB, grec blend, and submission packing.

Change only the recipe (Δ mix, λ, gate, leftover) at the call site.
Always splice onto v8 for unmatched / LIRF G. Ranking never enters fits.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

from common import ROOT, load_dep
from run_submitting_check import SUBMIT_PATH

HERE = Path(__file__).resolve().parent
def _v8_path() -> Path:
    for p in (
        ROOT / "submissions" / "likable-eagle_v8.parquet",
        ROOT / "likable-eagle_v8.parquet",
        ROOT / "experiments" / "results" / "E34" / "likable-eagle_v8.parquet",
    ):
        if p.exists():
            return p
    raise SystemExit("missing likable-eagle_v8.parquet")


V8_PATH = None  # resolved at pack time via _v8_path()
SEED = 1
P90, HTHR = 1462.0, 200.0
NUM0 = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
]
NUM = NUM0 + ["e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
LGB_KW = dict(
    n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1,
)


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df: pl.DataFrame, cols: list[str] | None = None) -> object:
    cols = cols or NUM
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def add_extra(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_aobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("aobt_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_lobt"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_iobt"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
            pl.col("ADEP_mvt").alias("airport"),
            pl.col("AIRCRAFT_TYPE_mvt").str.slice(0, 3).alias("ac_family"),
            pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
        ]
    )
    df = df.with_columns(pl.col("aobt_eobt").abs().alias("abs_aobt_eobt"))
    clocks = df.select(["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"])
    arr = []
    for c in clocks.columns:
        arr.append(clocks[c].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[ns]").astype(np.float64))
    stack = np.stack(arr, axis=1) / 1e9
    with np.errstate(all="ignore"):
        clock_std = np.nanstd(stack, axis=1, ddof=0)
        clock_range = np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)
        n_clocks = np.sum(np.isfinite(stack), axis=1)
    clock_std[~np.isfinite(clock_std)] = np.nan
    clock_range[~np.isfinite(clock_range)] = np.nan
    return df.with_columns(
        pl.Series("clock_std", clock_std),
        pl.Series("clock_range", clock_range),
        pl.Series("n_clocks", n_clocks.astype(np.int32)),
        pl.col("ac_family").fill_null("NA"),
    )


def e20_col(oof: pl.DataFrame) -> pl.DataFrame:
    e = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
    return oof.select("MVT_ID_mvt").with_columns(pl.Series("e20_pred", e))


def attach_oof_e20(dep: pl.DataFrame) -> pl.DataFrame:
    oof = pl.concat(
        [
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_janjul.parquet")),
            e20_col(pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_dec.parquet")),
        ]
    )
    dep = dep.join(oof, on="MVT_ID_mvt", how="left")
    have = dep.filter(pl.col("e20_pred").is_not_null())
    miss = dep.filter(pl.col("e20_pred").is_null())
    fill = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=80,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    fill.fit(pdf(have, NUM0), have["e20_pred"].to_numpy().astype(float), categorical_feature=CAT)
    miss = miss.with_columns(pl.Series("e20_pred", np.asarray(fill.predict(pdf(miss, NUM0)), float)))
    return pl.concat([have, miss], how="vertical_relaxed")


def fit_lgb(X, y, seed=SEED, alpha=None):
    kw = dict(LGB_KW, random_state=seed)
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **kw) if alpha is not None else lgb.LGBMRegressor(**kw)
    m.fit(X, y, categorical_feature=CAT)
    return m


def gated(base: np.ndarray, corr: np.ndarray, p90: float = P90, hthr: float = HTHR) -> np.ndarray:
    return ((base > p90) | (corr > hthr)).astype(float)


def grec(base: np.ndarray, rec: np.ndarray, lam: float = 0.5, p90: float = P90, hthr: float = HTHR) -> np.ndarray:
    return base + lam * (rec - base) * gated(base, rec - base, p90, hthr)


def load_arr_stand(rich: bool = False) -> pl.DataFrame:
    """ARR in-block + taxi-in by stand. Ranking ARR TAXITIME is present and legal."""
    from common import TRAIN_FILES

    cols = [
        pl.col("ADES_mvt").alias("airport"),
        "STAND_mvt",
        pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
    ]
    if rich:
        cols.extend(
            [
                pl.col("SCHED_TIME_UTC_mvt").alias("sibt"),
                "RUNWAY_mvt",
                pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type"),
            ]
        )
    df = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(cols)
        .drop_nulls(["airport", "STAND_mvt", "aibt", "arr_taxiin"])
        .collect()
    )
    if rich and "sibt" in df.columns:
        df = df.with_columns((pl.col("aibt") - pl.col("sibt")).dt.total_seconds().alias("arr_delay"))
    df = df.sort(["airport", "STAND_mvt", "aibt"])
    if rich:
        df = df.with_columns(pl.col("arr_taxiin").shift(1).over(["airport", "STAND_mvt"]).alias("arr_taxiin_2"))
    return df


def attach_arr_stand(dep: pl.DataFrame, arr: pl.DataFrame, rich: bool = False) -> pl.DataFrame:
    d = dep.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"])
    a = arr.sort(["airport", "STAND_mvt", "aibt"])
    j = d.join_asof(a, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "STAND_mvt"], strategy="backward")
    j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("aibt")).dt.total_seconds().alias("since_arr"))
    j = j.with_columns(
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600))
        .then(pl.col("arr_taxiin"))
        .otherwise(None)
        .alias("arr_taxiin"),
        pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600))
        .then(pl.col("since_arr"))
        .otherwise(None)
        .alias("since_arr"),
    )
    if rich:
        if "arr_delay" in j.columns:
            j = j.with_columns(
                pl.when((pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600))
                .then(pl.col("arr_delay"))
                .otherwise(None)
                .alias("arr_delay")
            )
        if "AOBT_3_flt" in j.columns:
            a2 = a.rename({"arr_taxiin": "arr_taxiin_aobt", "aibt": "aibt2"})
            if "arr_delay" in a2.columns:
                a2 = a2.rename({"arr_delay": "arr_delay_drop"})
            j = j.sort(["airport", "STAND_mvt", "AOBT_3_flt"]).join_asof(
                a2.select(["airport", "STAND_mvt", "aibt2", "arr_taxiin_aobt"]),
                left_on="AOBT_3_flt",
                right_on="aibt2",
                by=["airport", "STAND_mvt"],
                strategy="backward",
            )
            j = j.with_columns((pl.col("AOBT_3_flt") - pl.col("aibt2")).dt.total_seconds().alias("since_arr_aobt"))
            j = j.with_columns(
                pl.when((pl.col("since_arr_aobt") > 0) & (pl.col("since_arr_aobt") < 8 * 3600))
                .then(pl.col("arr_taxiin_aobt"))
                .otherwise(None)
                .alias("arr_taxiin_aobt"),
                pl.when((pl.col("since_arr_aobt") > 0) & (pl.col("since_arr_aobt") < 8 * 3600))
                .then(pl.col("since_arr_aobt"))
                .otherwise(None)
                .alias("since_arr_aobt"),
            )
        if "MVT_TIME_UTC_mvt" in dep.columns:
            prev = dep.select(["airport", "STAND_mvt", pl.col("MVT_TIME_UTC_mvt").alias("prev_mvt")]).sort(
                ["airport", "STAND_mvt", "prev_mvt"]
            )
            j = j.sort(["airport", "STAND_mvt", "MVT_TIME_UTC_mvt"]).join_asof(
                prev, left_on="MVT_TIME_UTC_mvt", right_on="prev_mvt", by=["airport", "STAND_mvt"], strategy="backward"
            )
            j = j.with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("prev_mvt")).dt.total_seconds().alias("since_dep"))
            j = j.with_columns(
                pl.when((pl.col("since_dep") > 1) & (pl.col("since_dep") < 12 * 3600))
                .then(pl.col("since_dep"))
                .otherwise(None)
                .alias("since_dep")
            )
        win = (pl.col("since_arr") > 0) & (pl.col("since_arr") < 8 * 3600)
        if "arr_taxiin_2" in j.columns:
            j = j.with_columns(pl.when(win).then(pl.col("arr_taxiin_2")).otherwise(None).alias("arr_taxiin_2"))
        if "arr_type" in j.columns:
            j = j.with_columns(pl.when(win).then(pl.col("arr_type")).otherwise(None).alias("arr_type"))
        if "RUNWAY_mvt" in j.columns and "rwy_arr_taxiin" not in j.columns:
            ar = arr.sort(["airport", "RUNWAY_mvt", "aibt"]).select(
                ["airport", "RUNWAY_mvt", "aibt", pl.col("arr_taxiin").alias("rwy_arr_taxiin")]
            )
            j = j.sort(["airport", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]).join_asof(
                ar, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport", "RUNWAY_mvt"], strategy="backward", suffix="_r"
            )
    return j


def load_matched_train() -> pl.DataFrame:
    dep = add_extra(load_dep()).filter(~pl.col("unmatched")).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
    return attach_oof_e20(dep)


def pack_and_write(version: str, matched_ids: np.ndarray, matched_pred: np.ndarray, note: str = "") -> Path:
    """Splice matched_pred onto v8 by MVT_ID; unmatched stay v8. Write v{N} parquet copies."""
    v8 = pl.read_parquet(_v8_path())
    repl = pl.DataFrame({"MVT_ID_mvt": matched_ids, "new": np.asarray(matched_pred, float)})
    j = v8.rename({"TAXITIME_SEC_mvt": "v8"}).join(repl, on="MVT_ID_mvt", how="left")
    final = np.where(j["new"].is_not_null().to_numpy(), j["new"].to_numpy(), j["v8"].to_numpy())
    final = np.maximum(final.astype(float), 0.0)
    out = j.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", final))
    sub = pl.read_parquet(SUBMIT_PATH)
    if out.height != 344841 or not (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        raise SystemExit("id check failed")
    if out["TAXITIME_SEC_mvt"].null_count() or int(out["TAXITIME_SEC_mvt"].is_nan().sum()):
        raise SystemExit("null/nan")
    nchg = int((np.abs(final - j["v8"].to_numpy()) > 1e-6).sum())
    paths = [
        ROOT / "experiments" / "results" / "submit" / f"likable-eagle_{version}.parquet",
        ROOT / f"likable-eagle_{version}.parquet",
        ROOT / "submissions" / f"likable-eagle_{version}.parquet",
    ]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for p in paths:
        out.write_parquet(p)
        log(f"WROTE {p}")
    print(f"version {version} {note}")
    print("rows", out.height, "changed_vs_v8", nchg)
    print("mean", float(np.mean(final)), "median", float(np.median(final)))
    print("VALIDATION_OK")
    return paths[2]
