"""E33 submission — likable-eagle_v7.parquet.

E20 (likable-eagle_v4) with the LIRF-unmatched slice replaced by the E33 gate-
delay decomposition:
    G = BLOCK - SCHED  (LightGBM trained on all 2025 LIRF rows)
    T_hat = D - G_hat,  D = MVT - SCHED
    final = 0.6 * T_hat + 0.4 * E20   (alpha=0.6 robust across Jan+Jul and Dec)

All other rows keep the E20 prediction. Validation: Jan+Jul overall 368.0->345.2,
Dec 228.7->226.8. Training-only statistics; ranking used for inference only.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, TRAIN_FILES  # noqa: E402
from run_submitting_check import SUBMIT_PATH  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402

OUT = ROOT / "experiments" / "results" / "E33" / "likable-eagle_v7.parquet"
OUT_ROOT = ROOT / "likable-eagle_v7.parquet"
E20_PATH = ROOT / "analysis" / "submitting_check" / "likable-eagle_v4.parquet"
ALPHA = 0.6
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def lirf_rows(months: list[int] | None):
    dep = pl.scan_parquet(TRAIN_FILES)
    lf = dep.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
    if months is not None:
        lf = lf.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    return lf.select([
        pl.col("MVT_ID_mvt"), pl.col("ADEP_mvt").alias("airport"), "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt",
        "TAXITIME_SEC_mvt",
    ]).collect()


def build_dep_features(df: pl.DataFrame, arr: pl.DataFrame, with_target: bool) -> pl.DataFrame:
    df = df.with_columns([
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"),
    ])
    df = add_surface(df, arr)
    if with_target:
        df = df.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
    return df


def pdf(df, cols):
    p = df.select(cols).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def main():
    log("E33 submission: verify inputs...")
    for p in (SUBMIT_PATH, E20_PATH):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    log("train G model on all 2025 LIRF rows...")
    # arrivals (2025) for surface
    arr25 = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
             .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
             .sort(["airport", "MVT_TIME_UTC_mvt"]))
    tr = build_dep_features(lirf_rows(None), arr25, with_target=True)
    Dt = tr["mvt_sched"].to_numpy().astype(float); yt = tr["y"].to_numpy().astype(float)
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(pdf(tr, NUM + CAT), Dt - yt, categorical_feature=CAT)
    log(f"  G model trained on {tr.height:,} LIRF rows")

    log("load ranking DEP/ARR and featurize...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet")
    rdep = rank.filter(pl.col("PHASE_mvt") == "DEP").select([
        "MVT_ID_mvt", "FLIGHT_mvt", pl.col("ADEP_mvt").alias("airport"), "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt",
    ]).collect()
    rarr = (rank.filter(pl.col("PHASE_mvt") == "ARR")
            .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
            .sort(["airport", "MVT_TIME_UTC_mvt"]))
    rd = build_dep_features(rdep, rarr, with_target=False)
    lirf_mask = (rd["airport"] == "LIRF") & (rd["AOBT_3_flt"].is_null())
    lirf_r = rd.filter(lirf_mask)
    log(f"  ranking DEP {rd.height:,}; LIRF unmatched {lirf_r.height}")

    ghat = np.asarray(model.predict(pdf(lirf_r, NUM + CAT)), dtype=float)
    D_r = lirf_r["mvt_sched"].to_numpy().astype(float)
    t_dec = np.maximum(D_r - ghat, 0.0)

    log("load E20 base submission and blend LIRF slice...")
    e20 = pl.read_parquet(E20_PATH)
    e20 = e20.join(lirf_r.select(["MVT_ID_mvt"]).with_columns(
        pl.Series("e33_pred", ALPHA * t_dec + 0.0)), on="MVT_ID_mvt", how="left")
    # need E20 values on those rows for the (1-alpha) term
    e20v = pl.read_parquet(E20_PATH).rename({"TAXITIME_SEC_mvt": "e20p"})
    j = lirf_r.select(["MVT_ID_mvt"]).with_columns(pl.Series("t_dec", t_dec)).join(e20v, on="MVT_ID_mvt", how="left")
    blended = ALPHA * j["t_dec"].to_numpy() + (1 - ALPHA) * j["e20p"].to_numpy()
    repl = j.select(["MVT_ID_mvt"]).with_columns(pl.Series("TAXITIME_SEC_mvt", blended))

    base = pl.read_parquet(E20_PATH).with_row_index("_i")
    final = (base.join(repl.rename({"TAXITIME_SEC_mvt": "_r"}), on="MVT_ID_mvt", how="left")
                 .with_columns(pl.when(pl.col("_r").is_not_null()).then(pl.col("_r")).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"))
                 .drop("_r").sort("_i").drop("_i").select("MVT_ID_mvt", "TAXITIME_SEC_mvt"))

    # validation
    sub = pl.read_parquet(SUBMIT_PATH)
    if final.height != 344841:
        raise SystemExit(f"row count {final.height}")
    if final["MVT_ID_mvt"].n_unique() != final.height:
        raise SystemExit("duplicate ids")
    if not (final["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        raise SystemExit("id order mismatch")
    if final["TAXITIME_SEC_mvt"].null_count() or int(final["TAXITIME_SEC_mvt"].is_nan().sum()):
        raise SystemExit("null/nan")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(OUT)
    final.write_parquet(OUT_ROOT)
    rp = final["TAXITIME_SEC_mvt"].to_numpy()
    changed = int((rp != pl.read_parquet(E20_PATH)["TAXITIME_SEC_mvt"].to_numpy()).sum())
    print("WROTE", OUT)
    print("COPY", OUT_ROOT)
    print("model E33: E20 + LIRF gate-delay decomposition (alpha 0.6)")
    print("rows", final.height, "changed_rows", changed, "nulls", final["TAXITIME_SEC_mvt"].null_count())
    print("mean", float(np.mean(rp)), "median", float(np.median(rp)), "min", float(np.min(rp)), "max", float(np.max(rp)))
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
