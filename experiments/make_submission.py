"""Fit E18-H on all training months, score ranking DEP, write submission parquet.

Fits use data/training_*.parquet only. Ranking is the prediction-time
information set (no DEP BLOCK/TAXITIME in any fit). submitting.parquet is
the ID template.

E18-H: matched-only geo_mean tables + LIRF-override rows dropped from
residual training + always-on LIRF unmatched MVT−SCHED.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, load_dep  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, attach_hour_baseline, load_arr_delay  # noqa: E402
from run_e18_unmatched_specialist import fit_residual  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_e4_e5 import load_arr  # noqa: E402
from run_submitting_check import (  # noqa: E402
    SUBMIT_PATH,
    featurize,
    load_ranking_score,
    verify_submitting,
)

OUT_PATH = ROOT / "likable-eagle_v3.parquet"
OUT_COPY = ROOT / "analysis" / "submitting_check" / "likable-eagle_v3.parquet"
PRED_PATH = ROOT / "analysis" / "submitting_check" / "tables" / "submitting_predictions_e18h.parquet"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pack_submission(pred: pl.DataFrame) -> pl.DataFrame:
    sub = pl.read_parquet(SUBMIT_PATH)
    filled = (
        sub.with_row_index("_i")
        .drop("TAXITIME_SEC_mvt")
        .join(pred.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt", how="left")
        .sort("_i")
        .drop("_i")
        .select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    )

    errors: list[str] = []
    if filled.height != 344841:
        errors.append(f"row count {filled.height}")
    if filled.columns != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        errors.append(f"columns {filled.columns}")
    if filled["MVT_ID_mvt"].n_unique() != filled.height:
        errors.append("MVT_ID not unique")
    if not (filled["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        errors.append("ID order/values do not match submitting.parquet")
    missing = sub.select("MVT_ID_mvt").join(filled.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").height
    if missing:
        errors.append(f"{missing} submitting IDs missing")
    extra = filled.select("MVT_ID_mvt").join(sub.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").height
    if extra:
        errors.append(f"{extra} extra IDs")
    nulls = filled["TAXITIME_SEC_mvt"].null_count()
    nans = int(filled["TAXITIME_SEC_mvt"].is_nan().sum())
    if nulls or nans:
        errors.append(f"null={nulls} nan={nans}")
    if filled["TAXITIME_SEC_mvt"].dtype != pl.Float64:
        errors.append(f"dtype {filled['TAXITIME_SEC_mvt'].dtype}")
    chk = filled.join(pred.select("MVT_ID_mvt", "TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left", suffix="_src")
    if not np.allclose(
        chk["TAXITIME_SEC_mvt"].to_numpy(),
        chk["TAXITIME_SEC_mvt_src"].to_numpy(),
        equal_nan=False,
    ):
        errors.append("prediction values do not match source predictions for IDs")
    if errors:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(errors))
    return filled


def main() -> None:
    log("E18-H submission: verify submitting.parquet...")
    verify_submitting()

    log("load ranking prediction-time frames (no DEP BLOCK/TAXITIME)...")
    score_dep, score_arr_t, score_arr_d, rank_meta = load_ranking_score()
    sub_ids = pl.read_parquet(SUBMIT_PATH, columns=["MVT_ID_mvt"])
    n_join = sub_ids.join(score_dep.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="inner").height
    log(
        f"  ranking DEP={rank_meta['ranking_dep']:,} unmatched={rank_meta['unmatched']:,} "
        f"LIRF unmatched={rank_meta['lirf_unmatched']:,}"
    )
    log(f"  submitting IDs in ranking DEP: {n_join:,}/{sub_ids.height:,}")
    if n_join != sub_ids.height:
        raise SystemExit("submitting IDs do not fully match ranking DEP")

    log("featurize training_*.parquet (fits live here)...")
    train_raw = load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    train_feat = featurize(train_raw, load_arr(), load_arr_delay())
    log(f"  training DEP {train_feat.height:,}")

    log("featurize ranking DEP (causal within 2026 ranking stream only)...")
    score_feat = featurize(score_dep, score_arr_t, score_arr_d)
    log(f"  ranking DEP {score_feat.height:,}")

    log("E18-H: matched-only geo_mean + drop LIRF-override from residual train...")
    tabs = geometry_tables(train_feat.filter(~pl.col("unmatched")))
    tr = attach_geometry(train_feat, tabs, 30)
    sc = attach_geometry(score_feat, tabs, 30)
    tr = attach_hour_baseline(tr, tr)
    sc = attach_hour_baseline(tr, sc)
    _, pred_s, _ = fit_residual(tr, sc, extra=list(MODEL_DISRUPT_COLS), drop_override_from_train=True)
    if int(np.isfinite(pred_s).sum()) != len(pred_s):
        raise SystemExit(f"non-finite predictions: {int((~np.isfinite(pred_s)).sum())}")

    um = sc["unmatched"].to_numpy()
    ap = sc["airport"].to_numpy()
    n_fb = int((um & (ap == "LIRF")).sum())
    p = np.asarray(pred_s, dtype=np.float64)
    log(
        f"  ranking preds n={len(p):,} mean={float(p.mean()):.1f} med={float(np.median(p)):.1f} "
        f"min={float(p.min()):.1f} max={float(p.max()):.1f} LIRF_fallback={n_fb:,} "
        f"neg={int((p < 0).sum())} gt60m={int((p >= 3600).sum())}"
    )

    pred_pl = sc.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt", p))
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_pl.write_parquet(PRED_PATH)
    log(f"  wrote {PRED_PATH}")

    log("pack submitting.parquet order...")
    filled = pack_submission(pred_pl)
    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(OUT_PATH)
    filled.write_parquet(OUT_COPY)

    reread = pl.read_parquet(OUT_PATH)
    assert reread.height == 344841
    assert reread.columns == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert (reread["MVT_ID_mvt"] == pl.read_parquet(SUBMIT_PATH)["MVT_ID_mvt"]).all()
    assert reread["TAXITIME_SEC_mvt"].null_count() == 0
    rp = reread["TAXITIME_SEC_mvt"].to_numpy()
    qs = np.quantile(rp, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    print("WROTE", OUT_PATH)
    print("COPY", OUT_COPY)
    print("model", "E18-H")
    print("rows", reread.height)
    print("columns", reread.columns)
    print("schema", {k: str(v) for k, v in reread.schema.items()})
    print("unique_ids", reread["MVT_ID_mvt"].n_unique())
    print("null_count", reread["TAXITIME_SEC_mvt"].null_count())
    print("nan_count", int(reread["TAXITIME_SEC_mvt"].is_nan().sum()))
    print("pred_min", float(np.min(rp)))
    print("pred_max", float(np.max(rp)))
    print("pred_mean", float(np.mean(rp)))
    print("pred_median", float(np.median(rp)))
    print("pred_std", float(np.std(rp, ddof=1)))
    print("p1", float(qs[0]))
    print("p5", float(qs[1]))
    print("p25", float(qs[2]))
    print("p50", float(qs[3]))
    print("p75", float(qs[4]))
    print("p95", float(qs[5]))
    print("p99", float(qs[6]))
    print("n_negative", int((rp < 0).sum()))
    print("n_gt_60m", int((rp >= 3600).sum()))
    print("n_lirf_fallback", n_fb)
    print("bytes", OUT_PATH.stat().st_size)
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
