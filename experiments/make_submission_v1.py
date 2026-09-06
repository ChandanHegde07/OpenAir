"""Build likeable-eagle_v1.parquet from already-generated E16-A predictions.

Does not retrain. Joins predictions onto submitting.parquet by MVT_ID_mvt
and preserves submitting row order.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SUB_PATH = ROOT / "data" / "submitting.parquet"
PRED_PATH = ROOT / "analysis" / "submitting_check" / "tables" / "submitting_predictions.parquet"
OUT_PATH = ROOT / "likeable-eagle_v1.parquet"
OUT_COPY = ROOT / "analysis" / "submitting_check" / "likeable-eagle_v1.parquet"


def main() -> None:
    sub = pl.read_parquet(SUB_PATH)
    pred = pl.read_parquet(PRED_PATH).select(
        "MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
    )

    filled = (
        sub.with_row_index("_i")
        .drop("TAXITIME_SEC_mvt")
        .join(pred, on="MVT_ID_mvt", how="left")
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
    chk = filled.join(pred, on="MVT_ID_mvt", how="left", suffix="_src")
    if not np.allclose(
        chk["TAXITIME_SEC_mvt"].to_numpy(),
        chk["TAXITIME_SEC_mvt_src"].to_numpy(),
        equal_nan=False,
    ):
        errors.append("prediction values do not match source predictions for IDs")
    if errors:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(errors))

    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(OUT_PATH)
    filled.write_parquet(OUT_COPY)

    reread = pl.read_parquet(OUT_PATH)
    assert reread.height == 344841
    assert reread.columns == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert (reread["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all()
    assert reread["TAXITIME_SEC_mvt"].null_count() == 0
    p = reread["TAXITIME_SEC_mvt"].to_numpy()
    qs = np.quantile(p, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    print("WROTE", OUT_PATH)
    print("COPY", OUT_COPY)
    print("rows", reread.height)
    print("columns", reread.columns)
    print("schema", {k: str(v) for k, v in reread.schema.items()})
    print("unique_ids", reread["MVT_ID_mvt"].n_unique())
    print("id_match_count", int((reread["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).sum()))
    print("null_count", reread["TAXITIME_SEC_mvt"].null_count())
    print("nan_count", int(reread["TAXITIME_SEC_mvt"].is_nan().sum()))
    print("pred_min", float(np.min(p)))
    print("pred_max", float(np.max(p)))
    print("pred_mean", float(np.mean(p)))
    print("pred_median", float(np.median(p)))
    print("pred_std", float(np.std(p, ddof=1)))
    print("p1", float(qs[0]))
    print("p5", float(qs[1]))
    print("p25", float(qs[2]))
    print("p50", float(qs[3]))
    print("p75", float(qs[4]))
    print("p95", float(qs[5]))
    print("p99", float(qs[6]))
    print("n_negative", int((p < 0).sum()))
    print("n_gt_60m", int((p >= 3600).sum()))
    print("bytes", OUT_PATH.stat().st_size)
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
