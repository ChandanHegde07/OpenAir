"""Production ranking inference: E20 (v4 incumbent) + temporal correction.

Training-time artifacts only (vocabs, scalers, TCN weights). Ranking is the
prediction-time stream. Ranking TAXITIME/BLOCK never enter the sequence store.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import DATA, load_dep  # noqa: E402
from openair.models.temporal_v2.config import RESULTS, TemporalV2Config, ensure_dirs  # noqa: E402
from openair.models.temporal_v2.dataset import (  # noqa: E402
    PackedMovementStore,
    TaxiSequenceDataset,
    attach_target_aliases,
)
from openair.models.temporal_v2.inference import load_model  # noqa: E402
from openair.models.temporal_v2.movements import load_ranking_movements, load_training_movements  # noqa: E402
from openair.models.temporal_v2.train import predict_dataset  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_submitting_check import SUBMIT_PATH, load_ranking_score, verify_submitting  # noqa: E402

OUT_PATH = ROOT / "likable-eagle_v5.parquet"
OUT_COPY = ROOT / "analysis" / "submitting_check" / "likable-eagle_v5.parquet"


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


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
    errors = []
    if filled.height != 344841:
        errors.append(f"row count {filled.height}")
    if filled["MVT_ID_mvt"].n_unique() != filled.height:
        errors.append("MVT_ID not unique")
    if not (filled["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all():
        errors.append("ID order/values mismatch")
    if filled["TAXITIME_SEC_mvt"].null_count() or int(filled["TAXITIME_SEC_mvt"].is_nan().sum()):
        errors.append("null/nan predictions")
    if errors:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(errors))
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default=str(RESULTS / "bundle_janjul.pkl"))
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--e20", type=str, default=str(ROOT / "likable-eagle_v4.parquet"))
    ap.add_argument("--mode", choices=["residual", "direct"], default="residual")
    ap.add_argument("--blend-w", type=float, default=1.0, help="w*E20 + (1-w)*temporal when mode=direct")
    ap.add_argument("--corr-scale", type=float, default=0.5, help="multiply residual correction (0.5 won Jan+Jul)")
    args = ap.parse_args()
    ensure_dirs()
    verify_submitting()

    ckpt = Path(args.ckpt) if args.ckpt else RESULTS / "checkpoints" / f"janjul_{args.mode}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    bundle = pickle.loads(Path(args.bundle).read_bytes())
    vocabs = bundle["vocabs"]
    seq_scaler = bundle["seq_scaler"]
    if args.mode == "residual":
        ctx_scaler = bundle["ctx_scaler_resid"]
        zero_e20 = False
    else:
        ctx_scaler = bundle["ctx_scaler_direct"]
        zero_e20 = True
    cfg: TemporalV2Config = bundle["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"load model {ckpt} mode={args.mode} device={device}")
    model = load_model(ckpt, vocabs, device, args.mode)

    log("training movement stream (2025)...")
    train_mov = load_training_movements()
    log("ranking movement stream (prediction-time only)...")
    rank_mov = load_ranking_movements(DATA / "ranking.parquet")
    mov = pl.concat([train_mov, rank_mov]).sort(["airport", "t_sec", "MVT_ID_mvt"])
    log(f"  packed movements {mov.height:,}")
    store = PackedMovementStore(mov, vocabs, seq_scaler)

    log("ranking DEP context + train geometry...")
    score_dep, _, _, _ = load_ranking_score()
    score_dep = attach_target_aliases(score_dep.with_columns(pl.lit(0.0).alias("y")))
    train_dep = attach_target_aliases(
        load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    )
    tabs = geometry_tables(train_dep.filter(~pl.col("unmatched")))
    sc = attach_geometry(score_dep, tabs, 30)
    e20 = pl.read_parquet(args.e20).rename({"TAXITIME_SEC_mvt": "e20_pred"})
    sc = sc.join(e20, on="MVT_ID_mvt", how="left")
    if int(sc["e20_pred"].null_count()) != 0:
        raise SystemExit("E20 ranking preds missing IDs")

    ds = TaxiSequenceDataset(
        store, sc, vocabs, ctx_scaler, cfg, require_e20=True, zero_e20_context=zero_e20
    )
    leak = ds.leakage_sample(3000)
    log(f"  ranking leakage audit {leak}")
    if not leak["ok"]:
        raise SystemExit(f"ranking sequence leakage: {leak}")
    log("predict...")
    pred = predict_dataset(model, ds, cfg, device)
    e20_np = ds.e20.astype(np.float64)
    if args.mode == "direct":
        p = args.blend_w * e20_np + (1.0 - args.blend_w) * pred
    else:
        corr = (pred - e20_np) * float(args.corr_scale)
        corr[ds.lirf_override] = 0.0
        p = e20_np + corr
    p[ds.lirf_override] = e20_np[ds.lirf_override]
    log(f"  mode={args.mode} corr_scale={args.corr_scale} blend_w={args.blend_w}")
    if int(np.isfinite(p).sum()) != len(p):
        raise SystemExit("non-finite ranking predictions")
    log(
        f"  n={len(p):,} mean={p.mean():.1f} med={np.median(p):.1f} "
        f"LIRF_fallback={int(ds.lirf_override.sum()):,} neg={int((p < 0).sum()):,} "
        f"gt60m={int((p >= 3600).sum()):,}"
    )
    pred_pl = pl.DataFrame({"MVT_ID_mvt": ds.mvt_id, "TAXITIME_SEC_mvt": p})
    filled = pack_submission(pred_pl)
    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(OUT_PATH)
    filled.write_parquet(OUT_COPY)
    log(f"WROTE {OUT_PATH} rows={filled.height:,}")


if __name__ == "__main__":
    main()
