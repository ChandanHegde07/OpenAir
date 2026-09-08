"""Chronological expanding-window E20 OOF for temporal residual targets.

E20 is fit only on earlier months than the predicted months. No random CV.
Ranking/submitting are never loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import load_dep  # noqa: E402
from openair.models.e20_baseline.fit import fit_predict_e20  # noqa: E402
from openair.models.temporal_v2.config import CACHE, RESULTS, ensure_dirs  # noqa: E402
from run_e16a import load_arr_delay  # noqa: E402
from run_e4_e5 import load_arr  # noqa: E402
from run_submitting_check import featurize  # noqa: E402

# Strictly causal folds for the Jan+Jul experiment (val months 1,7 never enter fits).
JANJUL_FOLDS = [
    {"train": [2, 3, 4], "pred": [5, 6]},
    {"train": [2, 3, 4, 5, 6], "pred": [8, 9]},
    {"train": [2, 3, 4, 5, 6, 8, 9], "pred": [10, 11, 12]},
]

# Causal folds for the December experiment (month 12 never enters fits).
DEC_FOLDS = [
    {"train": [1, 2, 3], "pred": [4, 5]},
    {"train": [1, 2, 3, 4, 5], "pred": [6, 7]},
    {"train": [1, 2, 3, 4, 5, 6, 7], "pred": [8, 9]},
    {"train": list(range(1, 10)), "pred": [10, 11]},
]


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_feat() -> pl.DataFrame:
    ensure_dirs()
    path = CACHE / "dep_feat.parquet"
    if path.exists():
        log(f"load cached features {path}")
        return pl.read_parquet(path)
    log("featurize training DEP (once)...")
    dep = load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    feat = featurize(dep, load_arr(), load_arr_delay())
    feat.write_parquet(path)
    log(f"  wrote {path} rows={feat.height:,}")
    return feat


def run_folds(feat: pl.DataFrame, folds: list[dict], tag: str, lite: bool) -> pl.DataFrame:
    parts = []
    for i, f in enumerate(folds, 1):
        log(f"===== {tag} fold {i}/{len(folds)} train={f['train']} pred={f['pred']} =====")
        out = fit_predict_e20(feat, f["train"], f["pred"], log=log, lite=lite)
        out = out.with_columns(pl.lit(i).alias("fold"), pl.lit(tag).alias("scheme"))
        rmse = ((out["y"] - out["e20_pred"]) ** 2).mean() ** 0.5
        log(f"    fold RMSE={float(rmse):.2f} n={out.height:,}")
        parts.append(out)
    return pl.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["janjul", "dec", "both"], default="janjul")
    ap.add_argument("--lite", action="store_true", help="skip XGB expert (tiny blend weight)")
    args = ap.parse_args()
    ensure_dirs()
    feat = load_feat()
    meta = {"generated": datetime.now().isoformat(), "lite": args.lite}
    if args.scheme in ("janjul", "both"):
        oof = run_folds(feat, JANJUL_FOLDS, "janjul", args.lite)
        path = RESULTS / "e20_oof_train_janjul.parquet"
        oof.write_parquet(path)
        log(f"WROTE {path} n={oof.height:,}")
        meta["janjul_n"] = oof.height
    if args.scheme in ("dec", "both"):
        oof = run_folds(feat, DEC_FOLDS, "dec", args.lite)
        path = RESULTS / "e20_oof_train_dec.parquet"
        oof.write_parquet(path)
        log(f"WROTE {path} n={oof.height:,}")
        meta["dec_n"] = oof.height
    (RESULTS / "e20_oof_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
