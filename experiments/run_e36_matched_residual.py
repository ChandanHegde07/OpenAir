"""E36 STEP 5 — direct v9 residual specialist for matched rows (cross-month OOF).

Since v9 == E20 on matched rows, target T_error = y - e20 with causal features.
Cross-month temporal OOF: (Jan fit -> Jul eval)+(Jul fit -> Jan eval) for the
Jan+Jul split; (Jan+Jul fit -> Dec eval) for Dec. Soft gating by a shrinkage
lambda and by a high-error classifier. E20/v9 kept as default.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E36"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt", "hour", "dow", "month", "e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def main():
    dep = load_dep().with_columns([
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("mvt_aobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("aobt_eobt"),
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
    ])
    frames = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        oof = oof.with_columns(pl.Series("e20_pred", 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()))
        f = dep.join(oof.select(["MVT_ID_mvt", "y", "unmatched", "e20_pred"]), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite())
        frames[split] = f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("_m"))
    payload = {}
    lam_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    for split, fit_folds, eval_fold in [
        ("janjul", [(1, 7), (7, 1)], None),
        ("dec", [("jj", 12)], None),
    ]:
        d = frames[split]
        if split == "janjul":
            parts = []
            for a, b in [(1, 7), (7, 1)]:
                tr = d.filter(pl.col("_m") == a); ev = d.filter(pl.col("_m") == b)
                ytr = tr["y"].to_numpy().astype(float); yev = ev["y"].to_numpy().astype(float)
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
                m.fit(pdf(tr), ytr - tr["e20_pred"].to_numpy().astype(float), categorical_feature=CAT)
                c = np.asarray(m.predict(pdf(ev)), float)
                parts.append((yev, ev["e20_pred"].to_numpy().astype(float), c))
        else:
            jj = pl.concat([frames["janjul"].filter(pl.col("_m").is_in([1, 7]))])
            ev = d
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(pdf(jj), jj["y"].to_numpy().astype(float) - jj["e20_pred"].to_numpy().astype(float), categorical_feature=CAT)
            c = np.asarray(m.predict(pdf(ev)), float)
            parts = [(ev["y"].to_numpy().astype(float), ev["e20_pred"].to_numpy().astype(float), c)]
        y = np.concatenate([p[0] for p in parts]); base = np.concatenate([p[1] for p in parts]); corr = np.concatenate([p[2] for p in parts])
        out = {"n": int(len(y)), "e20_rmse": float(rmse(y, base)), "e20_sse": float(np.sum((y - base) ** 2))}
        for lam in lam_grid:
            p = base + lam * corr
            out[f"lam{lam}"] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        # high-error gate: classify |err|>30m using base features
        hi = (np.abs(y - base) > 1800).astype(int)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        # train gate on the same fit data via a quick internal split (approximate): use corr magnitude as proxy
        gate = (np.abs(corr) > np.quantile(np.abs(corr), 0.9)).astype(float)
        p = base + gate * corr
        out["gated_top10corr"] = {"rmse": float(rmse(y, p)), "sse": float(np.sum((y - p) ** 2))}
        payload[split] = out
        log(f"  [{split}] E20 matched {out['e20_rmse']:.2f} (sse {out['e20_sse']:.3e})")
        for lam in [0.1, 0.2, 0.3, 0.5]:
            log(f"      lam{lam}: rmse {out[f'lam{lam}']['rmse']:.2f} sse {out[f'lam{lam}']['sse']:.3e}")
        log(f"      gated: rmse {out['gated_top10corr']['rmse']:.2f}")
    (RES / "E36_matched_residual.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E36_matched_residual.json"))


if __name__ == "__main__":
    main()
