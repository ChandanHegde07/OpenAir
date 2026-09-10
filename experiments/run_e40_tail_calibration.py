"""E40 — regime/tail calibration.

STEP 10/13: is v9 tail-compressed? Measure predicted-vs-actual quantiles and
rank correlation. Then learn a monotonic calibration (isotonic) and a
P-conditioned correction, fit on a DISJOINT month (cross-month OOF), and measure
matched RMSE + top-5%/1% SSE.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E40"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def frames():
    dep = load_dep().select(["MVT_ID_mvt", "airport", "mvt_aobt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "hour"])
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        oof = oof.with_columns(pl.Series("e20", 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()))
        f = dep.join(oof.select(["MVT_ID_mvt", "y", "unmatched", "e20"]), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20").is_finite())
        f = f.with_columns(pl.col("MVT_TIME_UTC_mvt") if False else pl.col("mvt_aobt").alias("P"))
        out[split] = f
    # need month; reload via join of month
    return out


def add_month(f):
    dep = load_dep().select(["MVT_ID_mvt", pl.col("month").alias("_mon")])
    return f.join(dep, on="MVT_ID_mvt", how="left")


def metrics(y, p):
    sse = (y - p) ** 2
    n = len(y)
    def top(frac):
        k = max(1, int(frac * n))
        return float(np.sort(sse)[::-1][:k].sum())
    return {"rmse": float(rmse(y, p)), "sse": float(sse.sum()), "top5": top(0.05), "top1": top(0.01)}


def main():
    raw = frames()
    payload = {}
    for split in ("janjul", "dec"):
        f = add_month(raw[split])
        y = f["y"].to_numpy().astype(float); e = f["e20"].to_numpy().astype(float)
        P = f["P"].to_numpy().astype(float); ap = f["airport"].to_numpy()
        mon = f["_mon"].to_numpy()
        base = metrics(y, e)
        # quantile compression diagnostic
        qs = [0.5, 0.75, 0.9, 0.95, 0.99]
        pred_q = np.quantile(e, qs); true_q = np.quantile(y, qs)
        diag = {"spearman": float(spearmanr(e, y).statistic), "pearson": float(pearsonr(e, y)[0]),
                "pred_q": pred_q.tolist(), "true_q": true_q.tolist(),
                "ratio_true_over_pred": (true_q / pred_q).tolist()}
        # cross-month calibration folds
        if split == "janjul":
            folds = [(mon == 1, mon == 7), (mon == 7, mon == 1)]
        else:
            jj = raw["janjul"]; jj = add_month(jj); jmon = jj["_mon"].to_numpy()
            folds = [(np.ones(len(y), bool), np.ones(len(y), bool))]  # fit on janjul (below), eval dec
        cal_iso = e.copy(); cal_lgb = e.copy(); cal_pscale = e.copy()
        for fit_mask, ev_mask in folds:
            if split == "dec":
                jj = add_month(raw["janjul"])
                yf = jj["y"].to_numpy().astype(float); ef = jj["e20"].to_numpy().astype(float); Pf = jj["P"].to_numpy().astype(float)
            else:
                yf = y[fit_mask]; ef = e[fit_mask]; Pf = P[fit_mask]
            ev = ev_mask
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(ef, yf)
            cal_iso[ev] = iso.predict(e[ev])
            # P-scaled residual: residual = yf - ef ; model residual with features [ef, Pf]; hypothesis residual ~ P*h(X)
            tr_feat = np.column_stack([ef, Pf, np.abs(ef - Pf)])
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(tr_feat, yf - ef)
            cal_lgb[ev] = e[ev] + np.asarray(m.predict(np.column_stack([e[ev], P[ev], np.abs(e[ev] - P[ev])])), float)
            # residual scaled by P
            ratio = np.where(Pf > 60, (yf - ef) / np.maximum(Pf, 60), 0.0)
            mr = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)
            mr.fit(tr_feat, ratio)
            cal_pscale[ev] = e[ev] + P[ev] * np.asarray(mr.predict(np.column_stack([e[ev], P[ev], np.abs(e[ev] - P[ev])])), float)
        out = {"base": base, "diag": diag,
               "isotonic": metrics(y, cal_iso), "lgb_corr": metrics(y, cal_lgb), "pscaled": metrics(y, cal_pscale)}
        payload[split] = out
        log(f"== {split} == base matched {base['rmse']:.2f} top5 {base['top5']:.3e} top1 {base['top1']:.3e}")
        log(f"   spearman {diag['spearman']:.3f}  true/pred q {np.round(diag['ratio_true_over_pred'],2).tolist()}")
        log(f"   isotonic {out['isotonic']['rmse']:.2f} top5 {out['isotonic']['top5']:.3e} | lgbcorr {out['lgb_corr']['rmse']:.2f} top5 {out['lgb_corr']['top5']:.3e} | pscaled {out['pscaled']['rmse']:.2f} top5 {out['pscaled']['top5']:.3e}")
    (RES / "E40_calibration.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E40_calibration.json"))


if __name__ == "__main__":
    main()
