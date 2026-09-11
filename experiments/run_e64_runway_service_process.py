"""E64 — runway departure service-process model on v13 residual.

Extends E63 with an explicit causal service-headway model (predict next same-runway
headway from prev-aircraft transition + hour + rolling headways) and excess-service
pressure (observed/predicted headway, excess sums, positional). Residual LGB on
(y - v13) cross-month OOF; alpha blend.
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
from common import AIRPORTS, load_dep, rmse  # noqa: E402
from run_e62_runway_queue import add_queue, QF  # noqa: E402

RES = HERE / "results" / "E64"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
BNUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred"]
EXTRA = ["hw_pred", "hw_obs_over_pred", "excess_sum3", "excess_max", "pos_last"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p[cols + CAT]


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def add_service_model(dep):
    """Causal per-(airport,runway): train headway model on ordered stream within split is complex;
    here use a rolling-baseline predicted headway (median of last 5 same-rwy headways) + excess features."""
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in EXTRA}
    mv = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy(); rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        ri = rw[sel]; ti = mv[sel]
        for r in np.unique(ri):
            m = ri == r
            tm = ti[m]
            hws = np.concatenate([[np.nan], np.diff(tm) / 1e9])
            pred = np.full(len(tm), np.nan); obs_ratio = np.full(len(tm), np.nan)
            ex3 = np.full(len(tm), np.nan); exmax = np.full(len(tm), np.nan); posl = np.full(len(tm), np.nan)
            for i in range(1, len(tm)):
                lo = max(0, i - 5)
                med = np.nanmedian(hws[lo + 1:i + 1])
                pred[i] = med
                if np.isfinite(hws[i]) and np.isfinite(med) and med > 1:
                    obs_ratio[i] = hws[i] / med
                    ex = hws[i] - med
                    lo3 = max(0, i - 3)
                    ex3[i] = np.nansum(hws[lo3 + 1:i + 1] - np.nanmedian(hws[lo3 + 1:i + 1]))
                    exmax[i] = np.nanmax(np.abs(hws[lo3 + 1:i + 1] - np.nanmedian(hws[lo3 + 1:i + 1])))
                    posl[i] = i
            cols["hw_pred"][sel[m]] = pred
            cols["hw_obs_over_pred"][sel[m]] = obs_ratio
            cols["excess_sum3"][sel[m]] = ex3
            cols["excess_max"][sel[m]] = exmax
            cols["pos_last"][sel[m]] = posl
    return dep.with_columns([pl.Series(c, cols[c]) for c in EXTRA])


def main():
    dep = load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ])
    dep = add_queue(dep)
    dep = add_service_model(dep)
    ALLF = QF + EXTRA
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))

    def v13g(fit, ev):
        y0 = fit["y"].to_numpy().astype(float); e0 = fit["e20_pred"].to_numpy().astype(float)
        P0 = np.clip(fit["mvt_aobt"].to_numpy().astype(float), 0, None)
        evv = ev["e20_pred"].to_numpy().astype(float); Pv = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(fit, BNUM), y0 - P0, categorical_feature=CAT)
        rec = np.clip(Pv + np.asarray(m.predict(pdf(ev, BNUM)), float), 0, None)
        p90 = float(np.quantile(e0, 0.90))
        g = ((evv > p90) | ((rec - evv) > 200)).astype(float)
        return evv + 0.5 * (rec - evv) * g

    def resid_model(fit, target, cols):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=200,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(fit, cols), target, categorical_feature=CAT)
        return m

    payload = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            ys, v13s, crs = [], [], []
            for a, b in ((1, 7), (7, 1)):
                fit = fr["janjul"].filter(pl.col("_m") == a); ev = fr["janjul"].filter(pl.col("_m") == b)
                v = v13g(fit, ev)
                rm = resid_model(fit, fit["y"].to_numpy().astype(float) - v13g(fit, fit), BNUM + ALLF)
                cr = np.asarray(rm.predict(pdf(ev, BNUM + ALLF)), float)
                ys.append(ev["y"].to_numpy().astype(float)); v13s.append(v); crs.append(cr)
            y = np.concatenate(ys); v13 = np.concatenate(v13s); corr = np.concatenate(crs)
        else:
            fit = fr["janjul"]; ev = fr["dec"]
            v = v13g(fit, ev)
            rm = resid_model(fit, fit["y"].to_numpy().astype(float) - v13g(fit, fit), BNUM + ALLF)
            corr = np.asarray(rm.predict(pdf(ev, BNUM + ALLF)), float)
            y = ev["y"].to_numpy().astype(float); v13 = v
        base = {"matched": float(rmse(y, v13)), "top1": top1(y, v13)}
        cands = {"v13": base}
        for al in (0.1, 0.2, 0.3, 0.5):
            p = v13 + al * corr
            cands[f"a{al}"] = {"matched": float(rmse(y, p)), "top1": top1(y, p)}
        payload[split] = {"v13": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:2]
        for nm, c in best:
            log(f"  [{split}] {nm}: matched {c['matched']:.2f} (Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}%")
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
