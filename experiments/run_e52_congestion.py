"""E52 — ranking-safe surface congestion + recent-history features into v13.

Adds to the E50 (v13) Δ-hat / leftover harness: active-surface counts at AOBT
(interval overlap, ranking-safe), rolling mean-P (=MVT-AOBT of recent flights,
a ranking-safe taxi-speed proxy), and dep/arr pressure imbalance. NO TAXITIME
rolling stats (not ranking-safe). Same cross-month OOF evaluation.
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
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402
from run_e50_matched_rich import top_sse  # noqa: E402

RES = HERE / "results" / "E52"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CONG = ["act_dep", "act_dep_rwy", "act_dep_heavy", "p_mean10", "p_mean30", "p_rwy_mean30",
        "dep_rate5", "arr_rate5", "dep_arr_ratio5", "n_prev10"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
NUM = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
] + CONG


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def fit_lgb(tr, target, leaves=31, seed=SEED, min_child=80, n=400, lam=5.0, lr=0.04):
    m = lgb.LGBMRegressor(
        n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=min_child,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=lam, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pc(s, x, side="right"):
    return np.searchsorted(s, x, side=side).astype(float)


def add_congestion(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "AOBT_3_flt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in CONG}
    t = dep["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    mv = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    wtc = dep["WK_TBL_CAT_flt"].fill_null("").cast(pl.String).to_numpy()
    P = np.clip((mv - t) / 1e9, 0, None)
    # arrival MVT per airport (from ARR stream) — reuse E41 load
    arr = (pl.scan_parquet(__import__("common").TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt"]).collect().sort(["airport", "MVT_TIME_UTC_mvt"]))
    amv = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    aap = arr["airport"].to_numpy()
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        ti, mi = t[sel], mv[sel]
        aobt_s, mvt_s = np.sort(ti), np.sort(mi)
        cols["act_dep"][sel] = pc(aobt_s, ti) - pc(mvt_s, ti)
        # P rolling means (cumsum over sorted AOBT)
        order = np.argsort(ti, kind="stable")
        ts = ti[order]; Ps = P[sel][order]
        cs = np.concatenate([[0.0], np.cumsum(Ps)])
        for w, nm in ((600, "p_mean10"), (1800, "p_mean30")):
            lo = np.searchsorted(ts, ts - w * 10**9, "left")
            cnt = np.arange(len(ts)) - lo
            mean = (cs[np.arange(len(ts)) + 1] - cs[lo]) / np.maximum(cnt, 1)
            cols[nm][sel] = mean[np.argsort(order)]
        cols["n_prev10"][sel] = (np.arange(len(ti)) - np.searchsorted(ts, ti - 600 * 10**9, "left"))[np.argsort(order)]
        # runway-specific active + P mean
        rwi = rw[sel]
        act_rwy = np.zeros(sel.size); prwy = np.zeros(sel.size)
        for r in np.unique(rwi):
            m = rwi == r
            a_s, v_s = np.sort(ti[m]), np.sort(mi[m])
            act_rwy[m] = pc(a_s, ti[m]) - pc(v_s, ti[m])
            Pm = P[sel][m]
            tsr = ti[m]; ordr = np.argsort(tsr, kind="stable")
            csr = np.concatenate([[0.0], np.cumsum(Pm[ordr])])
            lo = np.searchsorted(tsr[ordr], tsr - 1800 * 10**9, "left")
            k = np.arange(len(tsr))
            pr = (csr[k + 1] - csr[lo]) / np.maximum(k - lo, 1)
            prwy[m] = pr[np.argsort(ordr)]
        cols["act_dep_rwy"][sel] = act_rwy
        cols["p_rwy_mean30"][sel] = prwy
        # heavy active on runway
        act_h = np.zeros(sel.size)
        for r in np.unique(rwi):
            m = rwi == r
            hm = m & (wtc[sel] == "H")
            if hm.sum():
                a_s, v_s = np.sort(ti[hm]), np.sort(mi[hm])
                act_h[m] = pc(a_s, ti[m]) - pc(v_s, ti[m])
        cols["act_dep_heavy"][sel] = act_h
        # rates at AOBT
        cols["dep_rate5"][sel] = pc(aobt_s, ti) - pc(aobt_s, ti - 300 * 10**9)
        am = amv[aap == a]
        if am.size:
            am_s = np.sort(am)
            cols["arr_rate5"][sel] = pc(am_s, ti) - pc(am_s, ti - 300 * 10**9)
        else:
            cols["arr_rate5"][sel] = 0.0
        cols["dep_arr_ratio5"][sel] = cols["dep_rate5"][sel] / (cols["arr_rate5"][sel] + 1.0)
    return dep.with_columns([pl.Series(c, cols[c]) for c in CONG])


def load_frames():
    dep = add_congestion(load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
        pl.col("ac_family").fill_null("NA"),
    ]))
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20_pred", e20)), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def hats(tr, ev):
    ytr = tr["y"].to_numpy().astype(float); etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float); eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    d1 = fit_lgb(tr, ytr - Ptr, leaves=31, seed=1)
    rec1 = np.clip(Pev + np.asarray(d1.predict(pdf(ev)), float), 0, None)
    r1 = fit_lgb(tr, ytr - etr, leaves=31, seed=3)
    hat = np.asarray(r1.predict(pdf(ev)), float)
    return {"y": yev, "e": eev, "P": Pev, "rec1": rec1, "hat": hat,
            "ids": ev["MVT_ID_mvt"].to_numpy(), "ap": ev["airport"].to_numpy(),
            "p90": float(np.quantile(etr, 0.90))}


def concat_parts(parts):
    keys = ["y", "e", "P", "rec1", "hat", "ids", "ap"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    return out


def main():
    log("E52 load")
    fr = load_frames()
    parts = []
    for a, b in ((1, 7), (7, 1)):
        parts.append(hats(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    ojj = concat_parts(parts)
    odc = hats(fr["janjul"], fr["dec"])
    payload = {}
    for name, o in (("janjul", ojj), ("dec", odc)):
        y, e = o["y"], o["e"]; p90 = o["p90"]
        def gate(x): return ((e > p90) | (x > 200)).astype(float)
        rec1 = o["rec1"]
        base = {"matched": float(rmse(y, e)), "top1": top_sse(y, e, 0.01), "top5": top_sse(y, e, 0.05)}
        cands = {"e20": base}
        for lam in (0.4, 0.5, 0.6):
            g = gate(rec1 - e)
            p = e + lam * (rec1 - e) * g
            for hlam in (0.15, 0.25):
                ph = p + hlam * o["hat"]
                cands[f"grec{lam}_hat{hlam}"] = {"matched": float(rmse(y, ph)), "top1": top_sse(y, ph, 0.01), "top5": top_sse(y, ph, 0.05)}
        payload[name] = {"e20": base, "cands": cands}
        best = sorted(cands.items(), key=lambda kv: kv[1]["matched"])[:1]
        for n, c in best:
            log(f"  [{name}] best {n}: matched {c['matched']:.2f} (E20 {base['matched']:.2f}, Δ {c['matched']-base['matched']:+.2f}) top1 {100*(c['top1']-base['top1'])/base['top1']:+.1f}% top5 {100*(c['top5']-base['top5'])/base['top5']:+.1f}%")
    (RES / "E52_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
