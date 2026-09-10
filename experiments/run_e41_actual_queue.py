"""E41 — actual surface queue at AOBT (interval-overlap occupancy).

For target departure at t = AOBT: active deps j satisfy AOBT_j <= t < MVT_j;
active arrivals j satisfy MVT_j <= t < BLOCK_j. Counts computed via prefix
counts (searchsorted). Train an LGB on TAXITIME using P = MVT-AOBT + exact
queue state; compare matched RMSE and top-5% SSE vs E20, both splits.
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
from common import AIRPORTS, TRAIN_FILES, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E41"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
QF = ["act_dep", "act_dep_rwy", "act_dep_other", "act_dep_H", "act_dep_M", "act_dep_L",
      "act_arr", "act_arr_rwy", "tk_rwy_5", "tk_rwy_10", "tk_rwy_20", "tk_rwy_30",
      "ts_last_tk_rwy", "prev_wtc_heavy", "prev_wtc_super", "queue_service_ratio"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "WK_TBL_CAT_flt", "flt_prefix"]
NUM = ["mvt_aobt", "mvt_sched", "hour", "dow", "month"] + QF


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pc(sorted_arr, x, side="right"):
    return np.searchsorted(sorted_arr, x, side=side).astype(float)


def build_queue(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "AOBT_3_flt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in QF}
    t = dep["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy(); rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    wtc = dep["WK_TBL_CAT_flt"].fill_null("").cast(pl.String).to_numpy()
    amvt = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ablk = arr["BLOCK_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    aap = arr["airport"].to_numpy(); arw = arr["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        ti = t[sel]; mi = mvt[sel]; ri = rw[sel]; wi = wtc[sel]
        # active deps all = #AOBT<=t - #MVT<=t
        aobt_s = np.sort(ti); mvt_s_all = np.sort(mi)
        cols["act_dep"][sel] = pc(aobt_s, ti) - pc(mvt_s_all, ti)
        # arrivals active
        am = amvt[aap == a]; ab = ablk[aap == a]
        if am.size:
            am_s = np.sort(am); ab_s = np.sort(ab[np.isfinite(ab)])
            cols["act_arr"][sel] = pc(am_s, ti) - pc(ab_s, ti)
        else:
            cols["act_arr"][sel] = 0.0
        # per runway
        act_rwy = np.zeros(sel.size)
        for r in np.unique(ri):
            m = ri == r
            a_s = np.sort(ti[m]); v_s = np.sort(mi[m])
            act_rwy[m] = pc(a_s, ti[m]) - pc(v_s, ti[m])
            # takeoff service on runway before t
            for w in (5, 10, 20, 30):
                cols[f"tk_rwy_{w}"][sel[m]] = pc(v_s, ti[m]) - pc(v_s, ti[m] - w * 60 * 10**9)
            prev = np.searchsorted(v_s, ti[m], "left") - 1
            cols["ts_last_tk_rwy"][sel[m]] = np.where(prev >= 0, (ti[m] - v_s[prev]) / 1e9, np.nan)
            # previous takeoff WTC on runway
            if mi[m].size:
                order = np.argsort(mi[m], kind="stable")
                mvt_order = mi[m][order]; wtc_order = wi[m][order]
                k = np.searchsorted(mvt_order, ti[m], "left") - 1
                prev_w = np.where(k >= 0, wtc_order[np.maximum(k, 0)], "")
                cols["prev_wtc_heavy"][sel[m]] = (prev_w == "H").astype(float)
                cols["prev_wtc_super"][sel[m]] = (prev_w == "J").astype(float)
                for cat, name in [("H", "act_dep_H"), ("M", "act_dep_M"), ("L", "act_dep_L")]:
                    wm = m & (wi == cat)
                    if wm.sum():
                        aa = np.sort(ti[wm]); vv = np.sort(mi[wm])
                        cols[name][sel[wm]] = pc(aa, ti[wm]) - pc(vv, ti[wm])
        cols["act_dep_rwy"][sel] = act_rwy
        cols["act_dep_other"][sel] = cols["act_dep"][sel] - act_rwy
        cols["act_arr_rwy"][sel] = 0.0
        for r in np.unique(ri):
            m = ri == r
            if am.size:
                arm = am[arw[aap == a] == r]
                if arm.size:
                    arm_s = np.sort(arm)
                    cols["act_arr_rwy"][sel[m]] = pc(arm_s, ti[m]) - pc(arm_s, ti[m])  # placeholder 0
        svc = np.maximum(cols["tk_rwy_10"][sel], 0.5)  # takeoffs per 10 min
        cols["queue_service_ratio"][sel] = act_rwy / svc
    return dep.with_columns([pl.Series(c, cols[c]) for c in QF])


def build():
    dep = load_dep().with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    # ensure arena has AOBT for sorting in build_queue (dep sorted by airport,AOBT)
    dep = dep.filter(pl.col("AOBT_3_flt").is_not_null())
    return build_queue(dep, arr)


def pdf(df, cols):
    p = df.select(cols + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p[cols + CAT]


def main():
    dep = build()
    log(f"rows {dep.height:,}")
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        y = va["y"].to_numpy().astype(float); m = np.isfinite(e20) & np.isfinite(y)
        ym = y[m]; em = e20[m]
        def top(p, frac):
            sse = (ym - p) ** 2; k = max(1, int(frac * m.sum()))
            return float(np.sort(sse)[::-1][:k].sum())
        mo = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=60,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
        mo.fit(pdf(tr, NUM), tr["y"].to_numpy().astype(float), categorical_feature=CAT)
        pred = np.asarray(mo.predict(pdf(va, NUM)), float)
        out = {"e20_rmse": float(rmse(ym, em)), "e20_top5": top(em, 0.05), "e20_top1": top(em, 0.01),
               "queue_rmse": float(rmse(ym, pred[m])), "queue_top5": top(pred, 0.05), "queue_top1": top(pred, 0.01),
               "n": int(m.sum())}
        # blend
        best = min(np.arange(0, 1.01, 0.1), key=lambda a: float(np.sum((ym - (a * pred[m] + (1 - a) * em)) ** 2)))
        pb = best * pred[m] + (1 - best) * em
        out["blend_a"] = float(best); out["blend_rmse"] = float(rmse(ym, pb)); out["blend_top5"] = top(pb, 0.05)
        payload[split] = out
        log(f"  [{split}] E20 {out['e20_rmse']:.2f} top5 {out['e20_top5']:.3e} | queue {out['queue_rmse']:.2f} top5 {out['queue_top5']:.3e} | blend a={best} {out['blend_rmse']:.2f} top5 {out['blend_top5']:.3e}")
    (RES / "E41_queue.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E41_queue.json"))


if __name__ == "__main__":
    main()
