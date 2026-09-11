"""E62 — dynamic runway queue / service-rate model on v13 residual.

New structural variables (strictly causal, MVT-ordered same-runway stream):
  - headway-based service rate: rolling median headway (last 5/10/20/50 deps),
    service_rate = 60/median_headway, service_slowdown vs long-term
  - queue position: same-runway active deps (AOBT<=t<MVT interval overlap) and
    recent same-runway MVT counts; scheduled/EOBT queues
  - expected_wait = queue_position x median_service_headway
  - backlog = entering(AOBT) - served(MVT); demand_supply_ratio
Residual LGB (target y - v13) cross-month OOF; alpha/gated blend.
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

RES = HERE / "results" / "E62"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
QF = ["hw_med5", "hw_med10", "hw_med20", "hw_med50", "svc_rate10", "svc_rate50", "svc_slowdown",
      "q_act_rwy", "q_act_ap", "q_mvt10_rwy", "q_mvt30_rwy", "q_eobt10", "exp_wait5", "exp_wait20",
      "backlog30", "demand_supply", "hour_sin", "hour_cos"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]
    NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_lobt", "sd_iobt", "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred"] + QF
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p[NUM + CAT]


def add_queue(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in QF}
    mv = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy(); rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    ao = dep["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    eb = dep["EOBT_1_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        t = mv[sel]; ri = rw[sel]; aoi = ao[sel]; ebi = eb[sel]
        v_ao = ~np.isnat(aoi.astype("datetime64[ns]"))
        aobt_s = np.sort(aoi[v_ao])
        mvt_s = np.sort(t)
        # active deps at t: #AOBT<=t - #MVT<=t  (pushed, not taken off)
        cols["q_act_ap"][sel] = np.searchsorted(aobt_s, t, "right") - np.searchsorted(mvt_s, t, "right")
        # per runway stream: headways and queue
        for r in np.unique(ri):
            m = ri == r
            tm = t[m]
            # rolling median headway over last k deps (window by position)
            hws = np.concatenate([[np.nan], np.diff(tm) / 1e9])
            for k, nm in ((5, "hw_med5"), (10, "hw_med10"), (20, "hw_med20"), (50, "hw_med50")):
                med = np.full(len(tm), np.nan)
                for i in range(1, len(tm)):
                    lo = max(0, i - k)
                    med[i] = np.nanmedian(hws[lo + 1:i + 1])
                cols[nm][sel[m]] = med
            long = np.nanmedian(np.abs(np.diff(tm)) / 1e9)
            cols["svc_rate10"][sel[m]] = 60.0 / np.maximum(cols["hw_med10"][sel[m]], 1.0)
            cols["svc_rate50"][sel[m]] = 60.0 / np.maximum(cols["hw_med50"][sel[m]], 1.0)
            cols["svc_slowdown"][sel[m]] = cols["hw_med10"][sel[m]] / max(long, 1.0)
            # same-runway active + recent MVT counts
            # same-runway active queue at t: deps with AOBT<=t<MVT on this runway
            aoi_r = aoi[np.isin(np.arange(len(t)), np.flatnonzero(ri == r)) & v_ao]
            ao_r = np.sort(aoi_r)
            mv_r = np.sort(tm)
            cols["q_act_rwy"][sel[m]] = np.searchsorted(ao_r, tm, "right") - np.searchsorted(mv_r, tm, "right")
            for w, nm in ((10, "q_mvt10_rwy"), (30, "q_mvt30_rwy")):
                cols[nm][sel[m]] = np.searchsorted(mv_r, tm, "right") - np.searchsorted(mv_r, tm - w * 60 * 10**9, "right")
            # expected wait = q_act_rwy * med headway
            cols["exp_wait5"][sel[m]] = cols["q_act_rwy"][sel[m]] * np.maximum(cols["hw_med5"][sel[m]], 60)
            cols["exp_wait20"][sel[m]] = cols["q_act_rwy"][sel[m]] * np.maximum(cols["hw_med20"][sel[m]], 60)
            # backlog = entering(AOBT in 30m) - served(MVT in 30m)
            entering = np.searchsorted(ao_r, tm, "right") - np.searchsorted(ao_r, tm - 1800 * 10**9, "right")
            served = np.searchsorted(mv_r, tm, "right") - np.searchsorted(mv_r, tm - 1800 * 10**9, "right")
            cols["backlog30"][sel[m]] = entering - served
        # EOBT queue on airport: #EOBT<=t within 10m
        eb_s = np.sort(eb[np.isfinite(eb)])
        cols["q_eobt10"][sel] = np.searchsorted(eb_s, t, "right") - np.searchsorted(eb_s, t - 600 * 10**9, "right")
        svc = np.maximum(cols["svc_rate50"][sel], 0.1)
        cols["demand_supply"][sel] = cols["q_act_ap"][sel] / svc
        h = (dep["hour"].to_numpy()[sel]).astype(float)
        cols["hour_sin"][sel] = np.sin(2 * np.pi * h / 24)
        cols["hour_cos"][sel] = np.cos(2 * np.pi * h / 24)
    return dep.with_columns([pl.Series(c, cols[c]) for c in QF])


def main():
    dep = add_queue(load_dep().with_columns([
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
        pl.col("MARKET_SEGMENT_flt").fill_null("NA"), pl.col("ac_family").fill_null("NA"),
    ]))
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20_pred", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))

    def fit_res(tr, target):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(tr), target, categorical_feature=["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"])
        return m

    def top1(y, p):
        e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
        k = max(1, int(0.01 * len(e2)))
        return float(np.sort(e2)[::-1][:k].sum())

    payload = {}
    for split in ("janjul", "dec"):
        if split == "janjul":
            ys, v13s = [], []
            for a, b in ((1, 7), (7, 1)):
                fit = fr["janjul"].filter(pl.col("_m") == a); ev = fr["janjul"].filter(pl.col("_m") == b)
                v = ev["e20_pred"].to_numpy().astype(float)  # base = e20 (grec near v13 for residual proxy)
                ys.append(ev["y"].to_numpy().astype(float)); v13s.append(v)
            y = np.concatenate(ys); v13 = np.concatenate(v13s)
            # train residual model on Jan, apply to Jul, and vice versa
            corr = np.concatenate([(lambda fit, ev: np.asarray(fit_res(fit, fit["y"].to_numpy().astype(float) - fit["e20_pred"].to_numpy().astype(float)).predict(pdf(ev)), float))(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)) for a, b in ((1, 7), (7, 1))])
        else:
            fit = fr["janjul"]; ev = fr["dec"]
            v = ev["e20_pred"].to_numpy().astype(float)
            y = ev["y"].to_numpy().astype(float); v13 = v
            corr = np.asarray(fit_res(fit, fit["y"].to_numpy().astype(float) - fit["e20_pred"].to_numpy().astype(float)).predict(pdf(ev)), float)
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
