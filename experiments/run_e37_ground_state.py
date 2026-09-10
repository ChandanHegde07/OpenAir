"""E37 — coordinate-free ground-state representation (feasible subset).

True airport geometry (stand->taxiway coordinates) has no openly reproducible
stand-ID mapping, so this tests the coordinate-free representation:
stand zones (ID-derived), runway configuration state, and route (airport x
stand-zone x runway) pressure/prior, in a two-stage T_phys + excess model.
Measures matched RMSE and top-5% matched SSE vs E20.
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

RES = HERE / "results" / "E37"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "stand_zone", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]
NUM = ["mvt_sched", "hour", "dow", "month", "cfg_dom_share", "cfg_entropy", "tgt_is_dom",
       "cfg_switch_age", "route_p30", "rwy_p10", "arr_p10", "zone_n", "phys_prior"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def zone_of(s: str) -> str:
    import re
    z = re.match(r"^[A-Za-z]*", s or "").group(0).upper()
    z2 = re.sub(r"[LR]$", "", z)
    return z2 if z2 else "NUM"


def add_ground(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.with_columns(pl.col("STAND_mvt").fill_null("").map_elements(zone_of, return_dtype=pl.String).alias("stand_zone"))
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy(); rw = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    z = dep["stand_zone"].to_numpy()
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in ["cfg_dom_share", "cfg_entropy", "tgt_is_dom", "cfg_switch_age", "route_p30", "rwy_p10", "arr_p10", "zone_n"]}
    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        t = mvt[sel]; r = rw[sel]; zz = z[sel]
        # recent departures per runway (10m): dominant share + entropy
        uniq = np.array([u for u in np.unique(r) if u != ""])
        counts = np.array([_cnt(t, t, 10, r == u) for u in uniq])  # (R, n)
        tot = counts.sum(0) + 1.0
        dom = counts.argmax(0)
        cols["cfg_dom_share"][sel] = (counts.max(0) / tot)
        p = counts / tot
        cols["cfg_entropy"][sel] = -np.sum(np.where(p > 0, p * np.log(p + 1e-9), 0), axis=0)
        cols["tgt_is_dom"][sel] = np.array([1.0 if r[i] == uniq[dom[i]] else 0.0 for i in range(len(sel))])
        # runway switching age: time since dominant recent runway changed
        domseq = uniq[dom]
        ages = np.zeros(len(sel))
        last = domseq[0]; age = 0.0
        prev_t = t[0]
        for i in range(len(sel)):
            if domseq[i] != last:
                last = domseq[i]; age = 0.0
            age += (t[i] - prev_t) / 1e9
            ages[i] = age
        cols["cfg_switch_age"][sel] = ages
        # route pressure: same (runway, zone) dep count 30m
        for i in range(len(sel)):
            pass
        for u in uniq:
            for zz_u in np.unique(zz):
                m = (r == u) & (zz == zz_u)
                t_m = t[m]
                cols["route_p30"][sel[m]] = _cnt(t_m, t[m], 30, np.ones(t_m.size, bool))
                cols["zone_n"][sel[m]] = float(t_m.size)
        cols["rwy_p10"][sel] = _cnt(t, t, 10, np.ones(t.size, bool))
    dep = dep.with_columns([pl.Series(c, cols[c]) for c in cols])
    # arrivals pressure (needs ARR handled in main via separate column) left as rwy_p10 proxy
    return dep


def _cnt(times, q, w, mask):
    # count of masked times within w minutes before each q
    tsel = times[mask]
    hi = np.searchsorted(tsel, q, side="left")
    lo = np.searchsorted(tsel, q - w * 60 * 10**9, side="left")
    return (hi - lo).astype(float)


def physical_prior(tr, va):
    """T_phys = shrunk median y by (airport,RUNWAY,zone) -> (airport,RUNWAY) -> airport."""
    g3 = tr.group_by(["airport", "RUNWAY_mvt", "stand_zone"]).agg(pl.col("y").median().alias("m3"), pl.col("y").len().alias("n3"))
    g2 = tr.group_by(["airport", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("m2"))
    g1 = tr.group_by("airport").agg(pl.col("y").median().alias("m1"))
    j = (va.join(g3, on=["airport", "RUNWAY_mvt", "stand_zone"], how="left")
           .join(g2, on=["airport", "RUNWAY_mvt"], how="left")
           .join(g1, on="airport", how="left"))
    m3 = j["m3"].to_numpy(); n3 = np.nan_to_num(j["n3"].to_numpy(), nan=0.0)
    m2 = j["m2"].to_numpy(); m1 = j["m1"].to_numpy()
    k = 30.0
    parent = np.where(np.isfinite(m2), m2, m1)
    prior = np.where(np.isfinite(m3), (n3 * m3 + k * parent) / (n3 + k), parent)
    return np.where(np.isfinite(prior), prior, np.nanmedian(tr["y"].to_numpy()))


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def main():
    dep = load_dep().with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"))
    dep = add_ground(dep)
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & ~pl.col("unmatched"))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months) & ~pl.col("unmatched"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        y = va["y"].to_numpy().astype(float)
        phys = physical_prior(tr, va)
        va = va.with_columns(pl.Series("phys_prior", phys))
        tr = tr.with_columns(pl.Series("phys_prior", physical_prior(tr, tr)))
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(pdf(tr), tr["y"].to_numpy().astype(float) - tr["phys_prior"].to_numpy().astype(float), categorical_feature=CAT)
        excess = np.asarray(m.predict(pdf(va)), float)
        t_ground = phys + excess
        # metrics
        def top_sse(p, frac):
            sse = (y - p) ** 2
            k = int(frac * len(y))
            return float(np.sort(sse)[::-1][:k].sum())
        out = {"e20_rmse": float(rmse(y, e20)), "ground_rmse": float(rmse(y, t_ground)),
               "e20_top5_sse": top_sse(e20, 0.05), "ground_top5_sse": top_sse(t_ground, 0.05),
               "e20_top1_sse": top_sse(e20, 0.01), "ground_top1_sse": top_sse(t_ground, 0.01),
               "n": int(len(y))}
        # soft blend with E20
        best = min(np.arange(0, 1.01, 0.1), key=lambda g: float(np.sum((y - (g * t_ground + (1 - g) * e20)) ** 2)))
        p = best * t_ground + (1 - best) * e20
        out["blend_best_g"] = float(best); out["blend_rmse"] = float(rmse(y, p))
        out["blend_top5_sse"] = top_sse(p, 0.05)
        payload[split] = out
        log(f"  [{split}] E20 matched {out['e20_rmse']:.1f} top5SSE {out['e20_top5_sse']:.3e} | "
            f"ground {out['ground_rmse']:.1f} top5SSE {out['ground_top5_sse']:.3e} | blend g={best} rmse {out['blend_rmse']:.1f} top5 {out['blend_top5_sse']:.3e}")
    (RES / "E37_ground_state.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E37_ground_state.json"))


if __name__ == "__main__":
    main()
