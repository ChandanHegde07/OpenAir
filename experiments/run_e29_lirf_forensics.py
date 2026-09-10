"""E29 PHASE 7/8 — LIRF + unmatched top-SSE forensics (C28).

Before any model: what do the catastrophic unmatched rows share? Movement-only
fields only. No training. E20 untouched.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E29"
RES.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def enrich(df, key, top_mask, name):
    out = []
    tot = df.height
    for val in df[key].unique().to_list():
        if val is None:
            continue
        all_n = df.filter(pl.col(key) == val).height
        top_n = df.filter((pl.col(key) == val) & pl.Series("_t", top_mask)).height
        if all_n >= 5 and top_n >= 3:
            out.append((str(val), all_n, top_n, all_n / tot, (top_n / all_n)))
    out.sort(key=lambda t: -t[4])
    return out[:12]


def main():
    dep = load_dep().with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched")
    )
    # movement-only descriptor + target
    cols = ["MVT_ID_mvt", "y", "airport", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "SCHED_TIME_UTC_mvt",
            "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
            "hour", "month", "mvt_sched", "unmatched"]
    dep = dep.select([c for c in cols if c in dep.columns] + [pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow")])

    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        val = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        # E20 pred for unmatched = MVT-SCHED (LIRF override); use cached OOF for exactness
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        val = val.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E", "unmatched"]).rename({"unmatched": "_um"}),
                       on="MVT_ID_mvt", how="left")
        val = val.with_columns((0.456 * pl.col("pred_C") + 0.053 * pl.col("pred_D") + 0.491 * pl.col("pred_E")).alias("e20"))
        u = val.filter(pl.col("_um"))
        lirf = u.filter(pl.col("airport") == "LIRF")
        nlu = u.filter(pl.col("airport") != "LIRF")
        log(f"== {split} == unmatched {u.height} LIRF {lirf.height} nonLIRF {nlu.height}")

        # target vs clocks
        y = lirf["y"].to_numpy().astype(np.float64)
        ms = lirf["mvt_sched"].to_numpy().astype(np.float64)
        e20 = lirf["e20"].to_numpy().astype(np.float64)
        resid = y - e20
        order = np.argsort(-(resid ** 2))
        top = {k: {"n": int(np.sum(np.abs(resid[order[:k]]) > 1800)) if False else k,
                   "sse_share": float(np.sum(resid[order[:k]] ** 2) / np.sum(resid ** 2)),
                   "median_y": float(np.median(y[order[:k]])),
                   "median_mvt_sched": float(np.median(ms[order[:k]])),
                   "median_resid": float(np.median(resid[order[:k]]))} for k in (10, 25, 50)}
        log(f"  LIRF y: med {np.median(y):.0f} p90 {np.quantile(y,.9):.0f} max {y.max():.0f}; "
            f"mvt_sched med {np.median(ms):.0f} p90 {np.quantile(ms,.9):.0f}; resid(y-e20) med {np.median(resid):.0f}")
        log(f"  corr(y,mvt_sched) {np.corrcoef(y,ms)[0,1]:.3f}  corr(|resid|,mvt_sched) {np.corrcoef(np.abs(resid),ms)[0,1]:.3f}")
        log(f"  LIRF top-SSE: {top}")
        # how many LIRF rows have small residual (MVT-SCHED good)?
        small = np.mean(np.abs(resid) < 900)
        log(f"  frac |resid|<900s (MVT-SCHED good): {100*small:.1f}%   frac >30m: {100*np.mean(np.abs(resid)>1800):.1f}%")

        # feature enrichment among top-50 LIRF SSE
        tmask = np.zeros(lirf.height, dtype=bool)
        tmask[order[:50]] = True
        lf = lirf.with_columns(pl.Series("_top", tmask))
        enf = {}
        for k in ["RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt", "hour", "FLIGHT_mvt"]:
            enf[k] = enrich(lf, k, tmask, k)
        payload[split] = {"lirf_n": int(lirf.height), "lirf_y_median": float(np.median(y)),
                          "lirf_mvt_sched_median": float(np.median(ms)),
                          "lirf_resid_median": float(np.median(resid)),
                          "corr_y_ms": float(np.corrcoef(y, ms)[0, 1]),
                          "top_sse": top, "frac_resid_lt_900": float(small),
                          "enrich_top50": enf}

        # service recurrence for LIRF unmatched: matched train history of same (airport,FLIGHT,ADES)
        tr = dep.filter(~pl.col("unmatched"))
        prof = tr.filter(pl.col("airport") == "LIRF").group_by(["FLIGHT_mvt", "ADES_mvt"]).agg(
            pl.col("y").median().alias("hmed"), pl.col("y").std().alias("hsd"), pl.col("y").len().alias("hn"))
        j = lirf.join(prof, on=["FLIGHT_mvt", "ADES_mvt"], how="left")
        hn = j["hn"].to_numpy(); hsd = j["hsd"].to_numpy(); hmed = j["hmed"].to_numpy()
        payload[split]["service_support"] = {"coverage": float(np.mean(np.isfinite(hn))),
                                             "median_n": float(np.nanmedian(hn)),
                                             "median_hist_std": float(np.nanmedian(hsd)),
                                             "median_hist_med": float(np.nanmedian(hmed))}
        log(f"  LIRF service history coverage {100*np.mean(np.isfinite(hn)):.1f}% median_n {np.nanmedian(hn):.0f} "
            f"median_hist_std {np.nanmedian(hsd):.0f}")

        # non-LIRF top SSE characteristics
        ye = nlu["y"].to_numpy().astype(np.float64); ee = nlu["e20"].to_numpy().astype(np.float64)
        re = ye - ee; oe = np.argsort(-(re ** 2))
        payload[split]["nlu"] = {"n": int(nlu.height), "rmse": float(rmse(ye, ee)),
                                 "top100_sse_share": float(np.sum(re[oe[:100]] ** 2) / np.sum(re ** 2)),
                                 "top10": {"median_y": float(np.median(ye[oe[:10]])), "median_resid": float(np.median(re[oe[:10]]))},
                                 "frac_y_gt30": float(np.mean(ye > 1800))}

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (RES / "E29_lirf_forensics.json").write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")
    log("WROTE " + str(RES / "E29_lirf_forensics.json"))


if __name__ == "__main__":
    main()
