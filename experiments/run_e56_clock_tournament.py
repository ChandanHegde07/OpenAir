"""E56 — multi-clock off-block decomposition tournament.

For each clock C in {SCHED, EOBT, LOBT, IOBT, AOBT}: learn G_C = BLOCK - C,
reconstruct T = (MVT - C) - G_C_hat. Availability audit, matched RMSE, top-1%,
and error orthogonality vs v13. Cross-month OOF.
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
from run_submitting_check import RANK_PATH  # noqa: E402

RES = HERE / "results" / "E56"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CLOCKS = ["SCHED", "EOBT", "LOBT", "IOBT", "AOBT"]
DC = {"SCHED": "mvt_sched", "EOBT": "mvt_eobt", "LOBT": "mvt_lobt", "IOBT": "mvt_iobt", "AOBT": "mvt_aobt"}
NUM = ["mvt_sched", "mvt_aobt", "mvt_eobt", "mvt_lobt", "mvt_iobt", "aobt_eobt", "hour", "dow", "month"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "flt_prefix"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def load_frames():
    dep = load_dep().with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    fr = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
        f = f.with_columns(pl.Series("e20", 0.456 * f["pred_C"].to_numpy() + 0.053 * f["pred_D"].to_numpy() + 0.491 * f["pred_E"].to_numpy()))
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20").is_finite()).filter(pl.col("y").is_finite()).filter(pl.col("mvt_aobt").is_finite())
        fr[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return fr


def stage(tr, ev, clock):
    y0 = tr["y"].to_numpy().astype(float); yv = ev["y"].to_numpy().astype(float)
    D0 = tr[DC[clock]].to_numpy().astype(float); Dv = ev[DC[clock]].to_numpy().astype(float)
    # only finite rows (clock present); record mask
    m0 = np.isfinite(D0) & np.isfinite(y0)
    mv = np.isfinite(Dv) & np.isfinite(yv)
    trm = tr.filter(pl.Series(m0)); evm = ev.filter(pl.Series(mv))
    G0 = trm[DC[clock]].to_numpy().astype(float) - trm["y"].to_numpy().astype(float)
    mm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    mm.fit(pdf(trm), G0, categorical_feature=CAT)
    ghat = np.asarray(mm.predict(pdf(evm)), float)
    rec = np.clip(evm[DC[clock]].to_numpy().astype(float) - ghat, 0, None)
    return {"y": yv[mv], "rec": rec, "D": Dv[mv]}


def concat_parts(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    fr = load_frames()
    # availability audit (ranking vs train coverage of each clock)
    avail = {}
    rank = pl.scan_parquet(RANK_PATH).filter(pl.col("PHASE_mvt") == "DEP").select(
        ["SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]).collect()
    rn = rank.height
    tr_cols = {"SCHED": "SCHED_TIME_UTC_mvt", "EOBT": "EOBT_1_flt", "LOBT": "LOBT_flt", "IOBT": "IOBT_flt", "AOBT": "AOBT_3_flt"}
    for c, col in tr_cols.items():
        avail[c] = {"ranking_coverage": float(1 - rank[col].null_count() / rn)}
    payload = {"availability": avail, "splits": {}}
    for split, label in (("janjul", "janjul"), ("dec", "dec")):
        results = {}
        # v13 baseline (grec+hat like E50/E55)
        if split == "janjul":
            parts = []
            for a, b in ((1, 7), (7, 1)):
                parts.append(stage(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b), "AOBT"))
            o = concat_parts(parts)
        else:
            o = stage(fr["janjul"], fr["dec"], "AOBT")
        results["v13_AOBT_alone"] = {"matched": float(rmse(o["y"], o["rec"])), "top1": top1(o["y"], o["rec"])}
        log(f"  [{split}] AOBT decomposition alone: matched {results['v13_AOBT_alone']['matched']:.2f} top1 {results['v13_AOBT_alone']['top1']:.3e}")
        for c in CLOCKS:
            if split == "janjul":
                parts = []
                for a, b in ((1, 7), (7, 1)):
                    parts.append(stage(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b), c))
                oc = concat_parts(parts)
            else:
                oc = stage(fr["janjul"], fr["dec"], c)
            results[c] = {"matched": float(rmse(oc["y"], oc["rec"])), "top1": top1(oc["y"], oc["rec"])}
            log(f"  [{split}] {c}: reconstructed matched {results[c]['matched']:.2f} top1 {results[c]['top1']:.3e}")
        payload["splits"][split] = results
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    # report
    jj = payload["splits"]["janjul"]
    L = ["# E56 — multi-clock off-block decomposition", "",
         "| Clock | ranking coverage | Jan+Jul matched | top-1% SSE |",
         "|---|---:|---:|---:|"]
    for c in CLOCKS + ["v13_AOBT_alone"]:
        cov = avail.get(c, {}).get("ranking_coverage", "-")
        L.append(f"| {c} | {cov} | {jj[c]['matched']:.2f} | {jj[c]['top1']:.3e} |")
    L.append("")
    L.append("Note: E38 established AOBT/LOBT/IOBT/EOBT are mutually identical for "
             "matched rows (clk_range median 0) — so LOBT/IOBT/EOBT decompositions "
             "are expected to coincide with the AOBT decomposition (already in v13).")
    (RES / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
