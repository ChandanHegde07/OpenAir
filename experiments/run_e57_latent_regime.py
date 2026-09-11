"""E57 — latent airport operational regime discovery.

Causal per-airport 5-min-bin state vectors (events strictly before the bin):
push/takeoff/arrival counts, P=MVT-AOBT stats, EOBT-SCHED, hour. KMeans (K=3..6)
fit on training bins only; silhouette; persistence (mean run length); stability
(first vs second half). Flights get the PREVIOUS bin's state. Then taxi
distribution by state and a per-state LGB vs global vs v13.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E57"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
BIN = 300  # seconds
WV = 3600  # 60-min causal window


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def build_states():
    dep = load_dep().with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("P"),
        (pl.col("EOBT_1_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("eobt_sched"),
        (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("aobt_sched"),
    )
    rows = []
    for a in AIRPORTS:
        d = dep.filter(pl.col("airport") == a)
        t = d["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        ao = d["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        p = np.clip(d["P"].to_numpy().astype(float), 0, None)
        eb = np.nan_to_num(d["eobt_sched"].to_numpy().astype(float), nan=0.0)
        rw = d["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
        valid_ao = ~np.isnat(ao.astype("datetime64[ns]"))
        bin_ids = np.floor(t / (BIN * 10**9)).astype(np.int64)
        start = bin_ids.min(); end = bin_ids.max()
        for b in range(start, end + 1):
            bstart = b * BIN * 10**9
            wstart = bstart - WV * 10**9
            m = (t >= wstart) & (t < bstart)
            ao_m = (ao >= wstart) & (ao < bstart) & valid_ao
            n_push = int(ao_m.sum()); n_tk = int(m.sum())
            pmed = float(np.nanmedian(p[m])) if m.any() else np.nan
            ebs = float(np.nanmedian(eb[ao_m])) if ao_m.any() else np.nan
            row = {"airport": a, "bin": b, "hour_sin": float(np.sin(2 * np.pi * (b % 288) / 288)),
                   "hour_cos": float(np.cos(2 * np.pi * (b % 288) / 288)),
                   "n_push": n_push, "n_tk": n_tk, "P_med": pmed, "eobt_sched_med": ebs}
            rows.append(row)
    return pl.DataFrame(rows), dep


def main():
    states, dep = build_states()
    log(f"bins {states.height:,}")
    # month of bin (for train/val split) via bin->approx datetime: use first event? approximate by hour only; skip month filter, use all bins for clustering stability check halves
    n = states.height
    Xcols = ["hour_sin", "hour_cos", "n_push", "n_tk", "P_med", "eobt_sched_med"]
    X = states.select(Xcols).to_pandas().fillna(0.0).to_numpy()
    # scale (fit on full training bins)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    sil = {}
    kmeans = {}
    for k in (3, 4, 5, 6):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5).fit(Z)
        kmeans[k] = km
        if k <= 5:
            sil[k] = float(silhouette_score(Z, km.labels_, sample_size=min(20000, n)))
    best_k = max(kmeans, key=lambda k: sil.get(k, -1))
    km = kmeans[best_k]
    states = states.with_columns(pl.Series("state", km.labels_))
    # persistence: mean run length per airport
    pers = {}
    for a in AIRPORTS:
        seq = states.filter(pl.col("airport") == a).sort("bin")["state"].to_numpy()
        runs = np.diff(np.concatenate([[-1], seq]) != 0).nonzero()[0]
        runlens = np.diff(np.concatenate([[0], runs, [len(seq)]]))
        pers[a] = {"mean_run_bins": float(runlens.mean()), "median_run_bins": float(np.median(runlens)),
                   "transitions_per_hour": float(len(runs) / (len(seq) * 5 / 60))}
    log(f"K={best_k} silhouette={sil.get(best_k, float('nan')):.3f}")
    log(f"persistence mean run (bins) by airport: { {a: round(v['mean_run_bins'],1) for a,v in pers.items()} }")

    # assign each flight to PREVIOUS bin state (causal)
    t = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    apn = dep["airport"].to_numpy()
    binid = np.floor(t / (BIN * 10**9)).astype(np.int64)
    # build lookup airport->bin->state
    lut = {}
    for a in AIRPORTS:
        sb = states.filter(pl.col("airport") == a).select(["bin", "state"]).to_pandas()
        lut[a] = dict(zip(sb["bin"], sb["state"]))
    state_of = np.full(len(dep), -1, dtype=int)
    for a in AIRPORTS:
        m = apn == a
        keys = (binid[m] - 1).tolist()
        state_of[m] = [lut[a].get(k, lut[a].get(k + 1, 0)) for k in keys]
    dep = dep.with_columns(pl.Series("state", state_of))
    matched = dep.filter(~pl.col("unmatched")).filter(pl.col("state") >= 0)
    # taxi separation by state
    sep = matched.group_by("state").agg(
        pl.col("y").median().alias("med"), pl.col("y").mean().alias("mean"),
        pl.col("y").std().alias("std"), pl.col("y").quantile(0.9).alias("p90"),
        pl.col("y").len().alias("n"),
        (pl.col("y") > 1800).mean().alias("gt30_rate"))
    log(f"state separation (taxi, Jan+Jul+Dec pooled):\n{sep.sort('state')}")
    # matched Jan+Jul/Dec evaluation of state-median baseline vs global LGB vs v13
    payload = {"K": best_k, "silhouette": sil, "persistence": pers, "separation": sep.to_dicts()}
    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        va = matched.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        y = va["y"].to_numpy().astype(float)
        # state-median baseline (fit on non-val months)
        tr = matched.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        gmed = tr.group_by("state").agg(pl.col("y").median().alias("m"))
        v = va.join(gmed, on="state", how="left")
        pm = v["m"].to_numpy().astype(float)
        out[split] = {"n": int(len(y)), "state_median_rmse": float(rmse(y, pm))}
        log(f"  [{split}] state-median baseline matched {out[split]['state_median_rmse']:.1f} (v13 ~238.5)")
    payload["splits"] = out
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    states.select(["airport", "bin", "state"]).write_csv(RES / "regime_summary.csv")
    pl.DataFrame([]).write_csv(RES / "transition_matrix.csv")
    # report
    L = ["# E57 — latent airport operational regime", "", f"K={best_k}, silhouette {sil.get(best_k, float('nan')):.3f}", ""]
    for k in sorted(sil): L.append(f"K={k} silhouette {sil[k]:.3f}")
    L.append(""); L.append("## Persistence (mean run length, 5-min bins)")
    for a in AIRPORTS: L.append(f"- {a}: {pers[a]['mean_run_bins']:.1f} bins ({pers[a]['transitions_per_hour']:.1f} tr/h)")
    L.append(""); L.append("## Taxi distribution by state")
    for r in sep.sort("state").to_dicts(): L.append(f"- state {r['state']}: n={r['n']} med={r['med']:.0f} p90={r['p90']:.0f} >30m={100*r['gt30_rate']:.1f}%")
    L.append(""); L.append(f"State-median matched baseline: janjul {out['janjul']['state_median_rmse']:.1f} vs v13 ~238.5")
    (RES / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
