"""E31 quick test — do causal service-identity features improve the matched expert?

Service tables are fit on the training split only (never val/ranking targets),
attached by hierarchical keys with shrinkage, then added to the E20 feature
matrix. Compare expert-A (LGB residual) matched RMSE with vs without service
features on Jan+Jul and Dec. E20 untouched.
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
from common import add_causal_rolling, load_dep, mae, rmse  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e18_unmatched_specialist import prepare_split, fit_residual  # noqa: E402

RES = HERE / "results" / "E31"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def service_features(tr: pl.DataFrame, va: pl.DataFrame):
    """Causal (train-only) service statistics + hierarchical shrinkage, joined to va."""
    tr = tr.with_columns(pl.col("FLIGHT_mvt").fill_null("NA").alias("_svc"))
    # level 0: airport x service x ADES ; level 1: airport x ADES ; level 2: airport
    l0 = tr.group_by(["airport", "_svc", "ADES_mvt"]).agg([
        pl.col("y").median().alias("s0_med"), pl.col("y").mean().alias("s0_mean"),
        pl.col("y").std().alias("s0_std"), pl.col("y").len().alias("s0_n")])
    l1 = tr.group_by(["airport", "ADES_mvt"]).agg([
        pl.col("y").median().alias("s1_med"), pl.col("y").len().alias("s1_n")])
    l2 = tr.group_by("airport").agg([pl.col("y").median().alias("s2_med"), pl.col("y").std().alias("s2_std")])
    l0h = tr.group_by(["airport", "_svc", "hour"]).agg([pl.col("y").median().alias("s0h_med"), pl.col("y").len().alias("s0h_n")])
    v = (va.with_columns(pl.col("FLIGHT_mvt").fill_null("NA").alias("_svc"))
           .join(l0, on=["airport", "_svc", "ADES_mvt"], how="left")
           .join(l1, on=["airport", "ADES_mvt"], how="left")
           .join(l2, on="airport", how="left")
           .join(l0h, on=["airport", "_svc", "hour"], how="left"))
    m = 20.0
    s0n = v["s0_n"].fill_null(0).to_numpy().astype(float)
    p_med = np.where(v["s1_n"].fill_null(0).to_numpy() >= 5, v["s1_med"].to_numpy(), v["s2_med"].to_numpy())
    svc_med = np.where(s0n > 0, (s0n * v["s0_med"].to_numpy() + m * p_med) / (s0n + m), p_med)
    s0hn = v["s0h_n"].fill_null(0).to_numpy().astype(float)
    svc_hr = np.where(s0hn > 0, (s0hn * v["s0h_med"].to_numpy() + m * svc_med) / (s0hn + m), svc_med)
    out = v.with_columns([
        pl.Series("svc_med", svc_med), pl.Series("svc_hr", svc_hr),
        pl.col("s0_n").fill_null(0).cast(pl.Float64).alias("svc_n"),
        pl.col("s0_mean").fill_null(pl.col("s1_med")).alias("svc_mean"),
        pl.col("s0_std").fill_null(pl.col("s2_std")).cast(pl.Float64).alias("svc_std"),
        (pl.col("y") - pl.Series("svc_med", svc_med)).alias("svc_resid_hist"),
    ])
    return out, ["svc_med", "svc_hr", "svc_n", "svc_mean", "svc_std"]


def main():
    log("E31 service expert test: build features...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    log(f"rows {dep.height:,}")

    out = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        tr_s, svc_cols = service_features(tr, tr)
        va_s, _ = service_features(tr, va)  # train-only tables applied to va
        # residual target vs p_cal
        from run_e2b_e3 import per_airport_ols
        from common import fill_with_fallback, airport_mean_fallback
        p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, va))
        p_ap_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal_tr = fill_with_fallback(fill_with_fallback(p_ap_t, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
        # hygiene: drop LIRF override rows
        um = tr["unmatched"].to_numpy().astype(bool); ap = tr["airport"].to_numpy()
        ov = um & (ap == "LIRF")
        tr_m = tr_s.with_columns(pl.Series("p_cal", p_cal_tr)).filter(~pl.Series(ov))
        tr_fit, tr_es = time_es_split(tr_m)

        def xy(df, cols):
            pdf = df.select(cols + CAT_COLS).to_pandas()
            for c in CAT_COLS:
                pdf[c] = pdf[c].astype("category")
            return pdf[cols + CAT_COLS]

        num_base = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
        feats_base = num_base
        feats_svc = num_base + svc_cols
        for tag, feats in [("base", feats_base), ("svc", feats_svc)]:
            m = fit_lgb(xy(tr_fit, feats), tr_fit["y"].to_numpy().astype(float) - tr_fit["p_cal"].to_numpy(),
                        xy(tr_es, feats), tr_es["y"].to_numpy().astype(float) - tr_es["p_cal"].to_numpy(), seed=SEED)
            pred = p_cal + np.asarray(m.predict(xy(va_s, feats)), dtype=float)
            yva = va_s["y"].to_numpy().astype(float); umv = va_s["unmatched"].to_numpy().astype(bool)
            mask = ~umv & np.isfinite(yva)
            out.setdefault(split, {})[tag] = {"matched_rmse": rmse(yva[mask], pred[mask]), "n": int(mask.sum())}
            log(f"  [{split}] expert A {tag}: matched RMSE {out[split][tag]['matched_rmse']:.2f}")
    (RES / "E31_service_expert.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("WROTE " + str(RES / "E31_service_expert.json"))


if __name__ == "__main__":
    main()
