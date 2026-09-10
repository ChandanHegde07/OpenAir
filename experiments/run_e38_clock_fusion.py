"""E38 — latent off-block clock fusion.

STEP 1: do pairwise clock disagreements differ in the extreme taxi regime?
STEP 15: do all off-block clocks (AOBT/LOBT/IOBT/EOBT) add matched-model value
over E20's existing mvt_aobt/aobt_eobt features, especially top-5% SSE?
E20/v9 kept as baseline; matched expert A used for the ablation.
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
from common import add_causal_rolling, airport_mean_fallback, fill_with_fallback, load_dep, rmse  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402

RES = HERE / "results" / "E38"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CLOCKS = ["T_aobt", "T_lobt", "T_iobt", "T_eobt", "c_ae", "c_ai", "c_al", "c_le", "c_li", "c_ie", "clk_range"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def add_clocks(dep: pl.DataFrame) -> pl.DataFrame:
    s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)
    return dep.with_columns([
        s("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("T_aobt"),
        s("MVT_TIME_UTC_mvt", "LOBT_flt").alias("T_lobt"),
        s("MVT_TIME_UTC_mvt", "IOBT_flt").alias("T_iobt"),
        s("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("T_eobt"),
        s("AOBT_3_flt", "EOBT_1_flt").alias("c_ae"),
        s("AOBT_3_flt", "IOBT_flt").alias("c_ai"),
        s("AOBT_3_flt", "LOBT_flt").alias("c_al"),
        s("LOBT_flt", "EOBT_1_flt").alias("c_le"),
        s("LOBT_flt", "IOBT_flt").alias("c_li"),
        s("IOBT_flt", "EOBT_1_flt").alias("c_ie"),
    ]).with_columns(
        (pl.max_horizontal("c_ae", "c_ai", "c_al") - pl.min_horizontal("c_ae", "c_ai", "c_al")).alias("clk_range")
    )


def pdf(df, cols):
    p = df.select(cols + CAT_COLS).to_pandas()
    for c in CAT_COLS:
        p[c] = p[c].astype("category")
    return p[cols + CAT_COLS]


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr); dep = add_queue(dep)
    dep = add_push_disruption(dep); dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    dep = add_clocks(dep)
    log(f"rows {dep.height:,}")
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, va))
        p_ap_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal_tr = fill_with_fallback(fill_with_fallback(p_ap_t, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
        um = tr["unmatched"].to_numpy().astype(bool); ap = tr["airport"].to_numpy()
        ov = um & (ap == "LIRF")
        tr_m = tr.with_columns(pl.Series("p_cal", p_cal_tr)).filter(~pl.Series(ov))
        tr_fit, tr_es = time_es_split(tr_m)
        num_base = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
        yv = va["y"].to_numpy().astype(float); umv = va["unmatched"].to_numpy().astype(bool)
        mm = ~umv & np.isfinite(yv)
        def top_sse(p, frac):
            sse = (yv[mm] - p[mm]) ** 2
            k = max(1, int(frac * mm.sum()))
            return float(np.sort(sse)[::-1][:k].sum())
        # STEP 1: disagreement in extreme T
        Thr = np.quantile(yv[mm], 0.95)
        ext = mm & (yv >= Thr)
        norm = mm & (yv < np.quantile(yv[mm], 0.90))
        step1 = {}
        for c in ["clk_range", "c_ae", "c_ai", "c_al"]:
            x = np.abs(va[c].to_numpy().astype(float))
            step1[c] = {"extreme_median": float(np.nanmedian(x[ext])), "normal_median": float(np.nanmedian(x[norm]))}
        # STEP 15 ablation
        res = {}
        for tag, cols in [("base", num_base), ("base_clocks", num_base + CLOCKS)]:
            m = fit_lgb(pdf(tr_fit, cols), tr_fit["y"].to_numpy().astype(float) - tr_fit["p_cal"].to_numpy(),
                        pdf(tr_es, cols), tr_es["y"].to_numpy().astype(float) - tr_es["p_cal"].to_numpy(), seed=SEED)
            pred = p_cal + np.asarray(m.predict(pdf(va, cols)), float)
            res[tag] = {"matched_rmse": float(rmse(yv[mm], pred[mm])), "top5_sse": top_sse(pred, 0.05), "top1_sse": top_sse(pred, 0.01)}
        payload[split] = {"step1_disagreement": step1, "ablation": res}
        log(f"  [{split}] base {res['base']['matched_rmse']:.2f} top5 {res['base']['top5_sse']:.3e} | +clocks {res['base_clocks']['matched_rmse']:.2f} top5 {res['base_clocks']['top5_sse']:.3e}")
        log(f"      step1 extreme vs normal clk_range {step1['clk_range']}")
    (RES / "E38_clock.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E38_clock.json"))


if __name__ == "__main__":
    main()
