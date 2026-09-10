"""E38b — clock-augmented ensemble vs E20/v9.

Add an off-block-clock expert (LGB residual with MVT-AOBT/LOBT/IOBT/EOBT +
pairwise differences) to the cached E20 experts (C/D/E OOF) and NNLS on the
validation set. Report matched/overall/top5 SSE; estimate LB.
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
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import add_causal_rolling, airport_mean_fallback, fill_with_fallback, load_dep, rmse  # noqa: E402
from run_e2b_e3 import per_airport_ols  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402
from run_e38_clock_fusion import CLOCKS, add_clocks  # noqa: E402

RES = HERE / "results" / "E38"
SEED = 1


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT_COLS).to_pandas()
    for c in CAT_COLS:
        p[c] = p[c].astype("category")
    return p[cols + CAT_COLS]


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr(); dep = add_traffic(dep, arr); dep = add_queue(dep)
    dep = add_push_disruption(dep); dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay()); dep = add_clocks(dep)
    num_b = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, va))
        p_ap_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal_tr = fill_with_fallback(fill_with_fallback(p_ap_t, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
        um = tr["unmatched"].to_numpy().astype(bool); ap = tr["airport"].to_numpy()
        tr_m = tr.with_columns(pl.Series("p_cal", p_cal_tr)).filter(~pl.Series(um & (ap == "LIRF")))
        tr_fit, tr_es = time_es_split(tr_m)
        cols = num_b + CLOCKS
        m = fit_lgb(pdf(tr_fit, cols), tr_fit["y"].to_numpy().astype(float) - tr_fit["p_cal"].to_numpy(),
                    pdf(tr_es, cols), tr_es["y"].to_numpy().astype(float) - tr_es["p_cal"].to_numpy(), seed=SEED)
        a_clk = fill_with_fallback(p_cal + np.asarray(m.predict(pdf(va, cols)), float), p_cal)
        # join cached E20 experts
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.with_columns(pl.Series("a_clk", a_clk))
        j = va.select(["MVT_ID_mvt", "y", "unmatched", "airport", "a_clk", "mvt_sched"]).join(
            oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        y = j["y"].to_numpy().astype(float); umj = j["unmatched"].to_numpy().astype(bool); apj = j["airport"].to_numpy()
        sched = j["mvt_sched"].to_numpy()
        C = j["pred_C"].to_numpy(); D = j["pred_D"].to_numpy(); E = j["pred_E"].to_numpy(); A = j["a_clk"].to_numpy()
        lirf = umj & (apj == "LIRF")
        ov = lambda p: np.where(lirf, sched, p)
        e20 = ov(0.456 * C + 0.053 * D + 0.491 * E)
        mm = ~umj & np.isfinite(y)
        def top(p, frac):
            sse = (y[mm] - p[mm]) ** 2; k = max(1, int(frac * mm.sum()))
            return float(np.sort(sse)[::-1][:k].sum())
        out = {"e20_matched": float(rmse(y[mm], e20[mm])), "e20_top5": top(e20, 0.05), "n_matched": int(mm.sum())}
        # NNLS over matched with A added
        M = np.column_stack([C[mm], D[mm], E[mm], A[mm]]); yy = y[mm]
        w, _ = nnls(M, yy); w = w / max(w.sum(), 1e-9)
        p_m = M @ w
        full = e20.copy(); full[mm] = p_m
        out["nnls_w_CDE_A"] = [float(x) for x in w]
        out["clk_matched"] = float(rmse(y[mm], p_m)); out["clk_top5"] = top(full, 0.05)
        out["clk_overall"] = float(rmse(y, full))
        # E20 overall for reference
        out["e20_overall"] = float(rmse(y, e20))
        # 2-component blend (E20 vs clock expert) grid
        grid = {}
        for wa in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
            pm = (1 - wa) * e20[mm] + wa * A[mm]
            full2 = e20.copy(); full2[mm] = pm
            grid[wa] = {"matched": float(rmse(y[mm], pm)), "overall": float(rmse(y, full2)), "top5": top(full2, 0.05)}
        out["blend2_grid"] = grid
        payload[split] = out
        log(f"  [{split}] E20 matched {out['e20_matched']:.2f} top5 {out['e20_top5']:.3e} | clk-NNLS matched {out['clk_matched']:.2f} top5 {out['clk_top5']:.3e} w={np.round(w,3).tolist()}")
        log(f"      overall E20 {out['e20_overall']:.2f} -> clk {out['clk_overall']:.2f}")
    (RES / "E38b_ensemble.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E38b_ensemble.json"))


if __name__ == "__main__":
    main()
