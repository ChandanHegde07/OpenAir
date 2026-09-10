"""E39 — off-block delta (Delta = AOBT - BLOCK). Baselines B0-B4.

T = P + Delta, P = MVT - AOBT. Model Delta and test T_hat = P + Delta_hat on
matched rows, both splits, cross-month. Compare with E20/v9.
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
from common import add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import MODEL_DISRUPT_COLS, add_arrival_delay_state, add_push_disruption, add_rolling_quantiles, load_arr_delay  # noqa: E402
from run_e12_e9 import CAT_COLS, NUM_COLS, fit_lgb, time_es_split  # noqa: E402
from run_e18_unmatched_specialist import prepare_split  # noqa: E402

RES = HERE / "results" / "E39"
RES.mkdir(parents=True, exist_ok=True)
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
    dep = add_push_disruption(dep); dep = add_rolling_quantiles(dep); dep = add_arrival_delay_state(dep, load_arr_delay())
    dep = dep.with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("P"),
        (pl.col("AOBT_3_flt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("Delta"),
    )
    num_b = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS and c != "geo_mean"]
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr = pack["tr"].filter(~pl.col("unmatched"))
        va = pack["va"].filter(~pl.col("unmatched"))
        tr = tr.with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("P"),
            (pl.col("AOBT_3_flt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("Delta"))
        va = va.with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("P"))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        y = va["y"].to_numpy().astype(float); P = va["P"].to_numpy().astype(float)
        # Delta forensics
        Dt = tr["Delta"].to_numpy().astype(float)
        out = {"e20_matched": float(rmse(y, e20)), "n": int(len(y)),
               "delta_mean": float(np.nanmean(Dt)), "delta_sd": float(np.nanstd(Dt)),
               "delta_p90": float(np.nanquantile(np.abs(Dt), 0.9)), "P_rmse": float(rmse(y, P))}
        # B0/B1/B2
        g_mean = float(np.nanmean(Dt))
        g_ap = tr.group_by("airport").agg(pl.col("Delta").mean().alias("m_ap"))
        vap = va.join(g_ap, on="airport", how="left")
        out["B0_P"] = float(rmse(y, P))
        out["B1_P_globalmean"] = float(rmse(y, P + g_mean))
        out["B2_P_airportmean"] = float(rmse(y, P + np.nan_to_num(vap["m_ap"].to_numpy())))
        # B4 LGB Delta
        tr_m = tr.with_columns(pl.Series("_d", tr["Delta"].to_numpy()))
        tr_fit, tr_es = time_es_split(tr_m)
        m = fit_lgb(pdf(tr_fit, num_b), tr_fit["Delta"].to_numpy().astype(float),
                    pdf(tr_es, num_b), tr_es["Delta"].to_numpy().astype(float), seed=SEED)
        dhat = np.asarray(m.predict(pdf(va, num_b)), float)
        out["B4_P_lgbdelta"] = float(rmse(y, P + dhat))
        # E38 clock-proxy expert (num_b + P duplicates + diffs) for reference
        va = va.with_columns(pl.Series("T_lobt", (va["MVT_TIME_UTC_mvt"] - va["LOBT_flt"]).dt.total_seconds().cast(pl.Float64)))
        tr_m = tr_m.with_columns(pl.Series("T_lobt", (tr_m["MVT_TIME_UTC_mvt"] - tr_m["LOBT_flt"]).dt.total_seconds().cast(pl.Float64)))
        # target T for clock expert uses p_cal-free direct residual is complex; skip
        payload[split] = out
        log(f"  [{split}] E20 {out['e20_matched']:.2f} | P {out['B0_P']:.2f} | P+meanDelta {out['B1_P_globalmean']:.2f} | "
            f"P+apDelta {out['B2_P_airportmean']:.2f} | P+LGBdelta {out['B4_P_lgbdelta']:.2f} | Delta mean {out['delta_mean']:.0f} sd {out['delta_sd']:.0f}")
    (RES / "E39_delta.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E39_delta.json"))


if __name__ == "__main__":
    main()
