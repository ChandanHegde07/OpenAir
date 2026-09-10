"""E42 — flight/service identity residual priors (chronological).

residual = TAXITIME - base_pred. Identity priors (mean residual, empirical-Bayes
shrunk) are fitted on a train-internal temporal holdout (tr_es) and applied to
validation. alpha applied to E20 OOF on val. Candidates auto-selected on tr_es;
frozen evaluated on Jan+Jul and Dec.
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

RES = HERE / "results" / "E42"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
KEYS = [
    ["CALLSIGN"],
    ["CALLSIGN", "airport"],
    ["CALLSIGN", "airport", "ADES_mvt"],
    ["FLIGHT_mvt", "airport"],
    ["FLIGHT_mvt", "airport", "ADES_mvt"],
    ["CALLSIGN", "wd_bucket"],
    ["CALLSIGN", "wd_bucket", "hour_bucket"],
    ["op_prefix", "ADES_mvt"],
]
LAMBDAS = [1.0, 5.0, 20.0, 50.0, 200.0]
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df, cols):
    p = df.select(cols + CAT_COLS).to_pandas()
    for c in CAT_COLS:
        p[c] = p[c].astype("category")
    return p[cols + CAT_COLS]


def add_keys(df):
    return df.with_columns([
        pl.col("FLIGHT_mvt").fill_null("NA").cast(pl.String).alias("CALLSIGN"),
        pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA").cast(pl.String).alias("op_prefix"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("wd_bucket"),
        (pl.col("hour") // 2).cast(pl.Int64).alias("hour_bucket"),
    ])


def prior_table(fit, key, lam):
    g = (fit.group_by(key)
         .agg(pl.col("resid").mean().alias("m"), pl.col("resid").std().alias("s"),
              pl.col("resid").len().alias("n")))
    return g.with_columns((pl.col("n") / (pl.col("n") + lam) * pl.col("m")).alias("corr"))


def apply_prior(df, key, tab):
    return df.join(tab.select(key + ["corr"]), on=key, how="left")["corr"].fill_null(0.0).to_numpy()


def main():
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr(); dep = add_traffic(dep, arr); dep = add_queue(dep)
    dep = add_push_disruption(dep); dep = add_rolling_quantiles(dep); dep = add_arrival_delay_state(dep, load_arr_delay())
    dep = add_keys(dep)
    num_b = [c for c in NUM_COLS + MODEL_DISRUPT_COLS if c not in CAT_COLS]
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        pack = prepare_split(dep, months, matched_geo=True)
        tr, va = pack["tr"], pack["va"]
        # expert A pipeline (residual vs p_cal) to get honest OOF residual on tr_es
        p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, va))
        p_ap_t, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
        p_cal_tr = fill_with_fallback(fill_with_fallback(p_ap_t, tr["geo_mean"].to_numpy().astype(float)), airport_mean_fallback(tr, tr))
        um_tr = tr["unmatched"].to_numpy().astype(bool)
        trm = tr.filter(~pl.Series(um_tr)).with_columns(pl.Series("p_cal", p_cal_tr[~um_tr]))
        tr_fit, tr_es = time_es_split(trm)
        mA = fit_lgb(pdf(tr_fit, num_b), tr_fit["y"].to_numpy().astype(float) - tr_fit["p_cal"].to_numpy(),
                     pdf(tr_es, num_b), tr_es["y"].to_numpy().astype(float) - tr_es["p_cal"].to_numpy(), seed=SEED)
        tr_es = tr_es.with_columns(pl.Series("base", tr_es["p_cal"].to_numpy() + np.asarray(mA.predict(pdf(tr_es, num_b)), float)))
        tr_es = tr_es.with_columns((pl.col("y") - pl.col("base")).alias("resid"))
        # E20 OOF on val
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        yv = va["y"].to_numpy().astype(float); umv = va["unmatched"].to_numpy().astype(bool)
        m = ~umv & np.isfinite(yv) & np.isfinite(e20)
        base_e20 = float(rmse(yv[m], e20[m])); top5_e = float(np.sort((yv[m] - e20[m]) ** 2)[::-1][:int(0.05 * m.sum())].sum())
        # auto-select key+lambda+alpha on tr_es (resid there), then apply to val E20
        best = None
        for key in KEYS:
            for lam in LAMBDAS:
                tab = prior_table(tr_es, key, lam)
                c_es = apply_prior(tr_es, key, tab)
                ye = tr_es["y"].to_numpy().astype(float); be = tr_es["base"].to_numpy().astype(float)
                for al in ALPHAS:
                    s = float(np.sum((ye - (be + al * c_es)) ** 2))
                    if best is None or s < best[0]:
                        best = (s, key, lam, al)
        _, key, lam, al = best
        tab = prior_table(tr_es, key, lam)
        # coverage on va
        j = va.join(tab.select(key + ["corr", "n"]), on=key, how="left")
        c_va = j["corr"].fill_null(0.0).to_numpy().astype(float)
        cov = float(np.mean(j["n"].is_not_null().to_numpy()))
        pred = e20 + al * c_va
        sse = (yv[m] - pred[m]) ** 2
        out = {"key": key, "lambda": lam, "alpha": al, "coverage": cov,
               "e20_matched": base_e20, "e42_matched": float(rmse(yv[m], pred[m])),
               "e20_top5": top5_e, "e42_top5": float(np.sort(sse)[::-1][:int(0.05 * m.sum())].sum()),
               "n": int(m.sum())}
        payload[split] = out
        log(f"  [{split}] selected {key} lam={lam} al={al} cov={cov:.3f} | E20 {base_e20:.2f} top5 {top5_e:.3e} -> E42 {out['e42_matched']:.2f} top5 {out['e42_top5']:.3e}")
    (RES / "E42_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    # report
    jj = payload["janjul"]; dc = payload["dec"]
    lines = ["# E42 — flight identity residual prior", "",
             f"- selected: key={jj['key']} lambda={jj['lambda']} alpha={jj['alpha']} coverage={jj['coverage']:.2f}",
             "", "| Model | Jan+Jul Overall | Jan+Jul Matched | Top-5% SSE | Dec Matched |",
             "|---|---:|---:|---:|---:|",
             f"| E20 | 368.03 | {jj['e20_matched']:.2f} | {jj['e20_top5']:.3e} | {dc['e20_matched']:.2f} |",
             f"| Best E42 | - | {jj['e42_matched']:.2f} | {jj['e42_top5']:.3e} | {dc['e42_matched']:.2f} |", "",
             f"JAN+JUL GAIN: {jj['e20_matched']-jj['e42_matched']:+.2f}",
             f"TOP-5% SSE GAIN: {100*(jj['e20_top5']-jj['e42_top5'])/jj['e20_top5']:+.1f}%",
             f"DEC GAIN: {dc['e20_matched']-dc['e42_matched']:+.2f}", ""]
    gain = jj["e20_matched"] - jj["e42_matched"]
    tailgain = (jj["e20_top5"] - jj["e42_top5"]) / jj["e20_top5"]
    go = gain > 4 and tailgain > 0.02 and (dc["e20_matched"] - dc["e42_matched"]) > -1
    lines.append(f"GO / NO-GO: {'GO' if go else 'NO-GO'}")
    (RES / "E42_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"selected {jj['key']} lam={jj['lambda']} alpha={jj['alpha']}")
    log(f"JAN+JUL GAIN {gain:+.2f}  TOP5 {100*tailgain:+.1f}%  DEC {dc['e20_matched']-dc['e42_matched']:+.2f}  -> {'GO' if go else 'NO-GO'}")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
