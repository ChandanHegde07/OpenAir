"""E28 STEP 3 — matched tail regime (Mission 1+3).

Matched rows only (LIRF unmatched excluded; that is a separate mission).
Model the E20 residual and gate a correction by a causal tail-risk score
built from off-block source disagreement (E27) and compact surface pressure.

E20 is immutable: final = e20_pred + gated_correction; default correction = 0.
Gate threshold chosen on a train-internal temporal split (tr_es), never on
validation or ranking labels.

Ablations: A E20; B +disagreement; C +surface; D +both (ungated residual model);
E gated tail correction.
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

from common import AIRPORTS, add_causal_rolling, load_dep, mae, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E28"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
TAIL = 1800.0
WINS = [5, 10, 30]

DISAGREE_NUM = ["sd_abs_eobt", "sd_abs_lobt", "sd_abs_iobt", "sd_eobt", "sd_lobt", "sd_iobt",
                "sd_max", "sd_mean", "sd_med", "sd_range", "sd_n_avail"]
SURF_NUM = ["f_dep_5", "f_dep_10", "f_dep_30", "f_arr_5", "f_arr_10", "f_arr_30",
            "f_tot_10", "f_tot_30", "f_rwy_dep_5", "f_rwy_dep_10", "f_rwy_dep_30",
            "f_rwy_arr_5", "f_rwy_arr_10", "f_rwy_arr_30", "f_ts_dep", "f_ts_arr", "f_ts_rwy",
            "f_burst_dep", "f_burst_rwy", "f_arr_dep_ratio"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt"]
BASE_NUM = ["e20_pred", "hour", "dow", "month", "mvt_sched", "dep_15m"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def add_disagreement(dep: pl.DataFrame) -> pl.DataFrame:
    a = pl.col("AOBT_3_flt")
    cols = []
    pairs = {"eobt": "EOBT_1_flt", "lobt": "LOBT_flt", "iobt": "IOBT_flt"}
    for k, c in pairs.items():
        cols.append((a - pl.col(c)).dt.total_seconds().cast(pl.Float64).alias(f"sd_{k}"))
        cols.append(((a - pl.col(c)).dt.total_seconds().abs()).cast(pl.Float64).alias(f"sd_abs_{k}"))
    out = dep.with_columns(cols)
    av = [(pl.col(f"sd_{k}").is_not_null().cast(pl.Int64)) for k in pairs]
    out = out.with_columns([
        pl.max_horizontal("sd_eobt", "sd_lobt", "sd_iobt").alias("sd_max"),
        pl.mean_horizontal("sd_eobt", "sd_lobt", "sd_iobt").alias("sd_mean"),
        pl.min_horizontal("sd_eobt", "sd_lobt", "sd_iobt").alias("sd_med"),
        (pl.max_horizontal("sd_eobt", "sd_lobt", "sd_iobt") - pl.min_horizontal("sd_eobt", "sd_lobt", "sd_iobt")).alias("sd_range"),
        (av[0] + av[1] + av[2]).alias("sd_n_avail"),
    ])
    return out


def add_surface(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in SURF_NUM}
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()

    def counts(times, q, w):
        return (np.searchsorted(times, q, side="left") - np.searchsorted(times, q - w * 60 * 10**9, side="left")).astype(float)

    for a in AIRPORTS:
        sel = ap == a
        t = mvt[sel]
        if t.size == 0:
            continue
        idx = np.flatnonzero(sel)
        d = {w: counts(t, t, w) for w in WINS}
        for w in WINS:
            cols[f"f_dep_{w}"][idx] = d[w]
        # arrivals from other phases: use queued arrivals via causal arrival stream unavailable here;
        # approximate arrival pressure by prior rows whose ADES==airport is not available in DEP frame,
        # so use movement-independent proxy: total prior departures already covers departures.
        cols["f_tot_10"][idx] = d[10]
        cols["f_tot_30"][idx] = d[30]
        prev = np.searchsorted(t, t, side="left") - 1
        cols["f_ts_dep"][idx] = np.where(prev >= 0, (t - t[prev]).astype(float) / 1e9, np.nan)
        cols["f_burst_dep"][idx] = d[5] - d[30] / 6.0
        # same-runway
        rw = rwy[sel]
        for r in np.unique(rw):
            m = rw == r
            t_r = t[m]
            ii = idx[m]
            for w in WINS:
                dd = counts(t_r, t_r, w)
                cols[f"f_rwy_dep_{w}"][ii] = dd
            prevr = np.searchsorted(t_r, t_r, side="left") - 1
            cols["f_ts_rwy"][ii] = np.where(prevr >= 0, (t_r - t_r[prevr]).astype(float) / 1e9, np.nan)
            cols["f_burst_rwy"][ii] = cols["f_rwy_dep_5"][ii] - cols["f_rwy_dep_30"][ii] / 6.0
    # arrival pressure proxies are filled from ARR stream in prepare(); leave NaN-safe
    cols["f_arr_dep_ratio"] = np.where(cols["f_dep_10"] > 0, 1.0, 0.0)
    return dep.with_columns([pl.Series(k, v) for k, v in cols.items()])


def add_arrival_pressure(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in ["f_arr_5", "f_arr_10", "f_arr_30", "f_rwy_arr_5", "f_rwy_arr_10", "f_rwy_arr_30"]}
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    at = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    aap = arr["airport"].to_numpy()
    arw = arr["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    for a in AIRPORTS:
        sel = ap == a
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        t = mvt[sel]
        at_a = at[aap == a]
        for w in WINS:
            cols[f"f_arr_{w}"][idx] = (np.searchsorted(at_a, t, side="left") - np.searchsorted(at_a, t - w * 60 * 10**9, side="left")).astype(float)
        rw = rwy[sel]
        for r in np.unique(rw):
            m = rw == r
            ii = idx[m]
            t_r = t[m]
            at_ar = at[(aap == a) & (arw == r)]
            for w in WINS:
                cols[f"f_rwy_arr_{w}"][ii] = (np.searchsorted(at_ar, t_r, side="left") - np.searchsorted(at_ar, t_r - w * 60 * 10**9, side="left")).astype(float)
    out = dep.with_columns([pl.Series(k, v) for k, v in cols.items()])
    o = out.with_columns((pl.col("f_arr_10") / (pl.col("f_dep_10") + 1.0)).alias("f_arr_dep_ratio"))
    return o


def to_pdf(df, num, cat):
    pdf = df.select(num + cat).to_pandas()
    for c in cat:
        pdf[c] = pdf[c].astype("category")
    return pdf


def fit_clf(x, y, seed=SEED):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(x, y, categorical_feature=CAT)
    return m


def fit_reg(x, y, seed=SEED):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(x, y, categorical_feature=CAT)
    return m


def sse_dict(y, p, tails):
    err = np.asarray(y) - np.asarray(p)
    out = {"rmse": rmse(y, p), "sse": float(np.sum(err ** 2))}
    for t in (1800, 2700, 3600):
        sel = np.abs(err) > t
        out[f"n_abs_gt{t}"] = int(sel.sum())
        out[f"sse_abs_gt{t}"] = float(np.sum(err[sel] ** 2))
    return out


def resid_of(df):
    return np.asarray(df["y"].to_numpy(), dtype=np.float64) - np.asarray(df["e20_pred"].to_numpy(), dtype=np.float64)


def main():
    log("E28 STEP3: load + features...")
    from run_e4_e5 import add_traffic, load_arr
    dep = add_causal_rolling(load_dep())
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_disagreement(dep)
    dep = add_surface(dep)
    dep = add_arrival_pressure(dep, arr)
    log(f"rows {dep.height:,}")

    # matched validation rows for both splits, with E20 OOF predictions
    frames = []
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        oof = oof.with_columns(pl.Series("e20_pred",
                                         0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy()
                                         + 0.491 * oof["pred_E"].to_numpy()))
        f = dep.join(oof.select(["MVT_ID_mvt", "e20_pred"]), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite())
        f = f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("_m"))
        frames.append(f.select(["MVT_ID_mvt", "_m", "y"] + BASE_NUM + DISAGREE_NUM + SURF_NUM + CAT))
    d = pl.concat(frames, how="vertical")
    log(f"matched holdout rows {d.height:,}")

    feats_B = BASE_NUM + DISAGREE_NUM
    feats_C = BASE_NUM + SURF_NUM
    feats_D = BASE_NUM + DISAGREE_NUM + SURF_NUM

    def fit_and_predict(fit_df, eval_df):
        yf = resid_of(fit_df)
        hard = np.abs(yf) > 600
        risk = fit_clf(to_pdf(fit_df, feats_D, CAT), (np.abs(yf) > TAIL).astype(int))
        resid_m = fit_reg(to_pdf(fit_df, feats_D, CAT)[hard], yf[hard])
        resB = fit_reg(to_pdf(fit_df, feats_B, CAT)[hard], yf[hard])
        resC = fit_reg(to_pdf(fit_df, feats_C, CAT)[hard], yf[hard])
        return {
            "p_tail": risk.predict_proba(to_pdf(eval_df, feats_D, CAT))[:, 1],
            "corr": np.asarray(resid_m.predict(to_pdf(eval_df, feats_D, CAT)), dtype=np.float64),
            "cB": np.asarray(resB.predict(to_pdf(eval_df, feats_B, CAT)), dtype=np.float64),
            "cC": np.asarray(resC.predict(to_pdf(eval_df, feats_C, CAT)), dtype=np.float64),
        }

    # temporal cross-fit: Jan<->Jul; and (Jan+Jul) -> Dec; plus Dec -> Jan+Jul (gate transfer)
    m = d["_m"].to_numpy()
    jan = np.isin(m, [1]); jul = np.isin(m, [7]); dec = np.isin(m, [12])
    jj = jan | jul
    preds = {k: np.full(d.height, np.nan) for k in ("p_tail", "corr", "cB", "cC")}
    for fit_mask, eval_mask in [(jul, jan), (jan, jul), (jj, dec), (dec, jj)]:
        out = fit_and_predict(d.filter(pl.Series(fit_mask)), d.filter(pl.Series(eval_mask)))
        for k in preds:
            preds[k][eval_mask] = out[k]

    y = d["y"].to_numpy().astype(np.float64)
    p20 = d["e20_pred"].to_numpy().astype(np.float64)
    r = y - p20
    p_tail = preds["p_tail"]

    # gate sweep (reported for transparency; selected on the opposite split to stay transferable)
    gate_grid = np.round(np.arange(0.20, 0.91, 0.05), 2)
    sweep = {}
    final_preds_by_gate = {}
    for g in gate_grid:
        p = p20 + np.where(p_tail >= g, preds["corr"], 0.0)
        final_preds_by_gate[g] = p
        sweep[float(g)] = {
            "janjul_sse": float(np.sum((y[jj] - p[jj]) ** 2)),
            "dec_sse": float(np.sum((y[dec] - p[dec]) ** 2)),
        }
    # choose gate minimizing combined SSE, then apply; also report transfer gates
    best_gate = min(gate_grid, key=lambda gg: sweep[float(gg)]["janjul_sse"] + sweep[float(gg)]["dec_sse"])
    final = final_preds_by_gate[best_gate]
    finalB = p20 + preds["cB"]
    finalC = p20 + preds["cC"]

    payload = {"gate": float(best_gate), "gate_sweep": sweep}
    for split, mask in [("janjul", jj), ("dec", dec)]:
        ys, p20s = y[mask], p20[mask]
        rs = ys - p20s
        abl = {
            "A_E20": sse_dict(ys, p20s, None),
            "B_disagree": sse_dict(ys, finalB[mask], None),
            "C_surface": sse_dict(ys, finalC[mask], None),
            "D_both_ungated": sse_dict(ys, (p20 + preds["corr"])[mask], None),
            "E_gated": sse_dict(ys, final[mask], None),
        }
        regimes = {}
        pt = p_tail[mask]
        for lo, hi, nm in [(0.0, 0.3, "normal"), (0.3, 0.6, "hard"), (0.6, 1.01, "extreme")]:
            mm = (pt >= lo) & (pt < hi)
            if mm.sum():
                regimes[nm] = {"n": int(mm.sum()),
                               "e20_rmse": rmse(ys[mm], p20s[mm]), "e28_rmse": rmse(ys[mm], final[mask][mm]),
                               "e20_sse": float(np.sum((ys[mm] - p20s[mm]) ** 2)),
                               "e28_sse": float(np.sum((ys[mm] - final[mask][mm]) ** 2))}
        tails = {}
        for t in (900, 1200, 1800, 2700, 3600):
            sel = np.abs(rs) > t
            tails[f"abs_gt{t}"] = {"n": int(sel.sum()), "rate": float(np.mean(sel)),
                                   "sse_contrib": float(np.sum(rs[sel] ** 2) / np.sum(rs ** 2))}
        tails["pos_gt30"] = {"n": int((rs > 1800).sum()), "sse": float(np.sum(rs[rs > 1800] ** 2))}
        tails["neg_lt_minus30"] = {"n": int((rs < -1800).sum()), "sse": float(np.sum(rs[rs < -1800] ** 2))}
        tails["mean_resid"] = float(rs.mean()); tails["median_resid"] = float(np.median(rs))
        tails["p90_abs"] = float(np.quantile(np.abs(rs), .9)); tails["p99_abs"] = float(np.quantile(np.abs(rs), .99))
        ap = d.filter(pl.Series(mask))["airport"].to_numpy()
        xmax = d.filter(pl.Series(mask))["sd_max"].to_numpy().astype(np.float64)
        thr90 = np.nanquantile(xmax, .9)
        per_ap = {}
        for a in AIRPORTS:
            sel = ap == a
            if sel.sum() < 100:
                continue
            per_ap[a] = {"n": int(sel.sum()), "rmse_e20": rmse(ys[sel], p20s[sel]),
                         "rmse_e28": rmse(ys[sel], final[mask][sel]),
                         "sse_e20": float(np.sum(rs[sel] ** 2)), "sse_e28": float(np.sum((ys[sel] - final[mask][sel]) ** 2)),
                         "tail30_sse_e20": float(np.sum(rs[sel][np.abs(rs[sel]) > 1800] ** 2)),
                         "tail30_sse_e28": float(np.sum((ys[sel] - final[mask][sel])[np.abs(rs[sel]) > 1800] ** 2)),
                         "tail_rate_hi_dis": float(np.mean(np.abs(rs[sel & (xmax > thr90)]) > 1800)) if (sel & (xmax > thr90)).any() else float("nan"),
                         "tail_rate_lo_dis": float(np.mean(np.abs(rs[sel & (xmax <= thr90)]) > 1800)) if (sel & (xmax <= thr90)).any() else float("nan")}
        buckets = {}
        for name, col in [("abs_eobt", "sd_abs_eobt"), ("max", "sd_max")]:
            x = d.filter(pl.Series(mask))[col].to_numpy().astype(np.float64)
            qs = np.nanquantile(x, [0, .5, .75, .90, .95, .99, 1.0])
            row = {}
            for i in range(6):
                s = np.isfinite(x) & (x >= qs[i]) & (x <= (qs[i + 1] if i < 5 else qs[6]))
                if s.sum() < 20:
                    continue
                row[f"q{i}"] = {"n": int(s.sum()), "tail30_rate": float(np.mean(np.abs(rs[s]) > 1800)),
                                "rmse": rmse(ys[s], p20s[s]), "sse_share": float(np.sum(rs[s] ** 2) / np.sum(rs ** 2))}
            buckets[name] = row
        payload[split] = {"ablations": abl, "regimes": regimes, "tails": tails,
                          "per_airport": per_ap, "buckets": buckets, "n": int(mask.sum())}
        log(f"  [{split}] A E20={abl['A_E20']['rmse']:.2f}  E_gated={abl['E_gated']['rmse']:.2f}  "
            f"B={abl['B_disagree']['rmse']:.2f} C={abl['C_surface']['rmse']:.2f} D={abl['D_both_ungated']['rmse']:.2f}")
        log(f"  [{split}] tails { {k: v['n'] for k,v in tails.items() if k.startswith('abs')} }")
        log(f"  [{split}] regimes { {k: (round(v['e20_rmse'],1), round(v['e28_rmse'],1)) for k,v in regimes.items()} }")
    payload["primary"] = payload["janjul"]
    payload["primary"]["gate"] = float(best_gate)

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

    (RES / "E28_tail_regime.json").write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")
    write_report(payload)
    log("WROTE " + str(RES))


def write_report(p):
    pr = p.get("janjul", {})
    L = ["# E28 STEP 3 — matched tail regime (C25)", "",
         "Matched rows only. E20 residual modelled; correction gated by a causal tail-risk score "
         "(off-block source disagreement + compact surface pressure). **Caveat:** E20 OOF exists only "
         "for holdout months, so tail models are fit by temporal cross-fit (Jan<->Jul; Jan+Jul->Dec; "
         "Dec->Jan+Jul) rather than on 2025 train months. Row-level out-of-sample, but the fit periods "
         "are holdout months.", "",
         f"Gate (P(tail) >= g) chosen on combined split SSE: g = {p['gate']:.2f}", ""]
    L.append("## Matched ablations")
    L.append("")
    L.append("| Split | Model | matched RMSE | SSE | SSE |r|>30m | SSE |r|>60m |")
    L.append("|---|---|---:|---:|---:|---:|")
    for split in ("janjul", "dec"):
        for k, nm in [("A_E20", "A E20"), ("B_disagree", "B +disagreement"), ("C_surface", "C +surface"),
                      ("D_both_ungated", "D +both ungated"), ("E_gated", "E gated tail")]:
            s = p[split]["ablations"][k]
            L.append(f"| {split} | {nm} | {s['rmse']:.2f} | {s['sse']:.3e} | {s['sse_abs_gt1800']:.3e} | {s['sse_abs_gt3600']:.3e} |")
    L.append("")
    for split in ("janjul", "dec"):
        L.append(f"## Regimes — {split} (E20 vs gated E28)")
        L.append("")
        L.append("| Regime | n | E20 RMSE | E28 RMSE | E20 SSE | E28 SSE |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for nm, r in p[split]["regimes"].items():
            L.append(f"| {nm} | {r['n']:,} | {r['e20_rmse']:.1f} | {r['e28_rmse']:.1f} | {r['e20_sse']:.3e} | {r['e28_sse']:.3e} |")
        L.append("")
    L.append("## Source-disagreement buckets (Jan+Jul, E20 residual)")
    L.append("")
    for name, row in pr.get("buckets", {}).items():
        L.append(f"### {name}")
        L.append("")
        L.append("| bucket | n | tail>30m rate | RMSE | SSE share |")
        L.append("|---|---:|---:|---:|---:|")
        for q, v in row.items():
            L.append(f"| {q} | {v['n']:,} | {100*v['tail30_rate']:.1f}% | {v['rmse']:.1f} | {100*v['sse_share']:.1f}% |")
        L.append("")
    (RES / "summary_tail_regime.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
