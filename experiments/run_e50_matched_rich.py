"""E50 — richer matched P+Δ on top of v12 grec. No LIRF unmatched G.

v12 LB 284.97. Extra clocks + segment cats + a second LGB Δ (diversity) +
leftover residual after grec. Cross-month OOF. Must beat grec_l0.5.
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
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E50"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
NUM = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt",
]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "ac_family"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].fillna("NA").astype("category")
    return p


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def fit_lgb(tr, target, leaves=31, seed=SEED, min_child=80, n=400, lam=5.0, lr=0.04):
    m = lgb.LGBMRegressor(
        n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=min_child,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=lam, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def load_frames():
    dep = load_dep().with_columns(
        [
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            pl.col("MARKET_SEGMENT_flt").fill_null("NA"),
            pl.col("ac_family").fill_null("NA"),
        ]
    )
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        f = dep.join(
            oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20_pred", e20)),
            on="MVT_ID_mvt",
            how="inner",
        )
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite()).filter(pl.col("y").is_finite())
        f = f.filter(pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def hats(tr, ev):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    # Δ models (diversity)
    d1 = fit_lgb(tr, ytr - Ptr, leaves=31, seed=1)
    d2 = fit_lgb(tr, ytr - Ptr, leaves=63, seed=2, min_child=50, n=500, lam=3.0, lr=0.03)
    rec1 = np.clip(Pev + np.asarray(d1.predict(pdf(ev)), float), 0, None)
    rec2 = np.clip(Pev + np.asarray(d2.predict(pdf(ev)), float), 0, None)
    recm = 0.5 * rec1 + 0.5 * rec2
    # residual vs e20
    r1 = fit_lgb(tr, ytr - etr, leaves=31, seed=3)
    hat = np.asarray(r1.predict(pdf(ev)), float)
    p90 = float(np.quantile(etr, 0.90))
    return {
        "y": yev, "e": eev, "P": Pev, "rec1": rec1, "rec2": rec2, "recm": recm,
        "hat": hat, "p90": p90, "ids": ev["MVT_ID_mvt"].to_numpy(),
        "ap": ev["airport"].to_numpy(),
    }


def concat_parts(parts):
    keys = ["y", "e", "P", "rec1", "rec2", "recm", "hat", "ids", "ap"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    return out


def gated(e, corr, p90, hthr=200.0):
    return ((e > p90) | (corr > hthr)).astype(float)


def main():
    log("E50 load")
    fr = load_frames()
    parts = []
    for a, b in ((1, 7), (7, 1)):
        log(f"  fold {a}->{b}")
        parts.append(hats(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    ojj = concat_parts(parts)
    log("  Dec")
    odc = hats(fr["janjul"], fr["dec"])

    payload = {"splits": {}}

    def eval_o(name, o, split):
        y, e = o["y"], o["e"]
        p90 = o["p90"]
        grec = e + 0.5 * (o["rec1"] - e) * gated(e, o["rec1"] - e, p90, 200)
        # leftover after grec
        # we don't have a leftover model OOF easily without a third fit; use hat on leftover as proxy
        cands = {"e20": e, "grec_v12": grec}
        for lam in (0.4, 0.5, 0.6, 0.7):
            g1 = gated(e, o["rec1"] - e, p90, 200)
            g2 = gated(e, o["rec2"] - e, p90, 200)
            gm = gated(e, o["recm"] - e, p90, 200)
            cands[f"grec1_l{lam}"] = e + lam * (o["rec1"] - e) * g1
            cands[f"grec2_l{lam}"] = e + lam * (o["rec2"] - e) * g2
            cands[f"grecm_l{lam}"] = e + lam * (o["recm"] - e) * gm
            cands[f"rec1_l{lam}"] = (1 - lam) * e + lam * o["rec1"]
            cands[f"recm_l{lam}"] = (1 - lam) * e + lam * o["recm"]
        # leftover: grec + small hat
        for lam in (0.15, 0.25, 0.35):
            cands[f"grec_hat_l{lam}"] = grec + lam * o["hat"]
            gh = gated(grec, o["hat"], p90, 200)
            cands[f"grec_hatg_l{lam}"] = grec + lam * o["hat"] * gh
        # NNLS of e, rec1, rec2, e+hat
        X = np.vstack([e, o["rec1"], o["rec2"], np.clip(e + o["hat"], 0, None)]).T
        w, _ = nnls(X, y)
        w = w / w.sum() if w.sum() > 0 else w
        o["_nnls_w"] = w.tolist()
        nn = X @ w
        cands["nnls"] = nn
        g = gated(e, nn - e, p90, 200)
        cands["nnls_g0.5"] = e + 0.5 * (nn - e) * g
        cands["nnls_g0.7"] = e + 0.7 * (nn - e) * g
        # 3-way fixed weights from E49 (e20, rec, cb~rec2)
        cands["w173350"] = 0.17 * e + 0.33 * o["rec1"] + 0.50 * o["rec2"]
        cands["w173350_g"] = e + 0.5 * ((0.17 * e + 0.33 * o["rec1"] + 0.50 * o["rec2"]) - e) * gated(
            e, o["rec1"] - e, p90, 200
        )

        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        base_m, base_t = rmse(y, e), top_sse(y, e, 0.01)
        gref = rmse(y, grec)
        out = {}
        for n, p in cands.items():
            pf = e20f.copy()
            pf[idx] = p
            out[n] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "overall": float(rmse(yf, pf))}
        payload["splits"][name] = {
            "e20_matched": base_m, "e20_top1": base_t, "e20_overall": float(rmse(yf, e20f)),
            "grec": gref, "p90": p90, "nnls_w": o["_nnls_w"], "cands": out,
        }
        ranked = sorted(out.items(), key=lambda kv: kv[1]["matched"])[:8]
        for n, c in ranked:
            log(
                f"  [{name}] {n:16s} {c['matched']:.2f} (vsE20 {c['matched']-base_m:+.2f} vsGrec {c['matched']-gref:+.2f}) "
                f"top1 {(c['top1']-base_t)/base_t:+.1%} ov {c['overall']:.2f}"
            )

    eval_o("janjul", ojj, "janjul")
    eval_o("dec", odc, "dec")

    jj, dc = payload["splits"]["janjul"]["cands"], payload["splits"]["dec"]["cands"]
    gj, gd = payload["splits"]["janjul"]["grec"], payload["splits"]["dec"]["grec"]
    t1j, t1d = payload["splits"]["janjul"]["e20_top1"], payload["splits"]["dec"]["e20_top1"]
    go = []
    for n in jj:
        if n in ("e20", "grec_v12"):
            continue
        if jj[n]["matched"] <= gj - 0.4 and dc[n]["matched"] <= gd - 0.2 and jj[n]["top1"] <= t1j * 1.01 and dc[n]["top1"] <= t1d * 1.01:
            go.append(n)
            log(f"GO {n} vs grec {gj-jj[n]['matched']:+.2f}/{gd-dc[n]['matched']:+.2f}")
    payload["go"] = go
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E50_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    lines = [
        "# E50 — richer matched P+Δ",
        "",
        f"grec (v12-like) ref: Jan+Jul {gj:.2f} / Dec {gd:.2f}",
        "",
        "| split | cand | matched | vs grec | top1 | overall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        g = s["grec"]
        t1 = s["e20_top1"]
        keys = ["e20", "grec_v12"] + [k for k, _ in sorted(
            ((k, v) for k, v in s["cands"].items() if k not in ("e20", "grec_v12")),
            key=lambda kv: kv[1]["matched"],
        )[:8]]
        seen = []
        for k in keys:
            if k in s["cands"] and k not in seen:
                seen.append(k)
                c = s["cands"][k]
                lines.append(f"| {split} | {k} | {c['matched']:.2f} | {c['matched']-g:+.2f} | {(c['top1']-t1)/t1:+.1%} | {c['overall']:.2f} |")
    lines.append("\n**GO vs grec:** " + (", ".join(go) if go else "none") + "\n")
    (RES / "E50_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE E50")


if __name__ == "__main__":
    main()
