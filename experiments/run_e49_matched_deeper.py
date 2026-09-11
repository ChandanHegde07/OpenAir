"""E49 — deeper matched tail on top of the v11 gate. No LIRF unmatched G.

Stage 1: E48b allgate residual (y − e20), λ/gate grid.
Stage 2: leftover residual on LIRF / LFPG / global, small shrinkage.
Stage 3: CatBoost residual diversity + NNLS with stage-1.

Cross-month OOF. Splice matched rows only onto E20 OOF (v8-equivalent on matched).
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
from catboost import CatBoostRegressor  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E49"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt", "hour", "dow", "month", "e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]
CAT_CB = CAT


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def cdf(df):
    x = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        x[c] = x[c].fillna("NA").astype(str)
    return x


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def fit_lgb(tr, target, leaves=31, min_child=80, n=400, lam=5.0):
    m = lgb.LGBMRegressor(
        n_estimators=n, learning_rate=0.04, num_leaves=leaves, min_child_samples=min_child,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=lam, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def fit_cb(tr, target):
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, loss_function="RMSE",
        random_seed=SEED, verbose=False, thread_count=-1, od_type="Iter", od_wait=50,
    )
    m.fit(cdf(tr), target, cat_features=CAT_CB)
    return m


def load_frames():
    dep = load_dep().with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_sched"),
            (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_lobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_iobt"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
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
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def hats_for_pair(tr, ev):
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    m_all = fit_lgb(tr, ytr - etr)
    hat = np.asarray(m_all.predict(pdf(ev)), float)
    m_d = fit_lgb(tr, ytr - Ptr, leaves=31, min_child=80)
    dhat = np.asarray(m_d.predict(pdf(ev)), float)
    rec = np.clip(Pev + dhat, 0, None)
    m_cb = fit_cb(tr, ytr - etr)
    chat = np.asarray(m_cb.predict(cdf(ev)), float)
    # LIRF leftover specialist on y - e20 (full LIRF matched, not tail-only)
    hat_l = np.zeros_like(eev)
    tr_l = tr.filter(pl.col("airport") == "LIRF")
    ev_l = ev.filter(pl.col("airport") == "LIRF")
    if tr_l.height >= 200 and ev_l.height:
        ml = fit_lgb(tr_l, tr_l["y"].to_numpy().astype(float) - tr_l["e20_pred"].to_numpy().astype(float), leaves=31, min_child=40, n=500, lam=8.0)
        mask = (ev["airport"] == "LIRF").to_numpy()
        hat_l[mask] = np.asarray(ml.predict(pdf(ev_l)), float)
    hat_p = np.zeros_like(eev)
    tr_p = tr.filter(pl.col("airport") == "LFPG")
    ev_p = ev.filter(pl.col("airport") == "LFPG")
    if tr_p.height >= 200 and ev_p.height:
        mp = fit_lgb(tr_p, tr_p["y"].to_numpy().astype(float) - tr_p["e20_pred"].to_numpy().astype(float), leaves=31, min_child=40, n=400, lam=8.0)
        mask = (ev["airport"] == "LFPG").to_numpy()
        hat_p[mask] = np.asarray(mp.predict(pdf(ev_p)), float)
    p90 = float(np.quantile(etr, 0.90))
    p85 = float(np.quantile(etr, 0.85))
    return {
        "y": yev, "e": eev, "P": Pev, "hat": hat, "dhat": dhat, "rec": rec, "cb": chat,
        "lirf": hat_l, "lfpg": hat_p, "p90": p90, "p85": p85,
        "ids": ev["MVT_ID_mvt"].to_numpy(), "ap": ev["airport"].to_numpy(),
    }


def concat_fold(parts):
    keys = ["y", "e", "P", "hat", "dhat", "rec", "cb", "lirf", "lfpg", "ids", "ap"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    out["p85"] = float(np.mean([p["p85"] for p in parts]))
    return out


def gated(e, hat, p90, hthr=200.0):
    return ((e > p90) | (hat > hthr)).astype(float)


def main():
    log("E49 load")
    fr = load_frames()
    parts = []
    for a, b in ((1, 7), (7, 1)):
        log(f"  fold {a}->{b}")
        parts.append(hats_for_pair(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b)))
    ojj = concat_fold(parts)
    log("  Dec")
    odc = hats_for_pair(fr["janjul"], fr["dec"])

    payload = {"splits": {}}

    def eval_o(name, o, split):
        y, e = o["y"], o["e"]
        p90, p85 = o["p90"], o["p85"]
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        cands = {"e20": e}
        # v11-like: gated residual λ=0.5, hthr=200, p90
        for lam in (0.4, 0.5, 0.6, 0.7):
            for hthr in (100, 200, 400):
                g = gated(e, o["hat"], p90, hthr)
                cands[f"ghat_l{lam}_h{hthr}"] = e + lam * o["hat"] * g
                g85 = gated(e, o["hat"], p85, hthr)
                cands[f"g85_l{lam}_h{hthr}"] = e + lam * o["hat"] * g85
            cands[f"ungate_l{lam}"] = e + lam * o["hat"]
            cands[f"rec_l{lam}"] = (1 - lam) * e + lam * o["rec"]
            g = gated(e, o["rec"] - e, p90, 200)
            cands[f"grec_l{lam}"] = e + lam * (o["rec"] - e) * g
            cands[f"cb_l{lam}"] = e + lam * o["cb"]
            gcb = gated(e, o["cb"], p90, 200)
            cands[f"gcb_l{lam}"] = e + lam * o["cb"] * gcb
        # v11-like then LIRF extra
        v11 = e + 0.5 * o["hat"] * gated(e, o["hat"], p90, 200)
        for lam in (0.2, 0.3, 0.5):
            cands[f"v11_lirf_l{lam}"] = v11 + lam * o["lirf"]
            gl = gated(v11, o["lirf"], p90, 200)
            cands[f"v11_lirfgate_l{lam}"] = v11 + lam * o["lirf"] * gl
            cands[f"v11_lfpg_l{lam}"] = v11 + lam * o["lfpg"]
            cands[f"v11_both_l{lam}"] = v11 + lam * (o["lirf"] + o["lfpg"])
        # NNLS of e20, e20+hat, rec, cb on this split (diagnostic; will check both)
        X = np.vstack([e, np.clip(e + o["hat"], 0, None), o["rec"], np.clip(e + o["cb"], 0, None)]).T
        w, _ = nnls(X, y)
        w = w / w.sum() if w.sum() > 0 else w
        cands["nnls4"] = X @ w
        o["_nnls_w"] = w.tolist()
        # gated NNLS
        nn = X @ w
        g = gated(e, nn - e, p90, 200)
        cands["nnls4_g0.5"] = e + 0.5 * (nn - e) * g

        base_m, base_t = rmse(y, e), top_sse(y, e, 0.01)
        out = {}
        for n, p in cands.items():
            pf = e20f.copy()
            pf[idx] = p
            out[n] = {
                "matched": float(rmse(y, p)),
                "top1": top_sse(y, p, 0.01),
                "overall": float(rmse(yf, pf)),
            }
        payload["splits"][name] = {
            "e20_matched": base_m, "e20_top1": base_t, "e20_overall": float(rmse(yf, e20f)),
            "p90": p90, "nnls_w": o["_nnls_w"], "cands": out,
        }
        ranked = sorted(out.items(), key=lambda kv: kv[1]["matched"])[:10]
        for n, c in ranked:
            log(
                f"  [{name}] {n:22s} {c['matched']:.2f} ({c['matched']-base_m:+.2f}) "
                f"top1 {(c['top1']-base_t)/base_t:+.1%} ov {c['overall']:.2f}"
            )

    eval_o("janjul", ojj, "janjul")
    eval_o("dec", odc, "dec")

    jj, dc = payload["splits"]["janjul"]["cands"], payload["splits"]["dec"]["cands"]
    e20j, e20d = payload["splits"]["janjul"]["e20_matched"], payload["splits"]["dec"]["e20_matched"]
    t1j, t1d = payload["splits"]["janjul"]["e20_top1"], payload["splits"]["dec"]["e20_top1"]
    v11j = jj.get("ghat_l0.5_h200", jj.get("e20"))
    v11d = dc.get("ghat_l0.5_h200", dc.get("e20"))
    go = []
    for n in jj:
        if n == "e20":
            continue
        dj = e20j - jj[n]["matched"]
        dd = e20d - dc[n]["matched"]
        # must beat v11-like gate as well as E20
        beat_v11 = jj[n]["matched"] <= v11j["matched"] - 0.3 and dc[n]["matched"] <= v11d["matched"] + 0.15
        tail_ok = jj[n]["top1"] <= t1j * 1.01 and dc[n]["top1"] <= t1d * 1.01
        if dj >= 2.0 and dd >= 1.0 and beat_v11 and tail_ok:
            go.append({"name": n, "jj": dj, "dec": dd, "jj_vs_v11": v11j["matched"] - jj[n]["matched"]})
            log(f"GO {n} vs E20 {dj:+.2f}/{dd:+.2f} vs v11 {v11j['matched']-jj[n]['matched']:+.2f}")
    payload["go"] = go
    payload["v11_ref"] = {"janjul": v11j, "dec": v11d}
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E49_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    lines = [
        "# E49 — deeper matched tail",
        "",
        f"v11-like gate ref: Jan+Jul {v11j['matched']:.2f} / Dec {v11d['matched']:.2f}",
        "",
        "| split | cand | matched | ΔE20 | top1 | overall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        b = s["e20_matched"]
        t1 = s["e20_top1"]
        keep = ["e20", "ghat_l0.5_h200", "ghat_l0.6_h200", "ghat_l0.7_h200", "g85_l0.5_h200",
                "v11_lirf_l0.3", "v11_both_l0.3", "gcb_l0.5", "nnls4_g0.5", "grec_l0.5"]
        extra = [k for k, _ in sorted(s["cands"].items(), key=lambda kv: kv[1]["matched"])[:6]]
        shown = []
        for k in keep + extra:
            if k in s["cands"] and k not in shown:
                shown.append(k)
        for k in shown:
            c = s["cands"][k]
            lines.append(f"| {split} | {k} | {c['matched']:.2f} | {c['matched']-b:+.2f} | {(c['top1']-t1)/t1:+.1%} | {c['overall']:.2f} |")
    lines.append("\n**GO vs v11:** " + (", ".join(g["name"] for g in go) if go else "none") + "\n")
    (RES / "E49_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE E49")


if __name__ == "__main__":
    main()
