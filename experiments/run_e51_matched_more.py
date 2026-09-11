"""E51 — squeeze matched P+Δ further. No LIRF unmatched G.

v13 LB 284.10 = rich grec + leftover hat λ=0.25.
Here: seed-avg Δ, LIRF/LFPG Δ specialists, CatBoost Δ on a subsample,
geo_mean prior, leftover hat only on focus airports.

Must beat v13-like grec+hat 238.52 / 212.08.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E51"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FOCUS = ["LIRF", "LFPG"]
NUM = [
    "mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt",
    "mvt_lobt", "mvt_iobt", "hour", "dow", "month", "e20_pred",
    "clock_std", "clock_range", "n_clocks", "abs_aobt_eobt", "geo_mean",
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


def fit_lgb(tr, target, leaves=31, seed=SEED, n=400):
    m = lgb.LGBMRegressor(
        n_estimators=n, learning_rate=0.04, num_leaves=leaves, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def add_geo(tr, ev):
    g = (
        tr.group_by(["airport", "STAND_mvt", "RUNWAY_mvt"])
        .agg(pl.col("y").median().alias("geo_mean"))
    )
    g2 = tr.group_by(["airport", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("geo_rwy"))
    g3 = tr.group_by(["airport"]).agg(pl.col("y").median().alias("geo_ap"))
    ev = ev.join(g, on=["airport", "STAND_mvt", "RUNWAY_mvt"], how="left")
    ev = ev.join(g2, on=["airport", "RUNWAY_mvt"], how="left")
    ev = ev.join(g3, on=["airport"], how="left")
    ev = ev.with_columns(
        pl.coalesce(["geo_mean", "geo_rwy", "geo_ap"]).alias("geo_mean")
    )
    tr = tr.join(g, on=["airport", "STAND_mvt", "RUNWAY_mvt"], how="left")
    tr = tr.join(g2, on=["airport", "RUNWAY_mvt"], how="left")
    tr = tr.join(g3, on=["airport"], how="left")
    tr = tr.with_columns(pl.coalesce(["geo_mean", "geo_rwy", "geo_ap"]).alias("geo_mean"))
    return tr, ev


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
    tr, ev = add_geo(tr, ev)
    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    recs = []
    for seed, leaves in ((1, 31), (2, 31), (3, 63)):
        m = fit_lgb(tr, ytr - Ptr, leaves=leaves, seed=seed, n=400 if leaves == 31 else 350)
        recs.append(np.clip(Pev + np.asarray(m.predict(pdf(ev)), float), 0, None))
    rec_avg = np.mean(np.stack(recs, axis=0), axis=0)
    rec_f = recs[0].copy()
    for ap in FOCUS:
        tr_a = tr.filter(pl.col("airport") == ap)
        ev_a = ev.filter(pl.col("airport") == ap)
        if tr_a.height < 300 or ev_a.height == 0:
            continue
        ma = fit_lgb(tr_a, tr_a["y"].to_numpy().astype(float) - np.clip(tr_a["mvt_aobt"].to_numpy().astype(float), 0, None), leaves=31, seed=1, n=350)
        mask = (ev["airport"] == ap).to_numpy()
        rec_f[mask] = np.clip(Pev[mask] + np.asarray(ma.predict(pdf(ev_a)), float), 0, None)
    # CatBoost Δ on subsample
    rec_cb = recs[0].copy()
    nsub = min(400_000, tr.height)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(tr.height, size=nsub, replace=False)
    # polars take
    tr_s = tr.with_row_index("_i").filter(pl.col("_i").is_in(idx.tolist())).drop("_i")
    cb = CatBoostRegressor(
        iterations=220, learning_rate=0.06, depth=5, l2_leaf_reg=6.0, loss_function="RMSE",
        random_seed=SEED, verbose=False, thread_count=-1,
    )
    xs = pdf(tr_s)
    for c in CAT:
        xs[c] = xs[c].astype(str)
    xe = pdf(ev)
    for c in CAT:
        xe[c] = xe[c].astype(str)
    ysub = tr_s["y"].to_numpy().astype(float) - np.clip(tr_s["mvt_aobt"].to_numpy().astype(float), 0, None)
    cb.fit(xs, ysub, cat_features=CAT)
    rec_cb = np.clip(Pev + np.asarray(cb.predict(xe), float), 0, None)
    hat = np.asarray(fit_lgb(tr, ytr - etr, leaves=31, seed=7, n=400).predict(pdf(ev)), float)
    p90 = float(np.quantile(etr, 0.90))
    return {
        "y": yev, "e": eev, "rec": recs[0], "rec_avg": rec_avg, "rec_f": rec_f, "rec_cb": rec_cb,
        "hat": hat, "p90": p90, "ids": ev["MVT_ID_mvt"].to_numpy(), "ap": ev["airport"].to_numpy(),
    }


def concat_parts(parts):
    keys = ["y", "e", "rec", "rec_avg", "rec_f", "rec_cb", "hat", "ids", "ap"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["p90"] = float(np.mean([p["p90"] for p in parts]))
    return out


def gated(e, corr, p90, hthr=200.0):
    return ((e > p90) | (corr > hthr)).astype(float)


def grec(e, rec, p90, lam=0.5):
    return e + lam * (rec - e) * gated(e, rec - e, p90, 200)


def main():
    log("E51 load")
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
        y, e, p90 = o["y"], o["e"], o["p90"]
        v13 = grec(e, o["rec"], p90) + 0.25 * o["hat"]
        cands = {"e20": e, "v13": v13}
        for tag, rec in (("r", o["rec"]), ("avg", o["rec_avg"]), ("foc", o["rec_f"]), ("cb", o["rec_cb"])):
            g = grec(e, rec, p90)
            cands[f"g_{tag}"] = g
            cands[f"g_{tag}_h25"] = g + 0.25 * o["hat"]
            cands[f"g_{tag}_h20"] = g + 0.20 * o["hat"]
            cands[f"g_{tag}_h30"] = g + 0.30 * o["hat"]
            # leftover only on focus airports
            h = o["hat"].copy()
            mask = np.isin(o["ap"], FOCUS)
            h2 = np.where(mask, o["hat"], 0.0)
            cands[f"g_{tag}_hfoc25"] = g + 0.25 * h2
        # blend rec and cb
        recb = 0.7 * o["rec"] + 0.3 * o["rec_cb"]
        g = grec(e, recb, p90)
        cands["g_mix_h25"] = g + 0.25 * o["hat"]
        reca = 0.5 * o["rec"] + 0.5 * o["rec_avg"]
        cands["g_ravg_h25"] = grec(e, reca, p90) + 0.25 * o["hat"]

        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        base_m, base_t = rmse(y, e), top_sse(y, e, 0.01)
        v13m = rmse(y, v13)
        out = {}
        for n, p in cands.items():
            pf = e20f.copy()
            pf[idx] = p
            out[n] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "overall": float(rmse(yf, pf))}
        payload["splits"][name] = {
            "e20_matched": base_m, "e20_top1": base_t, "v13": v13m, "p90": p90, "cands": out,
        }
        ranked = sorted(out.items(), key=lambda kv: kv[1]["matched"])[:8]
        for n, c in ranked:
            log(
                f"  [{name}] {n:16s} {c['matched']:.2f} (v13 {c['matched']-v13m:+.2f}) "
                f"top1 {(c['top1']-base_t)/base_t:+.1%} ov {c['overall']:.2f}"
            )

    eval_o("janjul", ojj, "janjul")
    eval_o("dec", odc, "dec")
    jj, dc = payload["splits"]["janjul"]["cands"], payload["splits"]["dec"]["cands"]
    vj, vd = payload["splits"]["janjul"]["v13"], payload["splits"]["dec"]["v13"]
    t1j, t1d = payload["splits"]["janjul"]["e20_top1"], payload["splits"]["dec"]["e20_top1"]
    go = []
    for n in jj:
        if n in ("e20", "v13"):
            continue
        if jj[n]["matched"] <= vj - 0.25 and dc[n]["matched"] <= vd - 0.10 and jj[n]["top1"] <= t1j * 1.01 and dc[n]["top1"] <= t1d * 1.01:
            go.append(n)
            log(f"GO {n} vs v13 {vj-jj[n]['matched']:+.2f}/{vd-dc[n]['matched']:+.2f}")
    payload["go"] = go
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E51_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    lines = ["# E51 — more matched P+Δ", "", f"v13 ref: {vj:.2f} / {vd:.2f}", "",
             "| split | cand | matched | vs v13 | top1 | overall |", "|---|---|---:|---:|---:|---:|"]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        v = s["v13"]
        t1 = s["e20_top1"]
        keys = ["e20", "v13"] + [k for k, _ in sorted(
            ((k, v) for k, v in s["cands"].items() if k not in ("e20", "v13")),
            key=lambda kv: kv[1]["matched"],
        )[:8]]
        seen = []
        for k in keys:
            if k in s["cands"] and k not in seen:
                seen.append(k)
                c = s["cands"][k]
                lines.append(f"| {split} | {k} | {c['matched']:.2f} | {c['matched']-v:+.2f} | {(c['top1']-t1)/t1:+.1%} | {c['overall']:.2f} |")
    lines.append("\n**GO vs v13:** " + (", ".join(go) if go else "none") + "\n")
    (RES / "E51_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE E51")


if __name__ == "__main__":
    main()
