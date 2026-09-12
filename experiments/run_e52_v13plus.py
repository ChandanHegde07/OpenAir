"""E52 — beat v13 (238.52 / 212.08) on BOTH splits. No LIRF unmatched G.

v13 = grec λ=0.5 + leftover hat λ=0.25.
Variants: leftover clip/positive-only, LIRF extra residual, train-only geo_mean,
operator cat (no rolling).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_dep, rmse  # noqa: E402
from matched_submit import CAT, NUM, add_extra, e20_col, fit_lgb, gated, grec, log, pdf  # noqa: E402

RES = HERE / "results" / "E52"
RES.mkdir(parents=True, exist_ok=True)
V13 = (238.52, 212.08)


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def load_frames():
    dep = add_extra(load_dep()).with_columns(
        pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"),
        pl.col("WK_TBL_CAT_flt").fill_null("NA"),
    )
    out = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        f = dep.join(e20_col(oof).join(oof.select(["MVT_ID_mvt", "unmatched"]), on="MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
        f = f.filter(~pl.col("unmatched") & pl.col("e20_pred").is_finite() & pl.col("y").is_finite() & pl.col("mvt_aobt").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def add_geo(tr, ev):
    g = tr.group_by(["airport", "STAND_mvt", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("_g1"))
    g2 = tr.group_by(["airport", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("_g2"))
    g3 = tr.group_by("airport").agg(pl.col("y").median().alias("_g3"))
    def join(d):
        d = d.join(g, on=["airport", "STAND_mvt", "RUNWAY_mvt"], how="left")
        d = d.join(g2, on=["airport", "RUNWAY_mvt"], how="left")
        d = d.join(g3, on="airport", how="left")
        return d.with_columns(pl.coalesce(["_g1", "_g2", "_g3"]).alias("geo_mean"))
    return join(tr), join(ev)


def pair(tr, ev, extra_cat=None, use_geo=False):
    if use_geo:
        tr, ev = add_geo(tr, ev)
        cols = NUM + ["geo_mean"]
    else:
        cols = list(NUM)
    cats = list(CAT) + (extra_cat or [])
    # pdf with extra cats: pad
    def X(d):
        sel = cols + [c for c in cats if c in d.columns]
        p = d.select(sel).to_pandas()
        for c in cats:
            if c in p.columns:
                p[c] = p[c].fillna("NA").astype("category")
        return p

    ytr = tr["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    yev = ev["y"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)
    import lightgbm as lgb
    kw = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1)
    cats_present = [c for c in cats if c in X(tr).columns]
    dmod = lgb.LGBMRegressor(random_state=1, **kw)
    dmod.fit(X(tr), ytr - Ptr, categorical_feature=cats_present)
    rec = np.clip(Pev + np.asarray(dmod.predict(X(ev)), float), 0, None)
    hmod = lgb.LGBMRegressor(random_state=7, **kw)
    hmod.fit(X(tr), ytr - etr, categorical_feature=cats_present)
    hat = np.asarray(hmod.predict(X(ev)), float)
    p90 = float(np.quantile(etr, 0.90))
    g = grec(eev, rec, 0.5, p90, 200)
    v13 = g + 0.25 * hat
    # LIRF extra residual on y - e20, applied only LIRF
    lirf = np.zeros_like(eev)
    tr_l = tr.filter(pl.col("airport") == "LIRF")
    ev_l = ev.filter(pl.col("airport") == "LIRF")
    if tr_l.height >= 300 and ev_l.height:
        ml = lgb.LGBMRegressor(random_state=11, **kw)
        ml.fit(X(tr_l), tr_l["y"].to_numpy().astype(float) - tr_l["e20_pred"].to_numpy().astype(float), categorical_feature=cats_present)
        mask = (ev["airport"] == "LIRF").to_numpy()
        lirf[mask] = np.asarray(ml.predict(X(ev_l)), float)
    return dict(y=yev, e=eev, rec=rec, hat=hat, g=g, v13=v13, lirf=lirf, p90=p90,
                ids=ev["MVT_ID_mvt"].to_numpy(), ap=ev["airport"].to_numpy())


def catk(parts, k):
    return np.concatenate([p[k] for p in parts])


def main():
    log("E52 load")
    fr = load_frames()
    specs = {
        "base": dict(extra_cat=None, use_geo=False),
        "geo": dict(extra_cat=None, use_geo=True),
        "op": dict(extra_cat=["AIRCRAFT_OPERATOR_flt"], use_geo=False),
    }
    payload = {}
    for spec_name, spec in specs.items():
        log(f" spec {spec_name}")
        parts = []
        for a, b in ((1, 7), (7, 1)):
            parts.append(pair(fr["janjul"].filter(pl.col("_m") == a), fr["janjul"].filter(pl.col("_m") == b), **spec))
        ojj = {k: catk(parts, k) for k in ("y", "e", "rec", "hat", "g", "v13", "lirf", "ids", "ap")}
        ojj["p90"] = float(np.mean([p["p90"] for p in parts]))
        odc = pair(fr["janjul"], fr["dec"], **spec)
        payload[spec_name] = {}
        for split, o in (("janjul", ojj), ("dec", odc)):
            y, e = o["y"], o["e"]
            p90 = o["p90"]
            g, hat, v13, lirf = o["g"], o["hat"], o["v13"], o["lirf"]
            rec = o["rec"]
            cands = {
                "e20": e,
                "v13": v13,
                "h20": g + 0.20 * hat,
                "h22": g + 0.22 * hat,
                "h28": g + 0.28 * hat,
                "hpos": g + 0.25 * np.maximum(hat, 0),
                "hclip": g + 0.25 * np.clip(hat, -400, 900),
                "hgate": g + 0.25 * hat * ((np.abs(hat) > 80) | (e > p90)).astype(float),
                "lirf15": v13 + 0.15 * lirf,
                "lirf20": v13 + 0.20 * lirf,
                "lirf10": v13 + 0.10 * lirf,
                "grec_only": g,
            }
            oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
            e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
            yf = oof["y"].to_numpy().astype(float)
            pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
            idx = np.array([pos[int(i)] for i in o["ids"]])
            base = rmse(y, e)
            t1 = top_sse(y, e, 0.01)
            out = {}
            for n, p in cands.items():
                pf = e20f.copy()
                pf[idx] = p
                out[n] = {"matched": float(rmse(y, p)), "top1": top_sse(y, p, 0.01), "overall": float(rmse(yf, pf))}
            payload[spec_name][split] = out
            for n, c in sorted(out.items(), key=lambda kv: kv[1]["matched"])[:7]:
                log(f"  [{spec_name}/{split}] {n:10s} {c['matched']:.2f} ({c['matched']-base:+.2f}) ov {c['overall']:.2f}")

    # GO: both splits matched better than V13 by 0.20 / 0.10
    go = []
    for spec, splits in payload.items():
        if "janjul" not in splits or "dec" not in splits:
            continue
        for n in splits["janjul"]:
            if n in ("e20", "v13", "grec_only"):
                continue
            jj = splits["janjul"][n]["matched"]
            dc = splits["dec"][n]["matched"]
            if jj <= V13[0] - 0.20 and dc <= V13[1] - 0.10:
                go.append((spec, n, jj, dc))
                log(f"GO {spec}/{n} {jj:.2f}/{dc:.2f}")
    payload["go"] = [{"spec": a, "cand": b, "jj": c, "dec": d} for a, b, c, d in go]
    (RES / "E52_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE E52")
    log("GO list: " + (str(go) if go else "none"))


if __name__ == "__main__":
    main()
