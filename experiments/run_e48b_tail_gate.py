"""E48b — matched tail-gated residual (E36 features, no LIRF unmatched G).

Train the residual on high-y rows only, apply only when e20 or hat says tail.
Cross-month OOF. Splice matched rows only.
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
from common import load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E48"
SEED = 1
NUM = ["mvt_sched", "mvt_aobt", "aobt_eobt", "sd_eobt", "sd_lobt", "sd_iobt", "hour", "dow", "month", "e20_pred"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def pdf(df):
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def fit(tr, target):
    m = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
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
        f = f.filter(~pl.col("unmatched")).filter(pl.col("e20_pred").is_finite())
        out[split] = f.with_columns(pl.col("month").cast(pl.Int64).alias("_m"))
    return out


def run_pair(tr, ev, y_cut=1800.0):
    ytr = tr["y"].to_numpy().astype(float)
    yev = ev["y"].to_numpy().astype(float)
    etr = tr["e20_pred"].to_numpy().astype(float)
    eev = ev["e20_pred"].to_numpy().astype(float)
    # all-row residual (E36)
    m_all = fit(tr, ytr - etr)
    hat_all = np.asarray(m_all.predict(pdf(ev)), float)
    # tail-trained
    tr_t = tr.filter(pl.col("y") > y_cut)
    hat_tail = np.zeros_like(eev)
    if tr_t.height >= 200:
        m_t = fit(tr_t, tr_t["y"].to_numpy().astype(float) - tr_t["e20_pred"].to_numpy().astype(float))
        hat_tail = np.asarray(m_t.predict(pdf(ev)), float)
    # LIRF tail-trained
    hat_lirf = np.zeros_like(eev)
    tr_l = tr.filter((pl.col("airport") == "LIRF") & (pl.col("y") > y_cut))
    ev_l = ev.filter(pl.col("airport") == "LIRF")
    if tr_l.height >= 150 and ev_l.height:
        m_l = fit(tr_l, tr_l["y"].to_numpy().astype(float) - tr_l["e20_pred"].to_numpy().astype(float))
        mask = (ev["airport"] == "LIRF").to_numpy()
        hat_lirf[mask] = np.asarray(m_l.predict(pdf(ev_l)), float)
    p90 = float(np.quantile(etr, 0.90))
    return yev, eev, hat_all, hat_tail, hat_lirf, p90, ev["MVT_ID_mvt"].to_numpy()


def main():
    log("E48b load")
    fr = load_frames()
    # janjul OOF
    ys, es, ha, ht, hl, ids = [], [], [], [], [], []
    p90s = []
    for a, b in ((1, 7), (7, 1)):
        tr = fr["janjul"].filter(pl.col("_m") == a)
        ev = fr["janjul"].filter(pl.col("_m") == b)
        yev, eev, ahat, that, lhat, p90, idv = run_pair(tr, ev)
        ys.append(yev); es.append(eev); ha.append(ahat); ht.append(that); hl.append(lhat); ids.append(idv); p90s.append(p90)
        log(f"  fold {a}->{b} n={len(yev)} p90={p90:.0f} tail_train {tr.filter(pl.col('y')>1800).height}")
    ojj = {
        "y": np.concatenate(ys),
        "e": np.concatenate(es),
        "all": np.concatenate(ha),
        "tail": np.concatenate(ht),
        "lirf": np.concatenate(hl),
        "ids": np.concatenate(ids),
        "p90": float(np.mean(p90s)),
    }
    log("  Dec")
    tr = fr["janjul"]
    ev = fr["dec"]
    yev, eev, ahat, that, lhat, p90, idv = run_pair(tr, ev)
    odc = {"y": yev, "e": eev, "all": ahat, "tail": that, "lirf": lhat, "ids": idv, "p90": p90}

    payload = {"splits": {}}
    for split, o, full in (("janjul", ojj, fr["janjul"]), ("dec", odc, fr["dec"])):
        y, e = o["y"], o["e"]
        p90 = o["p90"]
        cands = {"e20": e}
        for lam in (0.3, 0.5, 1.0):
            cands[f"all_l{lam}"] = e + lam * o["all"]
            # gate: apply tail hat only if e20 > p90 or hat > 200
            g = (e > p90) | (o["tail"] > 200)
            cands[f"tailgate_l{lam}"] = e + lam * o["tail"] * g.astype(float)
            g2 = (e > p90) | (o["all"] > 200)
            cands[f"allgate_l{lam}"] = e + lam * o["all"] * g2.astype(float)
            gl = (e > p90) | (o["lirf"] > 200)
            cands[f"lirfgate_l{lam}"] = e + lam * o["lirf"] * gl.astype(float)
            # ungated tail-trained
            cands[f"tail_l{lam}"] = e + lam * o["tail"]

        # full overall: need unmatched rows from oof
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20f = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        yf = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])
        out = {}
        base_m, base_t = rmse(y, e), top_sse(y, e, 0.01)
        for n, p in cands.items():
            pf = e20f.copy()
            pf[idx] = p
            out[n] = {
                "matched": float(rmse(y, p)),
                "top1": top_sse(y, p, 0.01),
                "overall": float(rmse(yf, pf)),
                "applied_frac": None,
            }
            log(
                f"  [{split}] {n:16s} {out[n]['matched']:.2f} ({out[n]['matched']-base_m:+.2f}) "
                f"top1 {(out[n]['top1']-base_t)/base_t:+.1%} overall {out[n]['overall']:.2f}"
            )
        payload["splits"][split] = {"e20_matched": base_m, "e20_top1": base_t, "cands": out}

    # GO
    jj, dc = payload["splits"]["janjul"]["cands"], payload["splits"]["dec"]["cands"]
    e20j, e20d = payload["splits"]["janjul"]["e20_matched"], payload["splits"]["dec"]["e20_matched"]
    t1j, t1d = payload["splits"]["janjul"]["e20_top1"], payload["splits"]["dec"]["e20_top1"]
    go = []
    for n in jj:
        if n == "e20":
            continue
        dj = e20j - jj[n]["matched"]
        dd = e20d - dc[n]["matched"]
        if dj >= 1.5 and dd >= 0.5 and jj[n]["top1"] <= t1j * 1.01 and dc[n]["top1"] <= t1d * 1.01:
            go.append((n, dj, dd))
            log(f"GO {n} jj {dj:+.2f} dec {dd:+.2f}")
    payload["go"] = go
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E48b_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    lines = ["# E48b — tail-gated matched residual", "", "| split | cand | matched | Δ | top1 rel | overall |", "|---|---|---:|---:|---:|---:|"]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        b = s["e20_matched"]
        t1 = s["e20_top1"]
        items = [("e20", s["cands"]["e20"])] + sorted(
            ((k, v) for k, v in s["cands"].items() if k != "e20"), key=lambda kv: kv[1]["matched"]
        )[:8]
        for k, c in items:
            lines.append(f"| {split} | {k} | {c['matched']:.2f} | {c['matched']-b:+.2f} | {(c['top1']-t1)/t1:+.1%} | {c['overall']:.2f} |")
    lines.append("\n**GO:** " + (", ".join(g[0] for g in go) if go else "none") + "\n")
    (RES / "E48b_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE E48b")


if __name__ == "__main__":
    main()
