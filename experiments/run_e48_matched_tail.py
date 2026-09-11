"""E48 — matched-tail attack (no LIRF unmatched G).

v9 LB 300.95 (worse than v8 288.90). Any splice goes onto v8 / E20 matched rows.

E36 = global residual LGB (~−1.4 s). E40 = global isotonic (overfit).
This run: per-airport isotonic, per-airport residual, LIRF-matched specialist,
and P-conditioned Δ = y − (MVT−AOBT) per airport.

Cross-month OOF: Jan↔Jul for primary; Jan+Jul → Dec.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E48"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FOCUS = ["LIRF", "LFPG", "LTFM", "EGLL"]
NUM = ["e20", "mvt_aobt", "aobt_eobt", "hour", "dow", "month"]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def e20_blend(oof: pl.DataFrame) -> np.ndarray:
    return (
        0.456 * oof["pred_C"].to_numpy()
        + 0.053 * oof["pred_D"].to_numpy()
        + 0.491 * oof["pred_E"].to_numpy()
    )


def pdf(df: pl.DataFrame) -> "object":
    p = df.select(NUM + CAT).to_pandas()
    for c in CAT:
        p[c] = p[c].astype("category")
    return p


def top_sse(y, p, frac=0.01):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(frac * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


def load_split(split: str) -> pl.DataFrame:
    dep = load_dep().with_columns(pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"))
    oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
    e20 = e20_blend(oof)
    f = dep.join(
        oof.select(["MVT_ID_mvt", "unmatched"]).with_columns(pl.Series("e20", e20)),
        on="MVT_ID_mvt",
        how="inner",
    )
    return f.with_columns(pl.col("month").cast(pl.Int64))


def fit_lgb(tr: pl.DataFrame, target: np.ndarray) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(pdf(tr), target, categorical_feature=CAT)
    return m


def oof_janjul(matched: pl.DataFrame, target_col: str):
    """Return OOF predictions of target_col residual vs e20, plus isotonic etc on rows."""
    parts_idx = []
    parts_iso = []
    parts_glob = []
    parts_ap = []
    parts_lirf = []
    parts_delta = []
    y_all = []
    e_all = []
    id_all = []
    for a, b in ((1, 7), (7, 1)):
        tr = matched.filter(pl.col("month") == a)
        ev = matched.filter(pl.col("month") == b)
        ytr = tr["y"].to_numpy().astype(float)
        yev = ev["y"].to_numpy().astype(float)
        etr = tr["e20"].to_numpy().astype(float)
        eev = ev["e20"].to_numpy().astype(float)
        Ptr = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
        Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)

        # global residual (E36-like)
        mg = fit_lgb(tr, ytr - etr)
        ghat = np.asarray(mg.predict(pdf(ev)), float)

        # per-airport residual: focus airports own model, rest global
        ap_hat = ghat.copy()
        for ap in FOCUS:
            tr_a = tr.filter(pl.col("airport") == ap)
            ev_a = ev.filter(pl.col("airport") == ap)
            if tr_a.height < 200 or ev_a.height == 0:
                continue
            ma = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            ma.fit(
                pdf(tr_a),
                tr_a["y"].to_numpy().astype(float) - tr_a["e20"].to_numpy().astype(float),
                categorical_feature=CAT,
            )
            mask = (ev["airport"] == ap).to_numpy()
            ap_hat[mask] = np.asarray(ma.predict(pdf(ev_a)), float)

        # LIRF-only residual; elsewhere 0
        lirf_hat = np.zeros_like(eev)
        tr_l = tr.filter(pl.col("airport") == "LIRF")
        ev_l = ev.filter(pl.col("airport") == "LIRF")
        if tr_l.height >= 200 and ev_l.height:
            ml = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=8.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            ml.fit(
                pdf(tr_l),
                tr_l["y"].to_numpy().astype(float) - tr_l["e20"].to_numpy().astype(float),
                categorical_feature=CAT,
            )
            mask = (ev["airport"] == "LIRF").to_numpy()
            lirf_hat[mask] = np.asarray(ml.predict(pdf(ev_l)), float)

        # per-airport isotonic e20 -> y
        iso = np.array(eev, copy=True)
        for ap in AIRPORTS:
            tr_a = tr.filter(pl.col("airport") == ap)
            ev_a = ev.filter(pl.col("airport") == ap)
            if tr_a.height < 200 or ev_a.height == 0:
                continue
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(tr_a["e20"].to_numpy().astype(float), tr_a["y"].to_numpy().astype(float))
            mask = (ev["airport"] == ap).to_numpy()
            iso[mask] = ir.predict(ev_a["e20"].to_numpy().astype(float))

        # Δ = y - P per airport, reconstruct P + Δ
        dhat = np.zeros_like(eev)
        for ap in AIRPORTS:
            tr_a = tr.filter(pl.col("airport") == ap)
            ev_a = ev.filter(pl.col("airport") == ap)
            if tr_a.height < 200 or ev_a.height == 0:
                continue
            md = lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=80,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=8.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            md.fit(
                pdf(tr_a),
                tr_a["y"].to_numpy().astype(float) - np.clip(tr_a["mvt_aobt"].to_numpy().astype(float), 0, None),
                categorical_feature=CAT,
            )
            mask = (ev["airport"] == ap).to_numpy()
            dhat[mask] = np.asarray(md.predict(pdf(ev_a)), float)
        rec = np.clip(Pev + dhat, 0, None)

        parts_glob.append(ghat)
        parts_ap.append(ap_hat)
        parts_lirf.append(lirf_hat)
        parts_iso.append(iso)
        parts_delta.append(rec)
        y_all.append(yev)
        e_all.append(eev)
        id_all.append(ev["MVT_ID_mvt"].to_numpy())
        parts_idx.append(ev.height)
    return {
        "y": np.concatenate(y_all),
        "e20": np.concatenate(e_all),
        "ids": np.concatenate(id_all),
        "glob": np.concatenate(parts_glob),
        "ap": np.concatenate(parts_ap),
        "lirf": np.concatenate(parts_lirf),
        "iso": np.concatenate(parts_iso),
        "delta": np.concatenate(parts_delta),
    }


def fit_eval_dec(matched_jj: pl.DataFrame, matched_dec: pl.DataFrame):
    tr, ev = matched_jj, matched_dec
    ytr = tr["y"].to_numpy().astype(float)
    yev = ev["y"].to_numpy().astype(float)
    etr = tr["e20"].to_numpy().astype(float)
    eev = ev["e20"].to_numpy().astype(float)
    Pev = np.clip(ev["mvt_aobt"].to_numpy().astype(float), 0, None)

    mg = fit_lgb(tr, ytr - etr)
    ghat = np.asarray(mg.predict(pdf(ev)), float)
    ap_hat = ghat.copy()
    for ap in FOCUS:
        tr_a = tr.filter(pl.col("airport") == ap)
        ev_a = ev.filter(pl.col("airport") == ap)
        if tr_a.height < 200 or ev_a.height == 0:
            continue
        ma = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=40,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        ma.fit(pdf(tr_a), tr_a["y"].to_numpy().astype(float) - tr_a["e20"].to_numpy().astype(float), categorical_feature=CAT)
        mask = (ev["airport"] == ap).to_numpy()
        ap_hat[mask] = np.asarray(ma.predict(pdf(ev_a)), float)

    lirf_hat = np.zeros_like(eev)
    tr_l = tr.filter(pl.col("airport") == "LIRF")
    ev_l = ev.filter(pl.col("airport") == "LIRF")
    if tr_l.height >= 200 and ev_l.height:
        ml = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        ml.fit(pdf(tr_l), tr_l["y"].to_numpy().astype(float) - tr_l["e20"].to_numpy().astype(float), categorical_feature=CAT)
        mask = (ev["airport"] == "LIRF").to_numpy()
        lirf_hat[mask] = np.asarray(ml.predict(pdf(ev_l)), float)

    iso = np.array(eev, copy=True)
    for ap in AIRPORTS:
        tr_a = tr.filter(pl.col("airport") == ap)
        ev_a = ev.filter(pl.col("airport") == ap)
        if tr_a.height < 200 or ev_a.height == 0:
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(tr_a["e20"].to_numpy().astype(float), tr_a["y"].to_numpy().astype(float))
        mask = (ev["airport"] == ap).to_numpy()
        iso[mask] = ir.predict(ev_a["e20"].to_numpy().astype(float))

    dhat = np.zeros_like(eev)
    for ap in AIRPORTS:
        tr_a = tr.filter(pl.col("airport") == ap)
        ev_a = ev.filter(pl.col("airport") == ap)
        if tr_a.height < 200 or ev_a.height == 0:
            continue
        md = lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        md.fit(
            pdf(tr_a),
            tr_a["y"].to_numpy().astype(float) - np.clip(tr_a["mvt_aobt"].to_numpy().astype(float), 0, None),
            categorical_feature=CAT,
        )
        mask = (ev["airport"] == ap).to_numpy()
        dhat[mask] = np.asarray(md.predict(pdf(ev_a)), float)
    rec = np.clip(Pev + dhat, 0, None)
    return {
        "y": yev,
        "e20": eev,
        "ids": ev["MVT_ID_mvt"].to_numpy(),
        "glob": ghat,
        "ap": ap_hat,
        "lirf": lirf_hat,
        "iso": iso,
        "delta": rec,
    }


def pack(y, e20, pred, oof_full_y, oof_full_e20, matched_idx):
    """matched metrics + overall if we splice pred into full e20 vector via ids handled by caller."""
    out = {
        "matched_rmse": float(rmse(y, pred)),
        "matched_sse": float(np.sum((y - pred) ** 2)),
        "top1_sse": top_sse(y, pred, 0.01),
        "top5_sse": top_sse(y, pred, 0.05),
        "e20_matched": float(rmse(y, e20)),
        "e20_top1": top_sse(y, e20, 0.01),
    }
    return out


def main():
    log("E48 load...")
    jj = load_split("janjul")
    dc = load_split("dec")
    jj_m = jj.filter(~pl.col("unmatched") & pl.col("e20").is_finite() & pl.col("y").is_finite())
    dc_m = dc.filter(~pl.col("unmatched") & pl.col("e20").is_finite() & pl.col("y").is_finite())
    log(f"  matched janjul {jj_m.height:,} dec {dc_m.height:,}")

    log("  OOF Jan↔Jul...")
    o_jj = oof_janjul(jj_m, "y")
    log("  fit Jan+Jul → Dec...")
    o_dc = fit_eval_dec(jj_m, dc_m)

    payload = {"splits": {}}
    lambdas = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0]

    def eval_split(name, o, full_df):
        y, e = o["y"], o["e20"]
        cands = {
            "e20": e,
            "iso": o["iso"],
            "delta_rec": o["delta"],
        }
        for lam in lambdas:
            cands[f"glob_l{lam}"] = e + lam * o["glob"]
            cands[f"ap_l{lam}"] = e + lam * o["ap"]
            cands[f"lirf_l{lam}"] = e + lam * o["lirf"]
        # blend iso with e20
        for lam in (0.3, 0.5, 1.0):
            cands[f"iso_l{lam}"] = (1 - lam) * e + lam * o["iso"]
            cands[f"delta_l{lam}"] = (1 - lam) * e + lam * o["delta"]

        # overall splice: replace matched rows in full OOF
        full_e = full_df["e20"].to_numpy().astype(float)
        full_y = full_df["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(full_df["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[int(i)] for i in o["ids"]])

        out = {}
        for cn, pred in cands.items():
            pfull = full_e.copy()
            pfull[idx] = pred
            out[cn] = {
                "matched_rmse": float(rmse(y, pred)),
                "top1_sse": top_sse(y, pred, 0.01),
                "top5_sse": top_sse(y, pred, 0.05),
                "overall": float(rmse(full_y, pfull)),
            }
        payload["splits"][name] = {
            "n_matched": int(len(y)),
            "e20_matched": float(rmse(y, e)),
            "e20_top1": top_sse(y, e, 0.01),
            "e20_overall": float(rmse(full_y, full_e)),
            "cands": out,
        }
        # print best vs e20 on matched rmse and top1
        base_m = rmse(y, e)
        base_t = top_sse(y, e, 0.01)
        ranked = sorted(out.items(), key=lambda kv: kv[1]["matched_rmse"])[:8]
        for cn, c in ranked:
            log(
                f"    [{name}] {cn:16s} matched {c['matched_rmse']:.2f} ({c['matched_rmse']-base_m:+.2f}) "
                f"top1 {c['top1_sse']:.3e} ({(c['top1_sse']-base_t)/base_t:+.1%}) overall {c['overall']:.2f}"
            )

    eval_split("janjul", o_jj, jj)
    eval_split("dec", o_dc, dc)

    # GO: both splits matched drop >= 1.5s, top1 not worse, Dec overall not worse
    jj_c = payload["splits"]["janjul"]["cands"]
    dc_c = payload["splits"]["dec"]["cands"]
    e20j = payload["splits"]["janjul"]["e20_matched"]
    e20d = payload["splits"]["dec"]["e20_matched"]
    t1j = payload["splits"]["janjul"]["e20_top1"]
    t1d = payload["splits"]["dec"]["e20_top1"]
    decision = {}
    names = set(jj_c) & set(dc_c)
    best = None
    for n in names:
        dj = e20j - jj_c[n]["matched_rmse"]
        dd = e20d - dc_c[n]["matched_rmse"]
        t1ok = jj_c[n]["top1_sse"] <= t1j * 1.02 and dc_c[n]["top1_sse"] <= t1d * 1.02
        go = dj >= 1.5 and dd >= 0.5 and t1ok
        decision[n] = {"janjul_drop": dj, "dec_drop": dd, "GO": bool(go)}
        if go and (best is None or dj + dd > best[1]):
            best = (n, dj + dd)
    payload["decision"] = decision
    payload["best_go"] = best[0] if best else None
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E48_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    go_list = [k for k, v in decision.items() if v["GO"]]
    log(f"GO candidates: {go_list or 'none'}")
    lines = [
        "# E48 — matched tail (no LIRF unmatched G)",
        "",
        "Baseline E20 matched Jan+Jul 244.76 / Dec 214.78. Production LB is **v8 288.90**; v9 scored **300.95**.",
        "",
        "| split | cand | matched RMSE | Δ | top1 SSE | overall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        base = s["e20_matched"]
        # show e20 + top 6 by matched rmse
        items = [("e20", s["cands"]["e20"])] + sorted(
            ((k, v) for k, v in s["cands"].items() if k != "e20"),
            key=lambda kv: kv[1]["matched_rmse"],
        )[:8]
        for k, c in items:
            lines.append(
                f"| {split} | {k} | {c['matched_rmse']:.2f} | {c['matched_rmse']-base:+.2f} | "
                f"{c['top1_sse']:.3e} | {c['overall']:.2f} |"
            )
    lines += ["", "## GO", ""]
    if best:
        lines.append(f"**GO: {best[0]}**")
    else:
        lines.append("**NO-GO** — no candidate beat E20 by ≥1.5 s Jan+Jul and ≥0.5 s Dec with tail not worse.")
    lines.append("\nDo not touch LIRF unmatched G. Splice onto **v8** if GO.\n")
    (RES / "E48_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE " + str(RES / "E48_results.json"))


if __name__ == "__main__":
    main()
