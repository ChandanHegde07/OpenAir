"""E32 — causal surface-state reconstruction from the full movement event stream.

Builds airport/runway/stand/arrival-departure-interaction/sequence features
strictly from events before t (DEP+ARR), then trains direct surface models for
matched / unmatched / LIRF-unmatched and ensembles them with E20 via NNLS on
the validation OOF. If the candidate is expected to beat LB 300, writes
likable-eagle_v7.parquet.
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
from common import AIRPORTS, TRAIN_FILES, load_dep, mae, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E32"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
QW = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
RW = [5, 10, 15, 30]
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "flt_prefix", "seq5"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _cnt(times, q, w):
    return (np.searchsorted(times, q, side="left") - np.searchsorted(times, q - w * 60 * 10**9, side="left")).astype(np.float64)


def _since(times, q):
    idx = np.searchsorted(times, q, side="left") - 1
    out = np.full(len(q), np.nan)
    m = idx >= 0
    out[m] = (q[m] - times[idx[m]]) / 1e9
    return out


def build_surface(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    n = dep.height
    names = []
    # per-window airport/runway counts
    for w in QW:
        names += [f"dep_{w}", f"arr_{w}"]
    for w in RW:
        names += [f"rdep_{w}", f"rarr_{w}"]
    names += ["dep_ts", "arr_ts", "ev_ts", "arr_since_dep", "rdep_ts", "rarr_ts", "rmv_ts",
              "runway_dep_share", "runway_arr_share", "same_st_ts", "dep5_dep15", "dep15_dep30", "dep5_dep30",
              "arr5_arr30", "tot5_tot30", "seq5", "seq10", "nd_5", "na_5", "stand_dep_30", "stand_arr_30", "type_dep_60", "route_dep_60"]
    cols = {c: np.full(n, np.nan) for c in names}

    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    stand = dep["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    ades = dep["ADES_mvt"].fill_null("").cast(pl.String).to_numpy()
    atype = dep["AIRCRAFT_TYPE_mvt"].fill_null("").cast(pl.String).to_numpy()
    phase = dep["PHASE_mvt"].to_numpy()

    # arrival stream per airport (movements with PHASE ARR) from full dep frame? dep frame is DEP only.
    # We need arrivals: build from a separate call passing arrivals into this function via global.
    return dep, cols, names, (mvt, ap, rwy, stand, ades, atype, phase)


def main():
    log("E32: load DEP + ARR movement streams...")
    dep = load_dep()  # DEP only
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES)
           .filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"])
           .collect().sort(["airport", "MVT_TIME_UTC_mvt"]))
    log(f"DEP {dep.height:,} ARR {arr.height:,}")

    # Build features with a merged event-stream approach, per airport.
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    stand = dep["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    ades = dep["ADES_mvt"].fill_null("").cast(pl.String).to_numpy()
    atype = dep["AIRCRAFT_TYPE_mvt"].fill_null("").cast(pl.String).to_numpy()
    a_mvt = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    a_ap = arr["airport"].to_numpy()
    a_rwy = arr["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    a_stand = arr["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()

    n = dep.height
    names = []
    for w in QW:
        names += [f"dep_{w}", f"arr_{w}"]
    for w in RW:
        names += [f"rdep_{w}", f"rarr_{w}"]
    names += ["dep_ts", "arr_ts", "ev_ts", "arr_since_dep", "rdep_ts", "rarr_ts", "rmv_ts",
              "runway_dep_share", "runway_arr_share", "same_st_ts", "same_st_dep30", "same_st_arr30",
              "dep5_dep15", "dep15_dep30", "dep5_dep30", "arr5_arr30", "tot5_tot30", "arr_dep_ratio_10",
              "type_dep_60", "route_dep_60", "nd_5", "na_5"]
    cols = {c: np.full(n, np.nan) for c in names}

    for a in AIRPORTS:
        sel = np.flatnonzero(ap == a)
        if sel.size == 0:
            continue
        t = mvt[sel]
        a_t = a_mvt[a_ap == a]
        for w in QW:
            cols[f"dep_{w}"][sel] = _cnt(t, t, w)
            cols[f"arr_{w}"][sel] = _cnt(a_t, t, w) if a_t.size else 0.0
        cols["dep_ts"][sel] = _since(t, t)
        cols["arr_ts"][sel] = _since(a_t, t) if a_t.size else np.nan
        ev = np.sort(np.concatenate([t, a_t])) if a_t.size else t
        cols["ev_ts"][sel] = _since(ev, t)
        # arrivals since previous departure
        prev_dep = np.searchsorted(t, t, side="left") - 1
        hi = np.searchsorted(a_t, t, side="left")
        # arr in (prev_dep, t): use searchsorted on a_t
        pd_t = np.where(prev_dep >= 0, t[np.maximum(prev_dep, 0)], -10**18)
        cols["arr_since_dep"][sel] = (hi - np.searchsorted(a_t, pd_t, side="left")).astype(float) if a_t.size else 0.0
        # runway
        rw = rwy[sel]
        for r in np.unique(rw):
            m = rw == r
            t_r = t[m]
            for w in RW:
                cols[f"rdep_{w}"][sel] = np.where(m, _cnt(t_r, t, w), cols[f"rdep_{w}"][sel])
            cols["rdep_ts"][sel] = np.where(m, _since(t_r, t), cols["rdep_ts"][sel])
            if a_t.size:
                a_r = a_t[(a_rwy[a_ap == a]) == r]
                for w in RW:
                    cols[f"rarr_{w}"][sel] = np.where(m, _cnt(a_r, t, w) if a_r.size else 0.0, cols[f"rarr_{w}"][sel])
                cols["rarr_ts"][sel] = np.where(m, _since(a_r, t) if a_r.size else np.nan, cols["rarr_ts"][sel])
                evr = np.sort(np.concatenate([t_r, a_r])) if a_r.size else t_r
                cols["rmv_ts"][sel] = np.where(m, _since(evr, t), cols["rmv_ts"][sel])
        cols["runway_dep_share"][sel] = cols[f"rdep_10"][sel] / (cols["dep_10"][sel] + 1.0)
        cols["runway_arr_share"][sel] = cols[f"rarr_10"][sel] / (cols["arr_10"][sel] + 1.0)
        # stand
        st = stand[sel]
        for s in np.unique(st):
            m = st == s
            t_s = t[m]
            cols["same_st_ts"][sel] = np.where(m, _since(t_s, t), cols["same_st_ts"][sel])
            for w in (30,):
                cols["same_st_dep30"][sel] = np.where(m, _cnt(t_s, t, w), cols["same_st_dep30"][sel])
            if a_t.size:
                a_s = a_t[(a_stand[a_ap == a]) == s]
                cols["same_st_arr30"][sel] = np.where(m, _cnt(a_s, t, 30) if a_s.size else 0.0, cols["same_st_arr30"][sel])
        # temporal derivatives
        for k, expr in [("dep5_dep15", ("dep_5", "dep_15")), ("dep15_dep30", ("dep_15", "dep_30")),
                        ("arr5_arr30", ("arr_5", "arr_30"))]:
            cols[k][sel] = cols[expr[0]][sel] - cols[expr[1]][sel]
        cols["dep5_dep30"][sel] = cols["dep_5"][sel] / (cols["dep_30"][sel] + 1.0)
        cols["tot5_tot30"][sel] = (cols["dep_5"][sel] + cols["arr_5"][sel]) / (cols["dep_30"][sel] + cols["arr_30"][sel] + 1.0)
        cols["arr_dep_ratio_10"][sel] = cols["arr_10"][sel] / (cols["dep_10"][sel] + 1.0)
        # type / route activity
        for kk, key in [("type_dep_60", atype), ("route_dep_60", ades)]:
            v = key[sel]
            for u in np.unique(v):
                m = v == u
                t_u = t[m]
                cols[kk][sel] = np.where(m, _cnt(t_u, t, 60), cols[kk][sel])
        # event-sequence counts in last 5 events (deps vs arrs) via local array
        # approximate: use total event stream phase counts in last 5 minutes
        cols["nd_5"][sel] = cols["dep_5"][sel]
        cols["na_5"][sel] = cols["arr_5"][sel]

    dep = dep.with_columns([pl.Series(c, cols[c]) for c in names])
    # sequence signature from recent dep/arr counts
    dep = dep.with_columns(
        pl.when(pl.col("arr_5") > pl.col("dep_5")).then(pl.lit("Aheavy"))
        .when(pl.col("dep_5") > 0).then(pl.lit("D"))
        .otherwise(pl.lit("quiet")).alias("seq5"))

    feats = [c for c in names if c in dep.columns] + ["flt_prefix"]
    num = [c for c in feats if c not in CAT] + ["hour", "dow", "month", "mvt_sched"]
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_A", "pred_B", "pred_C", "pred_D", "pred_E", "unmatched"]).rename({"unmatched": "_um"}), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        yv = va["y"].to_numpy().astype(float)
        umv = va["_um"].to_numpy().astype(bool)
        apv = va["airport"].to_numpy()

        def pdf(df):
            p = df.select(num + CAT).to_pandas()
            for c in CAT:
                p[c] = p[c].astype("category")
            return p

        def fit_pred(tr_mask, va_mask, tag):
            trm = tr.filter(pl.Series(tr_mask))
            vam = va.filter(pl.Series(va_mask))
            if trm.height < 200:
                return None
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(pdf(trm), trm["y"].to_numpy().astype(float), categorical_feature=CAT)
            p = np.asarray(m.predict(pdf(vam)), dtype=float)
            return {"pred": p, "va_mask": va_mask, "n_tr": int(trm.height)}

        tr_um = tr["unmatched"].to_numpy().astype(bool)
        tr_ap = tr["airport"].to_numpy()
        res = {}
        # matched surface model
        r = fit_pred(~tr_um, ~umv, "match")
        if r:
            res["matched_surface"] = {"rmse": rmse(yv[~umv], r["pred"]), "n": int((~umv).sum())}
        # unmatched surface model
        r2 = fit_pred(tr_um, umv, "um")
        # LIRF unmatched
        lirf_tr = tr_um & (tr_ap == "LIRF")
        lirf_va = umv & (apv == "LIRF")
        r3 = fit_pred(lirf_tr, lirf_va, "lirf")

        def fill(base, r):
            out = base.copy()
            if r is not None:
                out[r["va_mask"]] = r["pred"]
            return out
        surf = np.full(len(yv), np.nan)
        if r:
            surf[~umv] = r["pred"]
        if r2:
            surf[umv] = r2["pred"]
        surf_lirf = surf.copy()
        if r3:
            surf_lirf[lirf_va] = r3["pred"]

        for tag, cand in [("E20", e20), ("surface", surf), ("surface_lirf", surf_lirf)]:
            good = np.isfinite(cand)
            res[tag] = {"overall": rmse(yv[good], cand[good]), "matched": rmse(yv[good & ~umv], cand[good & ~umv]),
                        "unmatched": rmse(yv[good & umv], cand[good & umv]),
                        "lirf_u": rmse(yv[good & lirf_va], cand[good & lirf_va]) if (good & lirf_va).any() else float("nan"),
                        "sse": float(np.nansum((yv[good] - cand[good]) ** 2))}
        # conditional NNLS with E20 as a component
        def blend(mask, cands):
            M = np.column_stack([c[mask] for c in cands])
            yy = yv[mask]
            ok = np.isfinite(yy) & np.isfinite(M).all(axis=1)
            if ok.sum() < 100:
                return None
            w, _ = nnls(M[ok], yy[ok])
            w = w / max(w.sum(), 1e-9)
            p = np.full(len(yv), np.nan)
            p[mask] = M @ w
            return {"w": [float(x) for x in w], "pred": p}
        final = e20.copy()
        for mask, cands in [(~umv, [e20, surf]), (umv & ~lirf_va, [e20, surf]),
                            (lirf_va, [e20, surf_lirf])]:
            b = blend(mask, cands)
            if b:
                final[mask] = b["pred"][mask]
        good = np.isfinite(final)
        res["ENSEMBLE"] = {"overall": rmse(yv[good], final[good]), "matched": rmse(yv[good & ~umv], final[good & ~umv]),
                           "unmatched": rmse(yv[good & umv], final[good & umv]),
                           "lirf_u": rmse(yv[good & lirf_va], final[good & lirf_va]),
                           "sse": float(np.sum((yv[good] - final[good]) ** 2))}
        payload[split] = res
        log(f"  [{split}] E20 overall {res['E20']['overall']:.1f} matched {res['E20']['matched']:.1f} unmatched {res['E20']['unmatched']:.1f} LIRF_u {res['E20']['lirf_u']:.0f}")
        log(f"  [{split}] surface overall {res['surface']['overall']:.1f} matched {res['surface']['matched']:.1f} unmatched {res['surface']['unmatched']:.1f}")
        log(f"  [{split}] ENSEMBLE overall {res['ENSEMBLE']['overall']:.1f} matched {res['ENSEMBLE']['matched']:.1f} unmatched {res['ENSEMBLE']['unmatched']:.1f} LIRF_u {res['ENSEMBLE']['lirf_u']:.0f}")

    (RES / "E32_surface.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    save_result("E32", payload)
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
