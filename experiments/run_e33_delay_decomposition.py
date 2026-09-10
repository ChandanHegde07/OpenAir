"""E33 — latent delay decomposition.

Identity: T = D - G, with D = MVT-SCHED, T = TAXITIME = MVT-BLOCK,
G = BLOCK-SCHED. MVT-SCHED is observed; learn G (and ratio forms) from
matched history, reconstruct T on unmatched.

Variants: direct T, direct G, gate-ratio R=G/D (T=D*(1-R)), taxi-fraction
Q=T/D (T=D*Q). Surface-state features reused. E20 kept as blend component.
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
from common import AIRPORTS, TRAIN_FILES, add_causal_rolling, load_dep, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E33"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
CAT = ["airport", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "flt_prefix"]
SURF = ["sdep_5", "sdep_15", "sdep_30", "sarr_5", "sarr_15", "sarr_30",
        "srdep_10", "srdep_30", "srarr_10", "srarr_30", "sdep_ts", "sarr_ts", "sev_ts", "sst_ts"]
NUM = ["mvt_sched", "hour", "dow", "month"] + SURF


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _cnt(times, q, w):
    return (np.searchsorted(times, q, side="left") - np.searchsorted(times, q - w * 60 * 10**9, side="left")).astype(np.float64)


def _since(times, q):
    i = np.searchsorted(times, q, side="left") - 1
    o = np.full(len(q), np.nan); m = i >= 0
    o[m] = (q[m] - times[i[m]]) / 1e9
    return o


def add_surface(dep, arr):
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    apn = dep["airport"].to_numpy()
    rwn = dep["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    stn = dep["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    amv = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    aap = arr["airport"].to_numpy()
    arw = arr["RUNWAY_mvt"].fill_null("").cast(pl.String).to_numpy()
    n = dep.height
    cols = {c: np.full(n, np.nan) for c in SURF}
    for a in AIRPORTS:
        sel = np.flatnonzero(apn == a)
        if sel.size == 0:
            continue
        t = mvt[sel]; at = amv[aap == a]
        for w in (5, 15, 30):
            cols[f"sdep_{w}"][sel] = _cnt(t, t, w)
            cols[f"sarr_{w}"][sel] = _cnt(at, t, w) if at.size else 0.0
        cols["sdep_ts"][sel] = _since(t, t)
        cols["sarr_ts"][sel] = _since(at, t) if at.size else np.nan
        ev = np.sort(np.concatenate([t, at])) if at.size else t
        cols["sev_ts"][sel] = _since(ev, t)
        rw = rwn[sel]
        for r in np.unique(rw):
            m = rw == r; t_r = t[m]
            for w in (10, 30):
                cols[f"srdep_{w}"][sel] = np.where(m, _cnt(t_r, t, w), cols[f"srdep_{w}"][sel])
            if at.size:
                ar = at[(arw[aap == a]) == r]
                for w in (10, 30):
                    cols[f"srarr_{w}"][sel] = np.where(m, _cnt(ar, t, w) if ar.size else 0.0, cols[f"srarr_{w}"][sel])
        st = stn[sel]
        for s in np.unique(st):
            m = st == s
            cols["sst_ts"][sel] = np.where(m, _since(t[m], t), cols["sst_ts"][sel])
    return dep.with_columns([pl.Series(c, cols[c]) for c in SURF])


def main():
    log("E33: build...")
    dep = add_causal_rolling(load_dep())
    dep = dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix"))
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    dep = add_surface(dep, arr)
    log(f"rows {dep.height:,}")

    def pdf(df, cols):
        p = df.select(cols + CAT).to_pandas()
        for c in CAT:
            p[c] = p[c].astype("category")
        return p

    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E", "unmatched"]).rename({"unmatched": "_um"}), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        yv = va["y"].to_numpy().astype(float)
        Dv = va["mvt_sched"].to_numpy().astype(float)
        umv = va["_um"].to_numpy().astype(bool)
        apv = va["airport"].to_numpy()

        def train_eval(target_kind, tr_mask, va_mask):
            trm = tr.filter(pl.Series(tr_mask)).with_columns(pl.Series("D", tr.filter(pl.Series(tr_mask))["mvt_sched"].to_numpy().astype(float)))
            vam = va.filter(pl.Series(va_mask))
            if trm.height < 150:
                return None
            D = trm["D"].to_numpy(); y = trm["y"].to_numpy().astype(float)
            Dc = np.maximum(D, 60.0)
            if target_kind == "T":
                tgt = y
            elif target_kind == "G":
                tgt = D - y
            elif target_kind == "R":
                tgt = np.clip((D - y) / Dc, 0, 1)
            elif target_kind == "Q":
                tgt = np.clip(y / Dc, 0, 1)
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=30,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(pdf(trm, NUM), tgt, categorical_feature=CAT)
            raw = np.asarray(m.predict(pdf(vam, NUM)), dtype=float)
            Dq = vam["mvt_sched"].to_numpy().astype(float)
            Dqc = np.maximum(Dq, 60.0)
            if target_kind == "T":
                pred = raw
            elif target_kind == "G":
                pred = Dq - raw
            elif target_kind == "R":
                pred = Dq * (1 - np.clip(raw, 0, 1))
            elif target_kind == "Q":
                pred = Dq * np.clip(raw, 0, 1)
            return pred

        tr_um = tr["unmatched"].to_numpy().astype(bool)
        tr_ap = tr["airport"].to_numpy()
        lirf_tr = tr_um & (tr_ap == "LIRF")
        lirf_va = umv & (apv == "LIRF")

        res = {"E20_lirf": rmse(yv[lirf_va], e20[lirf_va]), "E20_um": rmse(yv[umv], e20[umv]),
               "E20_all": rmse(yv, e20)}
        log(f"  [{split}] E20 overall {res['E20_all']:.1f} unmatched {res['E20_um']:.1f} LIRF_u {res['E20_lirf']:.0f}")
        variants = {}
        for kind in ["T", "G", "R", "Q"]:
            pred = train_eval(kind, lirf_tr, lirf_va)
            if pred is None:
                continue
            # alpha blend with E20 on LIRF unmatched
            best = None
            for al in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
                p = al * pred + (1 - al) * e20[lirf_va]
                s = float(np.sum((yv[lirf_va] - p) ** 2))
                if best is None or s < best[1]:
                    best = (al, s, rmse(yv[lirf_va], p))
            variants[kind] = {"lirf_rmse": rmse(yv[lirf_va], pred), "best_alpha": best[0],
                              "lirf_blend_rmse": best[2], "lirf_sse": best[1]}
            log(f"    {kind}: LIRF standalone {variants[kind]['lirf_rmse']:.0f} best_alpha {best[0]} blend {best[2]:.0f}")
        # also unmatched-wide Q model
        predQ_um = train_eval("Q", tr_um, umv)
        res["variants_lirf"] = variants
        if predQ_um is not None:
            res["Q_unmatched_rmse"] = rmse(yv[umv], predQ_um)
        payload[split] = res
    (RES / "E33_decomposition.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    save_result("E33", payload)
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
