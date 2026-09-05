"""E13: LIRF unmatched regimes. Training data only. Do not restart E0–E12."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    airport_mean_fallback,
    fill_with_fallback,
    load_dep,
    metrics_block,
    rmse,
    mae,
    save_result,
    split_by_months,
)
from run_e2b_e3 import attach_geometry, geometry_tables, per_airport_ols


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def slice_rmse(y, p, mask, name, fh):
    m = mask & np.isfinite(y) & np.isfinite(p)
    n = int(m.sum())
    if n == 0:
        log(fh, f"  {name:42s} n=0")
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    r = rmse(y[m], p[m])
    a = mae(y[m], p[m])
    log(fh, f"  {name:42s} n={n:6d} RMSE={r:8.2f} MAE={a:7.2f} mean_y={y[m].mean():7.1f} med_y={np.median(y[m]):7.1f}")
    return {"n": n, "rmse": r, "mae": a}


def apply_lirf_rule(p_cal, mvt_sched, lirf_um, use_sched_mask):
    out = p_cal.copy()
    sel = lirf_um & use_sched_mask & np.isfinite(mvt_sched)
    out[sel] = mvt_sched[sel]
    return out


def main():
    out = Path(__file__).resolve().parent / "results" / "E13.txt"
    with open(out, "w", encoding="utf-8") as fh:
        log(fh, "========== E13 LIRF UNMATCHED REGIMES ==========")
        log(fh, "Training files only. No ranking/submitting.")
        dep = load_dep().with_columns(
            pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("type_null"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("p3"),
            pl.col("STAND_mvt").str.slice(0, 1).alias("stand_prefix"),
        )

        # Full-year description (training DEP only) — not used to pick val thresholds.
        lirf_u = dep.filter((pl.col("airport") == "LIRF") & pl.col("unmatched"))
        log(fh, f"\nTraining LIRF unmatched n={lirf_u.height:,} type_null={int(lirf_u['type_null'].sum())}")
        y_all = lirf_u["y"].to_numpy()
        ms_all = lirf_u["mvt_sched"].to_numpy().astype(float)
        log(fh, f"corr(y, mvt_sched) full-year LIRF unmatched={np.corrcoef(y_all[np.isfinite(ms_all)], ms_all[np.isfinite(ms_all)])[0,1]:.4f}")
        log(fh, "\n--- y bins vs mvt_sched (full-year LIRF unmatched) ---")
        bins = [0, 900, 1800, 3600, 7200, 1e12]
        labels = ["<15m", "15-30m", "30-60m", "1-2h", ">2h"]
        for i, lab in enumerate(labels):
            m = (y_all >= bins[i]) & (y_all < bins[i + 1]) & np.isfinite(ms_all)
            if not m.any():
                continue
            err = y_all[m] - ms_all[m]
            log(
                fh,
                f"  {lab:8s} n={m.sum():4d} mean_y={y_all[m].mean():7.0f} med_y={np.median(y_all[m]):7.0f} "
                f"mean_ms={ms_all[m].mean():7.0f} med_ms={np.median(ms_all[m]):7.0f} "
                f"RMSE(ms)={np.sqrt(np.mean(err**2)):7.0f} MAE(ms)={np.mean(np.abs(err)):6.0f}",
            )

        # Feature association with oracle extreme y>1800 / y>3600 (description)
        log(fh, "\n--- prediction-time features vs oracle y>1800 (full-year LIRF unmatched) ---")
        ext = lirf_u.with_columns((pl.col("y") > 1800).alias("ext30"), (pl.col("y") > 3600).alias("ext1h"))
        log(fh, "hour P(y>1800)", ext.group_by("hour").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("hour"))
        log(fh, "stand_prefix", ext.group_by("stand_prefix").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("n", descending=True).head(12))
        log(fh, "runway", ext.group_by("RUNWAY_mvt").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("n", descending=True))
        log(fh, "top ADES", ext.group_by("ADES_mvt").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("n", descending=True).head(12))
        log(fh, "top p3", ext.group_by("p3").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("n", descending=True).head(12))
        log(fh, "dow", ext.group_by("dow").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("dow"))
        log(fh, "month", ext.group_by("month").agg(pl.len().alias("n"), pl.col("ext30").mean().alias("p30"), pl.col("y").median().alias("med")).sort("month"))

        payload = {}
        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr0, va0 = split_by_months(dep, months)
            tabs = geometry_tables(tr0)
            tr = attach_geometry(tr0, tabs, 30)
            va = attach_geometry(va0, tabs, 30)
            y = va["y"].to_numpy()
            um = va["unmatched"].to_numpy()
            ap = va["airport"].to_numpy()
            type_null = va["type_null"].to_numpy()
            fb = airport_mean_fallback(tr, va)
            geo = va["geo_mean"].to_numpy().astype(float)
            mvt_sched = va["mvt_sched"].to_numpy().astype(float)
            p_ap, _, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])
            p_cal = fill_with_fallback(fill_with_fallback(p_ap, geo), fb)

            lirf_um = (ap == "LIRF") & um
            # type_null at LIRF is almost the same set
            slice_tn = (ap == "LIRF") & type_null
            log(fh, f"\n========== {split_name} ==========")
            log(fh, f"val n={len(y):,} LIRF unmatched={lirf_um.sum()} LIRF type_null={slice_tn.sum()} overlap={int((lirf_um & slice_tn).sum())}")

            oracle_n = lirf_um & (y <= 1800)
            oracle_x = lirf_um & (y > 1800)
            oracle_1h = lirf_um & (y > 3600)
            log(fh, f"oracle LIRF unmatched y<=30m n={oracle_n.sum()} y>30m n={oracle_x.sum()} y>1h n={oracle_1h.sum()}")

            block = {}
            # current rule: all LIRF unmatched -> mvt_sched
            always = apply_lirf_rule(p_cal, mvt_sched, lirf_um, np.ones_like(lirf_um, dtype=bool))
            never = p_cal.copy()  # no override
            log(fh, "\n--- overall vs current E3+always MVT-SCHED ---")
            for name, pred in [("E3 no LIRF override", never), ("CURRENT always MVT-SCHED", always)]:
                m = metrics_block(y, pred, um, ap)
                log(fh, f"{name:32s} n={m['n']:,} overall={m['rmse']:.2f} MAE={m['mae']:.2f} matched={m['rmse_matched']:.2f} unmatched={m['rmse_unmatched']:.2f} LIRF={m['rmse_LIRF']:.2f}")
                block[name] = m
                slice_rmse(y, pred, lirf_um, name + " | LIRF unmatched", fh)
                slice_rmse(y, pred, oracle_n, name + " | oracle normal <=30m", fh)
                slice_rmse(y, pred, oracle_x, name + " | oracle extreme >30m", fh)
                slice_rmse(y, pred, oracle_1h, name + " | oracle >1h", fh)

            # Does MVT-SCHED hurt the oracle-normal half?
            log(fh, "\n--- oracle-normal LIRF unmatched: MVT-SCHED vs P_cal vs geo ---")
            slice_rmse(y, mvt_sched, oracle_n, "MVT-SCHED on oracle-normal", fh)
            slice_rmse(y, p_cal, oracle_n, "P_cal on oracle-normal", fh)
            slice_rmse(y, geo, oracle_n, "geo_mean on oracle-normal", fh)
            slice_rmse(y, mvt_sched, oracle_x, "MVT-SCHED on oracle-extreme>30m", fh)
            slice_rmse(y, p_cal, oracle_x, "P_cal on oracle-extreme>30m", fh)

            # Thresholds on MVT-SCHED, chosen using TRAIN LIRF unmatched only
            tr_y = tr["y"].to_numpy()
            tr_um = tr["unmatched"].to_numpy()
            tr_ap = tr["airport"].to_numpy()
            tr_ms = tr["mvt_sched"].to_numpy().astype(float)
            tr_geo = tr["geo_mean"].to_numpy().astype(float)
            p_tr, _, _, _ = per_airport_ols(tr, tr, ["mvt_aobt", "aobt_eobt", "geo_mean"])
            p_tr = fill_with_fallback(fill_with_fallback(p_tr, tr_geo), airport_mean_fallback(tr, tr))
            tr_lu = (tr_ap == "LIRF") & tr_um & np.isfinite(tr_ms)

            log(fh, "\n--- train-only threshold search on MVT-SCHED (LIRF unmatched RMSE) ---")
            best_t, best_r = None, 1e18
            rows = []
            for t in [900, 1200, 1500, 1800, 2400, 2700, 3600, 5400, 7200]:
                pred_tr = p_tr.copy()
                use = tr_lu & (tr_ms > t)
                pred_tr[use] = tr_ms[use]
                # also the complement of tr_lu keeps p_tr; LIRF unmatched with ms<=t keep p_tr
                r_lu = rmse(tr_y[tr_lu], pred_tr[tr_lu])
                r_n = rmse(tr_y[tr_lu & (tr_y <= 1800)], pred_tr[tr_lu & (tr_y <= 1800)])
                r_x = rmse(tr_y[tr_lu & (tr_y > 1800)], pred_tr[tr_lu & (tr_y > 1800)])
                n_pred_x = int((tr_lu & (tr_ms > t)).sum())
                log(fh, f"  T>{t:5d}s  pred_extreme={n_pred_x:4d}/{int(tr_lu.sum())}  LIRF_u_RMSE={r_lu:8.1f}  oracle-normal RMSE={r_n:7.1f}  oracle-ext RMSE={r_x:8.1f}")
                rows.append((t, r_lu, n_pred_x))
                if r_lu < best_r:
                    best_r, best_t = r_lu, t
            log(fh, f"best train LIRF unmatched RMSE threshold T={best_t}s ({best_r:.1f})")

            # also always-sched on train for reference
            pred_always_tr = p_tr.copy()
            pred_always_tr[tr_lu] = tr_ms[tr_lu]
            log(fh, f"train always MVT-SCHED LIRF_u_RMSE={rmse(tr_y[tr_lu], pred_always_tr[tr_lu]):.1f}")

            log(fh, "\n--- val: conditional MVT-SCHED if mvt_sched > T else P_cal ---")
            for t in [900, 1800, 2700, 3600, best_t]:
                use = np.isfinite(mvt_sched) & (mvt_sched > t)
                pred = apply_lirf_rule(p_cal, mvt_sched, lirf_um, use)
                m = metrics_block(y, pred, um, ap)
                n_ex = int((lirf_um & use).sum())
                log(
                    fh,
                    f"T>{t} pred_ext={n_ex}/{int(lirf_um.sum())} overall={m['rmse']:.2f} MAE={m['mae']:.2f} "
                    f"matched={m['rmse_matched']:.2f} unmatched={m['rmse_unmatched']:.2f} LIRF={m['rmse_LIRF']:.2f}",
                )
                slice_rmse(y, pred, lirf_um, f"T>{t} LIRF unmatched", fh)
                slice_rmse(y, pred, oracle_n, f"T>{t} oracle-normal", fh)
                slice_rmse(y, pred, oracle_x, f"T>{t} oracle-extreme>30m", fh)
                # predicted-regime RMSE
                slice_rmse(y, pred, lirf_um & use, f"T>{t} PRED extreme (ms>T)", fh)
                slice_rmse(y, pred, lirf_um & ~use, f"T>{t} PRED normal (ms<=T)", fh)
                block[f"T>{t}"] = {
                    "overall": m["rmse"],
                    "matched": m["rmse_matched"],
                    "unmatched": m["rmse_unmatched"],
                    "lirf": m["rmse_LIRF"],
                    "n_pred_extreme": n_ex,
                }

            # Simple extra rules from train associations, evaluated on val
            log(fh, "\n--- other prediction-time flags (val) ---")
            stand = va["STAND_mvt"].fill_null("").to_numpy()
            ades = va["ADES_mvt"].fill_null("").to_numpy()
            p3 = va["p3"].fill_null("").to_numpy()
            hour = va["hour"].to_numpy()
            # stand 8xx
            stand8 = np.array([str(s).startswith("8") for s in stand])
            llbg = ades == "LLBG"
            night = (hour <= 4) | (hour >= 20)
            prefixes = {"NOS", "ISR", "CHH", "BAW", "AMX", "EXS", "KMM", "TWB", "TKJ"}
            weird_p = np.array([str(x) in prefixes for x in p3])

            for lab, flag in [
                ("stand8xx", stand8),
                ("ADES=LLBG", llbg),
                ("hour<=4 or >=20", night),
                ("weird prefix NOS/ISR/...", weird_p),
                ("ms>1800 OR stand8", stand8 | (np.isfinite(mvt_sched) & (mvt_sched > 1800))),
                ("ms>1800 OR LLBG", llbg | (np.isfinite(mvt_sched) & (mvt_sched > 1800))),
            ]:
                pred = apply_lirf_rule(p_cal, mvt_sched, lirf_um, flag)
                m = metrics_block(y, pred, um, ap)
                n_ex = int((lirf_um & flag).sum())
                log(
                    fh,
                    f"{lab:28s} pred_ext={n_ex:4d} overall={m['rmse']:.2f} unmatched={m['rmse_unmatched']:.2f} LIRF={m['rmse_LIRF']:.2f}",
                )
                slice_rmse(y, pred, oracle_n, lab + " oracle-normal", fh)
                slice_rmse(y, pred, oracle_x, lab + " oracle-extreme", fh)

            # Confusion: can mvt_sched>1800 recover oracle y>1800?
            pred_x = lirf_um & np.isfinite(mvt_sched) & (mvt_sched > 1800)
            tp = int((pred_x & oracle_x).sum())
            fp = int((pred_x & oracle_n).sum())
            fn = int((~pred_x & oracle_x).sum())
            tn = int((~pred_x & oracle_n).sum())
            log(fh, f"\nConfusion mvt_sched>1800 vs oracle y>1800 on LIRF unmatched: TP={tp} FP={fp} FN={fn} TN={tn}")
            if tp + fp:
                log(fh, f"  precision={tp/(tp+fp):.3f} recall={tp/(tp+fn) if tp+fn else float('nan'):.3f}")

            pred_x2 = lirf_um & np.isfinite(mvt_sched) & (mvt_sched > 3600)
            tp2 = int((pred_x2 & oracle_1h).sum())
            fp2 = int((pred_x2 & ~oracle_1h & lirf_um).sum())
            fn2 = int((~pred_x2 & oracle_1h).sum())
            log(fh, f"Confusion mvt_sched>3600 vs oracle y>1h: TP={tp2} FP={fp2} FN={fn2}")

            payload[split_name] = block

        save_result("E13", payload)
    print("WROTE", out)


if __name__ == "__main__":
    main()
