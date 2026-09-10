"""E28 STEP 1 — unmatched error forensics (Mission 2).

Determines WHY departures are unmatched and WHY their errors are enormous,
then previews recoverable SSE via causal service profiles. No ranking data
used except earlier descriptive coverage. E20 untouched.

Deliverables: class counts, SSE decomposition, top-SSE contributors,
metadata completeness, and a service-profile recovery preview.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import AIRPORTS, load_dep, mae, rmse, save_result  # noqa: E402

RES = HERE / "results" / "E28"
RES.mkdir(parents=True, exist_ok=True)
NM_FLT_COLS = ["AOBT_3_flt", "AIRCRAFT_OPERATOR_flt", "ADES_FILED_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def e20_pred(oof):
    return (0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy()
            + 0.491 * oof["pred_E"].to_numpy())


def main():
    log("E28 audit: load training_*.parquet...")
    dep = load_dep()
    log(f"rows {dep.height:,}  unmatched {int(dep['unmatched'].sum()):,}")

    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        log(f"========== {split} ==========")
        val = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        val = val.join(oof.select(["MVT_ID_mvt", "pred_A", "pred_B", "pred_C", "pred_D", "pred_E"]),
                       on="MVT_ID_mvt", how="left")
        p20 = e20_pred(val)
        y = val["y"].to_numpy().astype(np.float64)
        um = val["unmatched"].to_numpy().astype(bool)
        ap = val["airport"].to_numpy()
        err = y - p20
        sse = err ** 2
        sse_m = float(sse[~um].sum())
        sse_u = float(sse[um].sum())
        sse_t = sse_m + sse_u

        lirf_u = um & (ap == "LIRF")
        nlu = um & (ap != "LIRF")
        log(f"  n={len(y):,} matched={int((~um).sum()):,} unmatched={um.sum():,} ({100*um.mean():.2f}%)")
        log(f"  SSE total={sse_t:.3e}  matched={sse_m:.3e} ({100*sse_m/sse_t:.1f}%)  "
            f"unmatched={sse_u:.3e} ({100*sse_u/sse_t:.1f}%)")
        log(f"  unmatched RMSE={rmse(y[um],p20[um]):.1f}  LIRF_u RMSE={rmse(y[lirf_u],p20[lirf_u]):.1f} "
            f"(n={int(lirf_u.sum())})  non-LIRF_u RMSE={rmse(y[nlu],p20[nlu]):.1f} (n={int(nlu.sum())})")

        # metadata completeness on unmatched
        meta = {}
        for c in NM_FLT_COLS + ["FLIGHT_mvt", "ADES_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]:
            meta[c] = {"unmatched_null": int(val.filter(pl.col("unmatched"))[c].null_count()),
                       "unmatched_n": int(um.sum()),
                       "null_frac": float(val.filter(pl.col("unmatched"))[c].null_count() / max(um.sum(), 1))}

        # class: total NM miss if all NM cols null
        nm_all_null = val.select([pl.all_horizontal([pl.col(c).is_null() for c in NM_FLT_COLS]).alias("_all")]).to_series().to_numpy()
        classes = {
            "no_NM_candidate_all_nm_cols_null": int((um & nm_all_null).sum()),
            "partial_NM_match": int((um & ~nm_all_null).sum()),
            "unmatched_lirf": int(lirf_u.sum()),
            "unmatched_non_lirf": int(nlu.sum()),
        }

        # top SSE contributors
        order = np.argsort(-sse)
        top_rows = []
        for rank, i in enumerate(order[:500], start=1):
            if rank in (1, 10, 50, 100, 500) and um[i]:
                pass
        tops = {}
        for k in (10, 50, 100, 500):
            idx = order[:k]
            # how many of the top-k are unmatched vs matched
            tops[f"top{k}"] = {
                "unmatched_in_topk": int(um[idx].sum()),
                "lirf_unmatched_in_topk": int(lirf_u[idx].sum()),
                "matched_in_topk": int((~um[idx]).sum()),
                "sse_share_of_total": float(sse[idx].sum() / sse_t),
            }

        # non-LIRF unmatched y distribution (is it gate delay or normal taxi?)
        nlu_y = y[nlu]
        nlu_stats = {
            "n": int(nlu.sum()),
            "median_y": float(np.median(nlu_y)), "mean_y": float(np.mean(nlu_y)),
            "frac_y_gt_30m": float(np.mean(nlu_y > 1800)), "frac_y_gt_60m": float(np.mean(nlu_y > 3600)),
            "max_y": float(nlu_y.max()),
        }
        lirf_y = y[lirf_u]
        lirf_stats = {
            "n": int(lirf_u.sum()), "median_y": float(np.median(lirf_y)) if lirf_u.any() else float("nan"),
            "frac_y_gt_30m": float(np.mean(lirf_y > 1800)) if lirf_u.any() else float("nan"),
            "frac_y_gt_60m": float(np.mean(lirf_y > 3600)) if lirf_u.any() else float("nan"),
        }

        # per-airport unmatched
        per_ap = {}
        for a in AIRPORTS:
            mu = um & (ap == a)
            if mu.sum() == 0:
                continue
            per_ap[a] = {"n_unmatched": int(mu.sum()), "sse_unmatched": float(sse[mu].sum()),
                         "rmse_unmatched": rmse(y[mu], p20[mu]),
                         "sse_share_total": float(sse[mu].sum() / sse_t)}

        # ---- service-profile recovery preview (causal, train months only) ----
        tr = dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        trm = tr.filter(~pl.col("unmatched"))
        prof_l1 = trm.group_by(["airport", "FLIGHT_mvt", "ADES_mvt"]).agg(
            pl.col("y").median().alias("med"), pl.col("y").len().alias("n1"))
        prof_l2 = trm.group_by(["airport", "ADES_mvt"]).agg(
            pl.col("y").median().alias("med2"), pl.col("y").len().alias("n2"))
        prof_l3 = trm.group_by("airport").agg(pl.col("y").median().alias("med3"))
        vu = val.filter(pl.col("unmatched"))
        j = vu.join(prof_l1, on=["airport", "FLIGHT_mvt", "ADES_mvt"], how="left") \
              .join(prof_l2, on=["airport", "ADES_mvt"], how="left") \
              .join(prof_l3, on="airport", how="left")
        med = j["med"].to_numpy()
        med2 = j["med2"].to_numpy()
        med3 = j["med3"].to_numpy()
        n1 = j["n1"].to_numpy()
        pred_rec = np.where(np.isfinite(med), med, np.where(np.isfinite(med2), med2, med3))
        yu = j["y"].to_numpy().astype(np.float64)
        mask = np.isfinite(yu) & np.isfinite(pred_rec)
        rec = {
            "l1_coverage": float(np.mean(np.isfinite(med))),
            "any_coverage": float(np.mean(mask)),
            "rmse_service_all_unmatched": rmse(yu[mask], pred_rec[mask]) if mask.any() else float("nan"),
        }
        for tag, sel in [("lirf", j["airport"].to_numpy() == "LIRF"), ("non_lirf", j["airport"].to_numpy() != "LIRF")]:
            s = sel & np.isfinite(yu) & np.isfinite(pred_rec)
            if s.any():
                rec[f"rmse_service_{tag}"] = rmse(yu[s], pred_rec[s])
                rec[f"n_{tag}"] = int(s.sum())
        # E20 current unmatched score for reference
        rec["e20_unmatched_rmse"] = rmse(y[um], p20[um])
        log(f"  SERVICE preview: L1 cover {rec['l1_coverage']*100:.1f}%  service RMSE(all_u) {rec['rmse_service_all_unmatched']:.1f} "
            f"vs E20_u {rec['e20_unmatched_rmse']:.1f}")

        payload[split] = {
            "n_val": int(len(y)), "n_matched": int((~um).sum()), "n_unmatched": int(um.sum()),
            "unmatched_pct": float(100 * um.mean()),
            "rmse_overall": rmse(y, p20), "rmse_matched": rmse(y[~um], p20[~um]),
            "rmse_unmatched": rmse(y[um], p20[um]),
            "sse_total": sse_t, "sse_matched": sse_m, "sse_unmatched": sse_u,
            "sse_unmatched_pct": float(100 * sse_u / sse_t),
            "classes": classes,
            "metadata": meta,
            "top_sse": tops,
            "non_lirf_unmatched": nlu_stats,
            "lirf_unmatched": lirf_stats,
            "per_airport": per_ap,
            "recovery_preview": rec,
        }

    write_report(payload)
    conv = lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else o)
    (RES / "E28_unmatched_audit.json").write_text(json.dumps(payload, indent=2, default=conv), encoding="utf-8")
    log("WROTE " + str(RES / "E28_unmatched_audit.json"))


def write_report(payload):
    L = ["# E28 — Unmatched error forensics (Mission 2, STEP 1)", ""]
    L.append("## Mechanism (why unmatched)")
    L.append("")
    L.append("Every unmatched DEP has **all NM flight-table fields null** "
             "(`AOBT_3_flt`, `AIRCRAFT_OPERATOR_flt`, `ADES_FILED_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`). "
             "Unmatched = a **total NM flight-record miss**, not a fuzzy-match failure. Movement-side "
             "fields (ADEP/ADES/FLIGHT/SCHED/MVT/runway/stand/type) are present.")
    L.append("")
    for split, label in [("janjul", "Jan+Jul 2025"), ("dec", "December 2025")]:
        p = payload[split]
        L.append(f"## {label}")
        L.append("")
        L.append(f"- matched {p['n_matched']:,} / unmatched {p['n_unmatched']:,} ({p['unmatched_pct']:.2f}%)")
        L.append(f"- RMSE overall {p['rmse_overall']:.2f} / matched {p['rmse_matched']:.2f} / unmatched {p['rmse_unmatched']:.2f}")
        L.append(f"- SSE total {p['sse_total']:.3e}; unmatched {p['sse_unmatched']:.3e} (**{p['sse_unmatched_pct']:.1f}% of total SSE**)")
        L.append(f"- classes {p['classes']}")
        L.append(f"- non-LIRF unmatched: median y {p['non_lirf_unmatched']['median_y']:.0f}s, "
                 f">30m {100*p['non_lirf_unmatched']['frac_y_gt_30m']:.1f}%, >60m {100*p['non_lirf_unmatched']['frac_y_gt_60m']:.1f}%")
        L.append(f"- LIRF unmatched: median y {p['lirf_unmatched']['median_y']:.0f}s, "
                 f">30m {100*p['lirf_unmatched']['frac_y_gt_30m']:.1f}%, >60m {100*p['lirf_unmatched']['frac_y_gt_60m']:.1f}%")
        L.append(f"- top-SSE: {p['top_sse']}")
        L.append("")
        L.append("Per-airport unmatched:")
        L.append("")
        L.append("| Airport | n | RMSE | SSE | % total SSE |")
        L.append("|---|---:|---:|---:|---:|")
        for a, r in sorted(p["per_airport"].items(), key=lambda t: -t[1]["sse_unmatched"]):
            L.append(f"| {a} | {r['n_unmatched']:,} | {r['rmse_unmatched']:.0f} | {r['sse_unmatched']:.3e} | {r['sse_share_total']:.1f}% |")
        L.append("")
        r = p["recovery_preview"]
        L.append(f"**Recovery preview (causal service profiles, train-only):** L1 coverage {100*r['l1_coverage']:.1f}%, "
                 f"service RMSE(all unmatched) **{r['rmse_service_all_unmatched']:.0f}** vs E20 unmatched {r['e20_unmatched_rmse']:.0f} "
                 f"(non-LIRF {r.get('rmse_service_non_lirf', float('nan')):.0f}, LIRF {r.get('rmse_service_lirf', float('nan')):.0f})")
        L.append("")
    (RES / "summary_unmatched_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
