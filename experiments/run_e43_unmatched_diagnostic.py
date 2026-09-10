"""E43 STEP 1 — per-airport unmatched diagnostic (no model code).

Uses current production (E34/v9) baseline for unmatched rows: for non-LIRF
unmatched, v9 == E20 (E20 OOF blend); LIRF unmatched is the E34 correction and
is reported separately. Jan+Jul primary, Dec sanity. Also ranking unmatched
rates per airport.
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
from common import AIRPORTS, rmse  # noqa: E402
from run_submitting_check import RANK_PATH  # noqa: E402

RES = HERE / "results" / "E43"
RES.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    payload = {}
    for split in ("janjul", "dec"):
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        e20 = 0.456 * oof["pred_C"].to_numpy() + 0.053 * oof["pred_D"].to_numpy() + 0.491 * oof["pred_E"].to_numpy()
        y = oof["y"].to_numpy().astype(float); um = oof["unmatched"].to_numpy().astype(bool); ap = oof["airport"].to_numpy()
        ok = np.isfinite(y) & np.isfinite(e20)
        sse_all = np.sum((y[ok] - e20[ok]) ** 2)
        rows = {}
        for a in AIRPORTS:
            ma = ok & (ap == a)
            mm = ma & ~um
            mu = ma & um
            sse_ap = np.sum((y[ma] - e20[ma]) ** 2)
            sse_u = np.sum((y[mu] - e20[mu]) ** 2) if mu.any() else 0.0
            rows[a] = {
                "matched_n": int(mm.sum()), "unmatched_n": int(mu.sum()),
                "matched_rmse": float(rmse(y[mm], e20[mm])) if mm.any() else float("nan"),
                "unmatched_rmse": float(rmse(y[mu], e20[mu])) if mu.any() else float("nan"),
                "unmatched_sse": float(sse_u),
                "unmatched_pct_of_airport_sse": float(100 * sse_u / sse_ap) if sse_ap else 0.0,
                "unmatched_pct_of_dataset_sse": float(100 * sse_u / sse_all) if sse_all else 0.0,
                "rmse_ratio_unmatched_over_matched": (float(rmse(y[mu], e20[mu])) / float(rmse(y[mm], e20[mm]))) if (mu.any() and mm.any()) else float("nan"),
            }
        payload[split] = {"sse_all": float(sse_all), "airports": rows}
        log(f"== {split} == (baseline = E20/v9 on non-LIRF unmatched; LIRF shown as E20 override)")
        for a in AIRPORTS:
            r = rows[a]
            log(f"   {a}: matched {r['matched_rmse']:6.1f} | unmatched n={r['unmatched_n']:>5} rmse={r['unmatched_rmse']:7.1f} "
                f"ratio={r['rmse_ratio_unmatched_over_matched']:5.1f} | {r['unmatched_pct_of_dataset_sse']:5.2f}% of dataset SSE")

    # ranking unmatched rate per airport
    rank = (pl.scan_parquet(RANK_PATH).filter(pl.col("PHASE_mvt") == "DEP")
            .select(["ADEP_mvt", "AOBT_3_flt"]).collect())
    rk = {}
    for a in AIRPORTS:
        s = rank.filter(pl.col("ADEP_mvt") == a)
        n = s.height; u = s["AOBT_3_flt"].null_count()
        rk[a] = {"n": int(n), "unmatched": int(u), "rate": float(100 * u / n) if n else 0.0}
    payload["ranking_unmatched_rate"] = rk
    log("== ranking unmatched rate ==")
    for a in AIRPORTS:
        log(f"   {a}: {rk[a]['unmatched']}/{rk[a]['n']} = {rk[a]['rate']:.2f}%")

    (RES / "E43_diagnostic.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_report(payload)
    log("WROTE " + str(RES))


def write_report(p):
    L = ["# E43 — Non-LIRF unmatched D−G opportunity (STEP 1 diagnostic)", ""]
    L.append("Baseline for non-LIRF unmatched = E20/v9 blend (E20 OOF); LIRF shown as E20 override "
             "(production v9 already applies the E34 D−G correction there).")
    L.append("")
    for split, title in [("janjul", "Jan+Jul 2025 (primary)"), ("dec", "December 2025 (sanity)")]:
        L.append(f"## {title}")
        L.append("")
        L.append("| Airport | matched RMSE | unmatched n | unmatched RMSE | ratio | % dataset SSE |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for a in AIRPORTS:
            r = p[split]["airports"][a]
            L.append(f"| {a} | {r['matched_rmse']:.1f} | {r['unmatched_n']} | {r['unmatched_rmse']:.1f} | "
                     f"{r['rmse_ratio_unmatched_over_matched']:.1f} | {r['unmatched_pct_of_dataset_sse']:.2f}% |")
        L.append("")
    L.append("## Ranking unmatched rate per airport")
    L.append("")
    L.append("| Airport | unmatched | n | rate |")
    L.append("|---|---:|---:|---:|")
    for a in AIRPORTS:
        r = p["ranking_unmatched_rate"][a]
        L.append(f"| {a} | {r['unmatched']} | {r['n']} | {r['rate']:.2f}% |")
    (RES / "E43_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
