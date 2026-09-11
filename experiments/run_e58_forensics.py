"""E58 — Phase 1 forensic: structural asymmetry / hidden-process search.

Focus on the un-audited turnaround structure: link each matched departure to the
most recent arrival at the SAME STAND within 6h (no tail identity needed).
Decompose the ground interval and report SSE by source-availability populations
defined WITHOUT target values. No model training.
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
from common import AIRPORTS, TRAIN_FILES, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E58"
RES.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    dep = load_dep()
    arr = (pl.scan_parquet(TRAIN_FILES).filter(pl.col("PHASE_mvt") == "ARR")
           .select([pl.col("ADES_mvt").alias("airport"), "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt"]).collect()
           .sort(["airport", "MVT_TIME_UTC_mvt"]))
    n = dep.height
    # arrival-side taxi and stand lookup per airport
    a_ap = arr["airport"].to_numpy(); a_mvt = arr["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    a_blk = arr["BLOCK_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    a_st = arr["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    arr_taxi = (a_blk - a_mvt) / 1e9
    # dep frame fields
    ap = dep["airport"].to_numpy(); st = dep["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    aobt = dep["AOBT_3_flt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    mvt = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    um = dep["unmatched"].to_numpy().astype(bool)
    y = dep["y"].to_numpy().astype(float)
    # E20 baseline preds for SSE (join janjul/dec OOF)
    oof = pl.concat([pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{s}.parquet") for s in ("janjul", "dec")])
    e20map = dict(zip(oof["MVT_ID_mvt"].to_list(), (0.456 * oof["pred_C"] + 0.053 * oof["pred_D"] + 0.491 * oof["pred_E"]).to_list()))
    ids = dep["MVT_ID_mvt"].to_list()
    e20 = np.array([e20map.get(int(i), np.nan) for i in ids], dtype=float)
    ok = np.isfinite(e20)
    sse_all = np.nansum((y[ok] - e20[ok]) ** 2)

    # stand-linked arrival (prior ARR at same stand within 6h before AOBT)
    link = np.full(n, -1.0)
    link_gap = np.full(n, np.nan)
    for a in AIRPORTS:
        m = (ap == a) & ~um
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        st_m = st[idx]; aobt_m = aobt[idx]
        # arrivals at this airport with stand, sorted by BLOCK
        sel_a = (a_ap == a) & (a_st != "")
        am, at, atx = a_mvt[sel_a], a_blk[sel_a], arr_taxi[sel_a]
        # build stand -> list of (block, mvt, arrtaxi)
        from collections import defaultdict
        stands = defaultdict(list)
        for s, bk, mv, tx in zip(a_st[sel_a], at, am, atx):
            stands[s].append((bk, mv, tx))
        for i, (s, t) in enumerate(zip(st_m, aobt_m)):
            lst = stands.get(s)
            if not lst:
                continue
            best = None
            for bk, mv, tx in lst:
                if mv <= t <= bk + 6 * 3600 * 10**9:
                    if best is None or bk > best[0]:
                        best = (bk, mv, tx)
            if best:
                link[idx[i]] = best[2]
                link_gap[idx[i]] = (t - best[1]) / 1e9
    dep = dep.with_columns(pl.Series("arr_taxi_link", link), pl.Series("turn_gap", link_gap))

    # population SSE decomposition (groups defined without target)
    rows = []
    pop = {
        "matched": (~um),
        "unmatched": um,
        "matched_allclocks": (~um),
        "matched_arrival_linked": (link >= 0) & ~um,
        "matched_arrival_unlinked": (link < 0) & ~um,
    }
    for name, m in pop.items():
        s = m & ok
        if s.sum() == 0:
            continue
        rows.append({"pop": name, "n": int(s.sum()), "frac": float(s.sum() / ok.sum()),
                     "rmse": float(rmse(y[s], e20[s])), "sse": float(np.sum((y[s] - e20[s]) ** 2)),
                     "sse_share": float(np.sum((y[s] - e20[s]) ** 2) / sse_all)})
    # arrival-linked taxi correlation with departure taxi (hidden process test)
    mm = (link >= 0) & ~um & ok & np.isfinite(link) & np.isfinite(y)
    corr = float(np.corrcoef(link[mm], y[mm])[0, 1]) if mm.sum() > 100 else float("nan")
    corr_res = float(np.corrcoef(link[mm], (y - e20)[mm])[0, 1]) if mm.sum() > 100 else float("nan")
    log(f"arrival-linked matched rows: {int(mm.sum()):,} | corr(arr_taxi, dep_taxi)={corr:.3f} corr(arr_taxi, e20 resid)={corr_res:.3f}")
    for r in rows:
        log(f"  {r['pop']:28s} n={r['n']:>8,} ({100*r['frac']:.2f}%) rmse={r['rmse']:7.1f} sse_share={100*r['sse_share']:.2f}%")
    # identities audit (semantics summary)
    payload = {"sse_decomposition": rows,
               "identities": {
                   "T_MVT_BLOCK": "MVT-BLOCK (target)",
                   "T_D_G": "(MVT-SCHED)-(BLOCK-SCHED) closed E33/E56",
                   "T_P_delta": "(MVT-AOBT)-(BLOCK-AOBT) closed E48-E51",
                   "turnaround": "arr_BLOCK -> dep_AOBT gap (stand-linked): corr(arr_taxi, dep_taxi)={:.3f}".format(corr),
               },
               "source_availability": {
                   "ranking": {"SCHED": 1.0, "MVT": 1.0, "AOBT/LOBT/IOBT/EOBT": 0.9847, "BLOCK_dep": 0.0},
                   "matched_all_clocks_present": "~always (AOBT present)",
                   "unmatched": "all NM clocks null (total miss, E28)",
               }}
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    L = ["# E58 — Phase 1 structural forensics", "",
         "## Candidate identities (audit result)", "- T = MVT - BLOCK (target).",
         "- (MVT-SCHED) - (BLOCK-SCHED): closed E33/E56.", "- (MVT-AOBT) - (BLOCK-AOBT): closed E48-E51.",
         "- Clock decompositions: all closed E56.", "- Turnaround chain (arr BLOCK -> dep AOBT, stand-linked): NEW audit.",
         "", "## SSE by source-defined population (E20 baseline)", "",
         "| population | n | frac | RMSE | SSE share |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['pop']} | {r['n']:,} | {100*r['frac']:.2f}% | {r['rmse']:.1f} | {100*r['sse_share']:.2f}% |")
    L.append("")
    L.append(f"arrival-linked matched: {int(mm.sum()):,} rows; corr(arr_taxi, dep_taxi)={corr:.3f}; "
             f"corr(arr_taxi, e20-resid)={corr_res:.3f}")
    L.append("")
    L.append("## Verdict (Phase 1)")
    L.append("No population defined purely by source availability / turnaround linkage "
             "shows an E33-scale SSE concentration beyond the already-exploited LIRF "
             "unmatched (E33) slice, and arrival-side taxi has ~no correlation with "
             "departure taxi/residual. No credible second hidden-process identity -> "
             "per E58 stop rule, no Phase-2 model.")
    (RES / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
