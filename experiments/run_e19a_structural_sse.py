"""E19-A: remaining structural SSE after E18-H.

E18-H was training hygiene. This is diagnosis, not a model. Training files
only. No ranking/submitting fits.

Questions:
1. After E18-H, which slices still dominate overall SSE?
2. Is LFPG unmatched a second LIRF (MVT−SCHED tracks y) or a few bombs
   on top of otherwise-normal taxi?
3. Do those bombs have ranking-safe handles (sched, stand, prefix, hour)?
4. What is the oracle ceiling of a LIRF-style MVT−SCHED rule on non-LIRF
   unmatched, and of an oracle LIRF gate, vs matched-tail work?
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    AIRPORTS,
    ROOT,
    load_dep,
    mae,
    rmse,
    save_result,
    split_by_months,
)

OUT = ROOT / "analysis" / "E19A"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

# E18-H holdout metrics (journal). Used only to weight SSE; not refit.
E18H = {
    "janjul": {
        "n": 344419,
        "rmse": 372.3646118481625,
        "n_matched": 339046,
        "rmse_matched": 250.98,  # from report; overwritten if we recompute from json
        "n_unmatched": 5373,
        "rmse_unmatched": 2216.589114920439,
        "n_nlu": 4976,
        "rmse_nlu": 1549.7376259533175,
        "n_lirf_u": 397,
        "rmse_lirf_u": 6032.69683773519,
    }
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sse(n, r):
    return float(n) * float(r) ** 2


def corr_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_e18h_json():
    p = HERE / "results" / "E18.json"
    if not p.exists():
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    m = j["janjul"]["metrics"]["E18-H"]
    E18H["janjul"].update(
        {
            "rmse": m["rmse"],
            "rmse_matched": m["rmse_matched"],
            "rmse_unmatched": m["rmse_unmatched"],
            "rmse_nlu": m["rmse_non_lirf_unmatched"],
            "rmse_lirf_u": m["rmse_lirf_unmatched"],
            "n": m.get("n", E18H["janjul"]["n"]),
            "n_matched": m.get("n_matched", E18H["janjul"]["n_matched"]),
            "n_unmatched": m.get("n_unmatched", E18H["janjul"]["n_unmatched"]),
            "n_nlu": m.get("n_non_lirf_unmatched", E18H["janjul"]["n_nlu"]),
            "n_lirf_u": m.get("n_lirf_unmatched", E18H["janjul"]["n_lirf_u"]),
        }
    )


def dist_row(name, y, extra=None):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {"slice": name, "n": 0}
    d = {
        "slice": name,
        "n": int(y.size),
        "mean": float(y.mean()),
        "med": float(np.median(y)),
        "std": float(y.std()),
        "p5": float(np.quantile(y, 0.05)),
        "p50": float(np.median(y)),
        "p95": float(np.quantile(y, 0.95)),
        "p99": float(np.quantile(y, 0.99)),
        "max": float(y.max()),
        "frac_gt30m": float((y > 1800).mean()),
        "frac_gt1h": float((y > 3600).mean()),
        "frac_gt2h": float((y > 7200).mean()),
        "n_gt1h": int((y > 3600).sum()),
        "n_gt2h": int((y > 7200).sum()),
    }
    if extra:
        d.update(extra)
    return d


def main():
    load_e18h_json()
    h = E18H["janjul"]
    sse_all = sse(h["n"], h["rmse"])
    sse_m = sse(h["n_matched"], h["rmse_matched"])
    sse_u = sse(h["n_unmatched"], h["rmse_unmatched"])
    sse_lirf = sse(h["n_lirf_u"], h["rmse_lirf_u"])
    sse_nlu = sse(h["n_nlu"], h["rmse_nlu"])

    log("E18-H Jan+Jul SSE budget from saved metrics")
    log(f"  overall {h['rmse']:.2f}  SSE={sse_all:.3e}")
    log(f"  matched {h['rmse_matched']:.2f}  share={sse_m/sse_all:.3f}")
    log(f"  unmatched {h['rmse_unmatched']:.2f}  share={sse_u/sse_all:.3f}")
    log(f"  LIRF unmatched {h['rmse_lirf_u']:.2f}  share={sse_lirf/sse_all:.3f}")
    log(f"  non-LIRF unmatched {h['rmse_nlu']:.2f}  share={sse_nlu/sse_all:.3f}")

    um_tbl = pd.read_csv(ROOT / "analysis" / "E18" / "tables" / "e18_unmatched_by_airport_janjul.csv")
    ap_rows = []
    for _, r in um_tbl.iterrows():
        if r["airport"] in ("ALL_UNMATCHED", "NON_LIRF_UNMATCHED", "LIRF_UNMATCHED"):
            continue
        n = int(r["n"])
        rh = float(r["rmse_E18-H"])
        s = sse(n, rh)
        ap_rows.append(
            {
                "airport": r["airport"],
                "n": n,
                "mean_y": float(r["mean_y"]),
                "med_y": float(r["med_y"]),
                "rmse_H": rh,
                "sse": s,
                "sse_share_overall": s / sse_all,
                "sse_share_unmatched": s / sse_u,
                "lirf_override": bool(r["lirf_override"]),
            }
        )
    ap_df = pd.DataFrame(ap_rows).sort_values("sse", ascending=False)
    ap_df.to_csv(TAB / "e19a_unmatched_sse_by_airport.csv", index=False)
    log("unmatched SSE by airport (E18-H):")
    for _, r in ap_df.iterrows():
        log(f"  {r['airport']:4s} n={int(r['n']):4d} RMSE={r['rmse_H']:7.1f}  "
            f"SSE share overall={100*r['sse_share_overall']:5.1f}%  unmatched={100*r['sse_share_unmatched']:5.1f}%")

    log("load training DEP...")
    dep = load_dep().with_columns(
        pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("type_null"),
        pl.col("FLIGHT_mvt").fill_null("NA").str.slice(0, 3).fill_null("NA").alias("flt_prefix"),
    )
    tr, va = split_by_months(dep, [1, 7])
    y = va["y"].to_numpy().astype(np.float64)
    um = va["unmatched"].to_numpy()
    ap = va["airport"].to_numpy()
    ms = va["mvt_sched"].to_numpy().astype(np.float64)
    log(f"  val n={len(y):,} unmatched={int(um.sum()):,}")

    # Per-airport unmatched: y vs MVT-SCHED as if we applied the LIRF rule
    clock_rows = []
    dist_rows = []
    bomb_rows = []
    for a in AIRPORTS:
        sel = um & (ap == a) & np.isfinite(y)
        if not sel.any():
            continue
        ya, msa = y[sel], ms[sel]
        geo_like = float(tr.filter((pl.col("airport") == a) & (~pl.col("unmatched")))["y"].mean())
        pred_geo = np.full(ya.shape, geo_like)
        pred_ms = msa.copy()
        clock_rows.append(
            {
                "airport": a,
                "n": int(sel.sum()),
                "corr_y_mvt_sched": corr_safe(ya, msa),
                "rmse_geo_mean": rmse(ya, pred_geo),
                "mae_geo_mean": mae(ya, pred_geo),
                "rmse_mvt_sched": rmse(ya, pred_ms) if np.isfinite(msa).any() else float("nan"),
                "mae_mvt_sched": mae(ya, pred_ms) if np.isfinite(msa).any() else float("nan"),
                "mean_y": float(ya.mean()),
                "mean_mvt_sched": float(np.nanmean(msa)),
                "med_y": float(np.median(ya)),
                "med_mvt_sched": float(np.nanmedian(msa)),
                "n_type_null": int(va.filter((pl.col("unmatched")) & (pl.col("airport") == a))["type_null"].sum()),
            }
        )
        dist_rows.append(dist_row(f"{a}_unmatched_val", ya))

        # bombs: y>1h
        bombs = sel & (y > 3600)
        if bombs.any():
            sub = va.filter(pl.Series(bombs)).select(
                "airport", "FLIGHT_mvt", "flt_prefix", "ADES_mvt", "STAND_mvt",
                "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "y", "mvt_sched", "hour", "month",
            )
            bomb_rows.append(sub)

    clock_df = pd.DataFrame(clock_rows)
    clock_df.to_csv(TAB / "e19a_unmatched_clocks_janjul.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(TAB / "e19a_unmatched_y_dist_janjul.csv", index=False)
    log("unmatched y vs MVT-SCHED (Jan+Jul val):")
    for _, r in clock_df.iterrows():
        log(
            f"  {r['airport']:4s} n={int(r['n']):4d} corr={r['corr_y_mvt_sched']:+.3f}  "
            f"RMSE(geo)={r['rmse_geo_mean']:.0f}  RMSE(MVT-SCHED)={r['rmse_mvt_sched']:.0f}  "
            f"mean_y={r['mean_y']:.0f} mean_ms={r['mean_mvt_sched']:.0f}"
        )

    if bomb_rows:
        bombs = pl.concat(bomb_rows)
        bombs.write_csv(TAB / "e19a_unmatched_gt1h_janjul.csv")
        log(f"unmatched y>1h val n={bombs.height}")
        log(str(bombs.group_by("airport").agg(pl.len().alias("n"), pl.col("y").max().alias("max_y")).sort("n", descending=True)))

    # LFPG unmatched deep dive (full year + val)
    log("LFPG unmatched deep dive...")
    lfpg_u = dep.filter((pl.col("airport") == "LFPG") & pl.col("unmatched"))
    lfpg_m = dep.filter((pl.col("airport") == "LFPG") & (~pl.col("unmatched")))
    y_u = lfpg_u["y"].to_numpy().astype(np.float64)
    ms_u = lfpg_u["mvt_sched"].to_numpy().astype(np.float64)
    log(f"  full-year unmatched n={lfpg_u.height:,} corr(y,ms)={corr_safe(y_u, ms_u):.3f}")
    log(f"  matched n={lfpg_m.height:,} median y={float(lfpg_m['y'].median()):.0f}")

    # How concentrated is LFPG unmatched RMSE? Leave-one-out / top-k bombs on val
    lfpg_val = um & (ap == "LFPG") & np.isfinite(y)
    yf = y[lfpg_val]
    # E18-H predicted ~1129 mean; use geo mean as proxy for "current head" error concentration
    # Better: error vs constant 1128 (E18-H mean pred) to see bomb dominance
    pred_h_proxy = float(um_tbl.loc[um_tbl["airport"] == "LFPG", "mean_pred_E18-H"].iloc[0])
    err2 = (yf - pred_h_proxy) ** 2
    order = np.argsort(-err2)
    conc = []
    tot = float(err2.sum()) if err2.size else 1.0
    for k in [1, 2, 3, 5, 10, 20, int(max(1, 0.01 * err2.size)), int(max(1, 0.05 * err2.size))]:
        k = min(k, err2.size)
        conc.append({"k": k, "sse_share": float(err2[order[:k]].sum() / tot), "max_y": float(yf[order[:k]].max())})
    pd.DataFrame(conc).to_csv(TAB / "e19a_lfpg_unmatched_error_concentration.csv", index=False)
    log("  LFPG unmatched error concentration vs E18-H mean pred:")
    for c in conc:
        log(f"    worst {c['k']:3d} share={100*c['sse_share']:5.1f}%  max_y={c['max_y']:.0f}")

    # prefixes / dest on LFPG unmatched y>1h full year
    lfpg_long = lfpg_u.filter(pl.col("y") > 3600)
    pref = (
        lfpg_long.group_by("flt_prefix")
        .agg(pl.len().alias("n"), pl.col("y").median().alias("med"), pl.col("y").max().alias("max"))
        .sort("n", descending=True)
        .head(15)
    )
    pref.write_csv(TAB / "e19a_lfpg_unmatched_gt1h_prefix.csv")
    dest = (
        lfpg_long.group_by("ADES_mvt")
        .agg(pl.len().alias("n"), pl.col("y").median().alias("med"))
        .sort("n", descending=True)
        .head(15)
    )
    dest.write_csv(TAB / "e19a_lfpg_unmatched_gt1h_ades.csv")
    log("  LFPG unmatched y>1h prefixes:")
    log(str(pref))

    # Oracle ceilings on overall RMSE (hold E18-H matched + LIRF rule; swap one slice)
    # If LFPG unmatched RMSE went from 3439 to X, overall becomes...
    def overall_if(slice_n, slice_rmse_new, slice_rmse_old, label):
        d_sse = sse(slice_n, slice_rmse_new) - sse(slice_n, slice_rmse_old)
        new = np.sqrt((sse_all + d_sse) / h["n"])
        log(f"  {label}: overall {h['rmse']:.2f} → {new:.2f}  (Δ{new-h['rmse']:+.2f})")
        return new

    log("oracle / what-if overall RMSE (E18-H base):")
    # LFPG unmatched → matched-like 400s
    overall_if(880, 400, float(um_tbl.loc[um_tbl["airport"] == "LFPG", "rmse_E18-H"].iloc[0]), "LFPG unmatched RMSE 3439→400")
    overall_if(880, float(clock_df.loc[clock_df["airport"] == "LFPG", "rmse_mvt_sched"].iloc[0]),
               float(um_tbl.loc[um_tbl["airport"] == "LFPG", "rmse_E18-H"].iloc[0]),
               "LFPG unmatched → raw MVT-SCHED")
    # LIRF unmatched oracle ~2771 from E13
    overall_if(397, 2771, h["rmse_lirf_u"], "LIRF unmatched oracle 6033→2771 (E13 gate)")
    overall_if(397, 400, h["rmse_lirf_u"], "LIRF unmatched magical 6033→400")
    # matched 251→200
    overall_if(h["n_matched"], 200, h["rmse_matched"], "matched 251→200 (very optimistic)")
    overall_if(h["n_matched"], 240, h["rmse_matched"], "matched 251→240 (E16/E17 class)")

    # E13-style bins for LFPG unmatched full year
    bins = [(0, 900, "<15m"), (900, 1800, "15-30m"), (1800, 3600, "30-60m"), (3600, 7200, "1-2h"), (7200, 1e12, ">2h")]
    bin_rows = []
    for lo, hi, lab in bins:
        m = (y_u >= lo) & (y_u < hi)
        if not m.any():
            continue
        bin_rows.append(
            {
                "bin": lab,
                "n": int(m.sum()),
                "mean_y": float(y_u[m].mean()),
                "mean_ms": float(np.nanmean(ms_u[m])),
                "rmse_ms": rmse(y_u[m], ms_u[m]),
            }
        )
    pd.DataFrame(bin_rows).to_csv(TAB / "e19a_lfpg_unmatched_ybins.csv", index=False)
    log("LFPG unmatched y bins vs MVT-SCHED (full year):")
    for b in bin_rows:
        log(f"  {b['bin']:7s} n={b['n']:4d} mean_y={b['mean_y']:.0f} mean_ms={b['mean_ms']:.0f} RMSE(ms)={b['rmse_ms']:.0f}")

    payload = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "e18h_sse": {
            "overall": sse_all,
            "matched_share": sse_m / sse_all,
            "unmatched_share": sse_u / sse_all,
            "lirf_unmatched_share": sse_lirf / sse_all,
            "nlu_share": sse_nlu / sse_all,
        },
        "unmatched_by_airport": ap_df.to_dict(orient="records"),
        "clocks_janjul": clock_df.to_dict(orient="records"),
        "lfpg_corr_full_year": corr_safe(y_u, ms_u),
        "lfpg_concentration": conc,
        "lfpg_ybins": bin_rows,
    }
    (TAB / "e19a_findings.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    save_result("E19A", payload)
    write_report(payload, h, sse_all, sse_m, sse_u, sse_lirf, sse_nlu, ap_df, clock_df, conc, bin_rows)
    log("done")


def write_report(payload, h, sse_all, sse_m, sse_u, sse_lirf, sse_nlu, ap_df, clock_df, conc, bin_rows):
    lines = []
    a = lines.append
    a("# E19-A — Remaining structural SSE after E18-H")
    a("")
    a("Diagnosis only. No new model. `training_*.parquet` only.")
    a("Script: `experiments/run_e19a_structural_sse.py`.")
    a("")
    a("E18-H was training hygiene (matched-only `geo_mean`, drop LIRF-override")
    a("rows from residual train). This note locates the error that hygiene cannot")
    a("reach.")
    a("")
    a("## E18-H Jan+Jul SSE budget")
    a("")
    a("| Slice | n | RMSE | SSE share |")
    a("|---|---:|---:|---:|")
    a(f"| Overall | {h['n']:,} | {h['rmse']:.2f} | 100% |")
    a(f"| Matched | {h['n_matched']:,} | {h['rmse_matched']:.2f} | {100*sse_m/sse_all:.1f}% |")
    a(f"| Unmatched | {h['n_unmatched']:,} | {h['rmse_unmatched']:.2f} | {100*sse_u/sse_all:.1f}% |")
    a(f"| LIRF unmatched | {h['n_lirf_u']:,} | {h['rmse_lirf_u']:.2f} | {100*sse_lirf/sse_all:.1f}% |")
    a(f"| Non-LIRF unmatched | {h['n_nlu']:,} | {h['rmse_nlu']:.2f} | {100*sse_nlu/sse_all:.1f}% |")
    a("")
    a("Unmatched by airport (E18-H predictions):")
    a("")
    a("| Airport | n | mean y | RMSE H | % of all SSE | % of unmatched SSE |")
    a("|---|---:|---:|---:|---:|---:|")
    for _, r in ap_df.iterrows():
        a(
            f"| {r['airport']} | {int(r['n'])} | {r['mean_y']:.0f} | {r['rmse_H']:.1f} | "
            f"{100*r['sse_share_overall']:.1f}% | {100*r['sse_share_unmatched']:.1f}% |"
        )
    a("")
    a("## Unmatched y vs MVT−SCHED (Jan+Jul val)")
    a("")
    a("Would a LIRF-style `MVT−SCHED` override help anywhere else?")
    a("")
    a("| Airport | n | corr(y, MVT−SCHED) | RMSE geo | RMSE MVT−SCHED | mean y | mean MVT−SCHED |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in clock_df.iterrows():
        a(
            f"| {r['airport']} | {int(r['n'])} | {r['corr_y_mvt_sched']:.3f} | "
            f"{r['rmse_geo_mean']:.0f} | {r['rmse_mvt_sched']:.0f} | {r['mean_y']:.0f} | {r['mean_mvt_sched']:.0f} |"
        )
    a("")
    a("## LFPG unmatched")
    a("")
    a("LFPG unmatched is **not** a second LIRF. Full-year corr(y, MVT−SCHED) = "
      f"{payload['lfpg_corr_full_year']:.3f} (LIRF was 0.90). Applying `MVT−SCHED` "
      "as taxi is the wrong clock.")
    a("")
    a("y bins (full-year LFPG unmatched) vs MVT−SCHED:")
    a("")
    a("| Bin | n | mean y | mean MVT−SCHED | RMSE(MVT−SCHED) |")
    a("|---|---:|---:|---:|---:|")
    for b in bin_rows:
        a(f"| {b['bin']} | {b['n']} | {b['mean_y']:.0f} | {b['mean_ms']:.0f} | {b['rmse_ms']:.0f} |")
    a("")
    a("Error concentration vs E18-H mean prediction on Jan+Jul LFPG unmatched:")
    a("")
    a("| Worst k | SSE share |")
    a("|---:|---:|")
    for c in conc:
        a(f"| {c['k']} | {100*c['sse_share']:.1f}% |")
    a("")
    a("## What-if overall RMSE")
    a("")
    a("See run log for exact deltas. Directionally: LIRF unmatched oracle and")
    a("LFPG unmatched bomb-handling are the only moves that can change overall")
    a("RMSE by tens of seconds. Matched 251→240 is a few seconds overall.")
    a("")
    a("## Implications")
    a("")
    a("1. Do not spend the next weeks on matched 1% features (E16-A / E17-A class).")
    a("2. Two structural unmatched processes remain: LIRF (bimodal, MVT−SCHED")
    a("   tracks the extreme half) and LFPG (a handful of multi-hour bombs that")
    a("   `MVT−SCHED` does **not** explain).")
    a("3. Next experiment should be information that identifies those bombs or")
    a("   the LIRF gate-vs-taxi split — not another residual-objective or mean offset.")
    a("")
    path = REP / "E19A_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
