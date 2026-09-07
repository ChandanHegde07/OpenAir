"""E19-B: ranking-safe gate for wrong-day BLOCK bombs.

E19-A: 97% of LFPG unmatched SSE on Jan+Jul is two easyJet rows whose
BLOCK is 16–23 h *before* schedule (y − (MVT−SCHED) ≈ 16–23 h). A 24 h
wrap would score them; applying it ungated wrecks the other unmatched.

This experiment:
  Phase 1 — catalog unmatched (and matched) rows with y − mvt_sched > 12 h.
  Phase 2 — train-only ranking-safe gate search (airport, prefix, mvt_sched,
            hour). Precision must be high: wrap on a normal unmatched row
            is a ~24 h error.
  Phase 3 — if a gate passes, splice wrap onto frozen E18-H on Jan+Jul and
            December. Kill if December overall rises.

Fits: training_*.parquet only. No ranking/submitting.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
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
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    load_arr_delay,
    score_pred,
)
from run_e18_unmatched_specialist import (  # noqa: E402
    fit_residual,
    slice_metrics,
)
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from run_e16a import attach_hour_baseline  # noqa: E402
from common import add_causal_rolling  # noqa: E402

OUT = ROOT / "analysis" / "E19B"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for d in (FIG, TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

DAY = 86400.0
HALFDAY = 43200.0
BOMB_Y = 28800.0  # 8 h: true bombs, not 1–2 h long taxi
WRAP = 86400.0
BASELINE_OVERALL = 372.36
BASELINE_MATCHED = 250.98


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def add_prefix(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("FLIGHT_mvt").fill_null("NA").str.slice(0, 3).fill_null("NA").alias("flt_prefix"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"),
        (pl.col("y") - pl.col("mvt_sched")).alias("block_minus_sched_neg"),  # SCHED-BLOCK = y - mvt_sched
    )


def catalog(dep: pl.DataFrame) -> pl.DataFrame:
    """Rows whose implied BLOCK is >12 h before SCHED: y - mvt_sched > 12 h."""
    return dep.filter(
        pl.col("y").is_finite()
        & pl.col("mvt_sched").is_finite()
        & ((pl.col("y") - pl.col("mvt_sched")) > HALFDAY)
    ).select(
        "airport",
        "unmatched",
        "FLIGHT_mvt",
        "flt_prefix",
        "ADES_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt",
        "y",
        "mvt_sched",
        "hour",
        "month",
        "type_null",
    ).with_columns(
        (pl.col("y") - pl.col("mvt_sched")).alias("sched_minus_block"),
        ((pl.col("y") - pl.col("mvt_sched")) / DAY).alias("days_block_early"),
        (pl.col("mvt_sched") + DAY).alias("wrap_24h"),
    )


def gate_mask(df: pl.DataFrame, spec: dict) -> np.ndarray:
    um = df["unmatched"].to_numpy()
    ap = df["airport"].to_numpy()
    ms = df["mvt_sched"].to_numpy().astype(np.float64)
    pref = df["flt_prefix"].to_numpy()
    hour = df["hour"].to_numpy()
    m = np.ones(len(um), dtype=bool)
    if spec.get("unmatched_only", True):
        m &= um.astype(bool)
    if spec.get("exclude_lirf", True):
        m &= ap != "LIRF"
    if spec.get("airports"):
        m &= np.isin(ap, spec["airports"])
    if spec.get("prefixes"):
        m &= np.isin(pref, spec["prefixes"])
    if spec.get("ms_max") is not None:
        m &= np.isfinite(ms) & (ms < spec["ms_max"])
    if spec.get("ms_min") is not None:
        m &= np.isfinite(ms) & (ms > spec["ms_min"])
    if spec.get("hours"):
        m &= np.isin(hour, spec["hours"])
    return m


def bomb_label(y, ms) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    ms = np.asarray(ms, dtype=np.float64)
    return np.isfinite(y) & np.isfinite(ms) & ((y - ms) > HALFDAY) & (y > BOMB_Y)


def eval_gate(df: pl.DataFrame, spec: dict) -> dict:
    y = df["y"].to_numpy().astype(np.float64)
    ms = df["mvt_sched"].to_numpy().astype(np.float64)
    g = gate_mask(df, spec)
    b = bomb_label(y, ms)
    tp = int((g & b).sum())
    fp = int((g & ~b).sum())
    fn = int((~g & b).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "name": spec["name"],
        "n_gate": int(g.sum()),
        "n_bomb": int(b.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "spec": spec,
    }


def wrap_pred(ms, k=1.0) -> np.ndarray:
    return np.asarray(ms, dtype=np.float64) + k * WRAP


def splice(base, replacement, mask) -> np.ndarray:
    out = np.asarray(base, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool) & np.isfinite(replacement)
    out[m] = np.asarray(replacement, dtype=np.float64)[m]
    return out


def featurize(dep: pl.DataFrame) -> pl.DataFrame:
    dep = add_causal_rolling(dep)
    if "type_null" not in dep.columns:
        dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    return add_prefix(dep)


def prepare_e18h(dep: pl.DataFrame, months: list[int]) -> dict:
    tr0, va0 = split_by_months(dep, months)
    tabs = geometry_tables(tr0.filter(~pl.col("unmatched")))
    tr = attach_geometry(tr0, tabs, 30)
    va = attach_geometry(va0, tabs, 30)
    tr = attach_hour_baseline(tr, tr)
    va = attach_hour_baseline(tr, va)
    tr = add_prefix(tr) if "flt_prefix" not in tr.columns else tr
    va = add_prefix(va) if "flt_prefix" not in va.columns else va
    return {"tr": tr, "va": va}


def json_conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main():
    log("E19-B: catalog wrong-day BLOCK on training DEP...")
    dep0 = add_prefix(load_dep())
    cat = catalog(dep0)
    cat.write_csv(TAB / "e19b_wrongday_catalog.csv")
    log(f"  catalog n={cat.height} (y - mvt_sched > 12h)")
    log(str(cat.group_by(["airport", "unmatched"]).agg(
        pl.len().alias("n"),
        pl.col("y").median().alias("med_y"),
        pl.col("y").max().alias("max_y"),
        pl.col("mvt_sched").median().alias("med_ms"),
    ).sort(["unmatched", "n"], descending=[True, True])))

    um_cat = cat.filter(pl.col("unmatched"))
    log("  unmatched wrong-day prefixes:")
    log(str(um_cat.group_by("flt_prefix").agg(
        pl.len().alias("n"), pl.col("airport").n_unique().alias("n_ap"),
        pl.col("y").median().alias("med_y"), pl.col("mvt_sched").median().alias("med_ms"),
    ).sort("n", descending=True).head(20)))
    log("  unmatched wrong-day sample:")
    log(str(um_cat.select("airport", "FLIGHT_mvt", "ADES_mvt", "y", "mvt_sched", "sched_minus_block", "hour", "month").sort("y", descending=True).head(15)))

    # --- gate search on TRAIN side of Jan+Jul (months not in {1,7}) ---
    tr_j, va_j = split_by_months(dep0, [1, 7])
    tr_d, va_d = split_by_months(dep0, [12])

    candidates = []
    for ms_max in [1800, 2400, 3600, 5400, 7200, 10800]:
        candidates.append({"name": f"nlu_ms<{ms_max}", "unmatched_only": True, "exclude_lirf": True, "ms_max": ms_max, "ms_min": 0})
        candidates.append({"name": f"LFPG_ms<{ms_max}", "unmatched_only": True, "exclude_lirf": True, "airports": ["LFPG"], "ms_max": ms_max, "ms_min": 0})
        candidates.append({"name": f"EJU_ms<{ms_max}", "unmatched_only": True, "exclude_lirf": True, "prefixes": ["EJU"], "ms_max": ms_max, "ms_min": 0})
        candidates.append({"name": f"LFPG_EJU_ms<{ms_max}", "unmatched_only": True, "exclude_lirf": True, "airports": ["LFPG"], "prefixes": ["EJU"], "ms_max": ms_max, "ms_min": 0})
    candidates.append({"name": "LFPG_EJU", "unmatched_only": True, "exclude_lirf": True, "airports": ["LFPG"], "prefixes": ["EJU"]})
    candidates.append({"name": "EJU_unmatched_nlu", "unmatched_only": True, "exclude_lirf": True, "prefixes": ["EJU"]})
    # easyJet + easyJet-like
    for prefs in (["EJU"], ["EJU", "EZY"], ["EJU", "EZY", "EZS"]):
        candidates.append({"name": f"LFPG_{'_'.join(prefs)}_ms<3600", "unmatched_only": True, "exclude_lirf": True, "airports": ["LFPG"], "prefixes": prefs, "ms_max": 3600, "ms_min": 0})

    gate_rows = []
    log("Phase 2: gate search on Jan+Jul TRAIN (not val)...")
    for spec in candidates:
        tr_s = eval_gate(tr_j, spec)
        va_s = eval_gate(va_j, spec)
        dec_s = eval_gate(va_d, spec)
        row = {
            **{f"tr_{k}": v for k, v in tr_s.items() if k != "spec"},
            **{f"va_{k}": v for k, v in va_s.items() if k not in ("spec", "name")},
            **{f"dec_{k}": v for k, v in dec_s.items() if k not in ("spec", "name")},
            "spec": spec,
        }
        gate_rows.append(row)
    gdf = pd.DataFrame(gate_rows)
    gdf.to_csv(TAB / "e19b_gate_search.csv", index=False)

    # Acceptable: train precision >= 0.5 and at least 1 TP, n_gate not huge
    # Prefer high precision; recall secondary
    viable = []
    for _, r in gdf.iterrows():
        if r["tr_n_gate"] == 0:
            continue
        if r["tr_precision"] >= 0.5 and r["tr_tp"] >= 1 and r["tr_n_gate"] <= 30:
            viable.append(r)
            log(f"  viable {r['tr_name']}: train n={int(r['tr_n_gate'])} P={r['tr_precision']:.2f} R={r['tr_recall']:.2f}  "
                f"val n={int(r['va_n_gate'])} P={r['va_precision']:.2f} R={r['va_recall']:.2f}  "
                f"dec n={int(r['dec_n_gate'])} P={r['dec_precision']:.2f}")
    if not viable:
        log("  no gate with train precision>=0.5 and n_gate<=30")
        # still show best precision
        gdf2 = gdf[gdf["tr_n_gate"] > 0].sort_values(["tr_precision", "tr_recall"], ascending=False)
        log("  top train precision:")
        for _, r in gdf2.head(8).iterrows():
            log(f"    {r['tr_name']}: n={int(r['tr_n_gate'])} P={r['tr_precision']:.2f} R={r['tr_recall']:.2f} valP={r['va_precision']:.2f}")

    # Choose: max train precision, then max val TP, then smaller n
    chosen = None
    if viable:
        chosen = sorted(
            viable,
            key=lambda r: (-float(r["tr_precision"]), -int(r["va_tp"]), int(r["tr_n_gate"])),
        )[0]
        log(f"CHOSEN GATE {chosen['tr_name']}")

    # --- Phase 3: E18-H + splice ---
    log("featurize for E18-H (training only)...")
    dep = featurize(load_dep())
    payload = {"catalog_n": int(cat.height), "catalog_unmatched_n": int(um_cat.height), "gates": gate_rows, "chosen": None}

    def run_split(months, split_name, spec):
        label = "Jan+Jul" if split_name == "janjul" else "December"
        log(f"===== {split_name} E18-H + wrap =====")
        pack = prepare_e18h(dep, months)
        tr, va = pack["tr"], pack["va"]
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy()
        ap = va["airport"].to_numpy()
        ms = va["mvt_sched"].to_numpy().astype(np.float64)
        log("  fitting E18-H...")
        _, pred0, _ = fit_residual(tr, va, extra=list(MODEL_DISRUPT_COLS), drop_override_from_train=True)
        m0 = slice_metrics(y, pred0, um, ap)
        log(f"  E18-H overall={m0['rmse']:.2f} matched={m0['rmse_matched']:.2f} unmatched={m0['rmse_unmatched']:.2f}")

        wrap = wrap_pred(ms, 1.0)
        b = bomb_label(y, ms) & um.astype(bool) & (ap != "LIRF")
        pred_or = splice(pred0, wrap, b)
        metrics = {
            "E18-H": m0,
            "oracle_nlu_bomb_wrap": slice_metrics(y, pred_or, um, ap),
        }
        log(f"  oracle wrap n={int(b.sum())} overall={metrics['oracle_nlu_bomb_wrap']['rmse']:.2f}")
        gstat = {"n_bomb": int(b.sum())}
        if spec is not None:
            g = gate_mask(va, spec["spec"])
            pred_w = splice(pred0, wrap, g)
            metrics["E19-B"] = slice_metrics(y, pred_w, um, ap)
            gstat.update(
                {
                    "n_gate": int(g.sum()),
                    "n_gate_and_bomb": int((g & b).sum()),
                    "n_fp": int((g & ~b).sum()),
                    "y_gated": [float(v) for v in y[g]] if g.any() else [],
                    "ms_gated": [float(v) for v in ms[g]] if g.any() else [],
                    "pred0_gated": [float(v) for v in pred0[g]] if g.any() else [],
                    "airports_gated": [str(v) for v in ap[g]] if g.any() else [],
                    "prefixes_gated": [str(v) for v in va["flt_prefix"].to_numpy()[g]] if g.any() else [],
                }
            )
            log(f"  gate n={gstat['n_gate']} overlap={gstat['n_gate_and_bomb']} fp={gstat['n_fp']}")
            log(f"  E19-B overall={metrics['E19-B']['rmse']:.2f} unmatched={metrics['E19-B']['rmse_unmatched']:.2f}")
        return {"metrics": metrics, "gate": gstat, "n": int(len(y))}

    spec_chosen = None if chosen is None else {"name": chosen["tr_name"], "spec": chosen["spec"]}
    payload["chosen"] = spec_chosen
    payload["janjul"] = run_split([1, 7], "janjul", spec_chosen)
    payload["dec"] = run_split([12], "dec", spec_chosen)

    # decision
    j0 = payload["janjul"]["metrics"]["E18-H"]["rmse"]
    d0 = payload["dec"]["metrics"]["E18-H"]["rmse"]
    if spec_chosen and "E19-B" in payload["janjul"]["metrics"]:
        j1 = payload["janjul"]["metrics"]["E19-B"]["rmse"]
        d1 = payload["dec"]["metrics"]["E19-B"]["rmse"]
        jo = payload["janjul"]["metrics"]["oracle_nlu_bomb_wrap"]["rmse"]
        do = payload["dec"]["metrics"]["oracle_nlu_bomb_wrap"]["rmse"]
        if j1 < j0 - 0.5 and d1 <= d0 + 0.5:
            verdict = "ACCEPTED"
        elif jo < j0 - 2 and (j1 >= j0 or d1 > d0 + 0.5):
            verdict = "INCONCLUSIVE"
        else:
            verdict = "REJECTED"
        reason = (
            f"Gate {spec_chosen['name']}. Jan+Jul {j0:.2f}→{j1:.2f} ({j1-j0:+.2f}); "
            f"Dec {d0:.2f}→{d1:.2f} ({d1-d0:+.2f}). "
            f"Oracle wrap on true nlu bombs: Jan+Jul {jo:.2f}, Dec {do:.2f}."
        )
    else:
        verdict = "REJECTED"
        reason = "No ranking-safe gate with train precision>=0.5 and n<=30. Wrap remains an oracle, not a rule."
        jo = payload["janjul"]["metrics"].get("oracle_nlu_bomb_wrap", {}).get("rmse")
    payload["decision"] = {"verdict": verdict, "reason": reason}
    log(f"VERDICT {verdict}")
    log(reason)

    write_report(payload, cat, um_cat, gdf)
    save_result("E19B", payload)
    (TAB / "e19b_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e19b.py")
    log("done")


def write_report(payload, cat, um_cat, gdf):
    lines = []
    a = lines.append
    a("# E19-B — Wrong-day BLOCK gate")
    a("")
    a("Training files only. Script: `experiments/run_e19b_wrongday_block.py`.")
    a("")
    a("## Catalog")
    a("")
    a(f"Rows with `y − (MVT−SCHED) > 12 h` (implied BLOCK more than 12 h before SCHED): **{cat.height}**.")
    a(f"Unmatched among them: **{um_cat.height}**.")
    a("")
    a("## Gate search")
    a("")
    a("Train = all 2025 months except Jan+Jul. Precision = fraction of gated rows that are true bombs")
    a("(`y − mvt_sched > 12 h` and `y > 8 h`). Wrap on a false positive is a ~24 h RMSE bomb.")
    a("")
    if payload.get("chosen"):
        a(f"**Chosen:** `{payload['chosen']['name']}`")
        a("")
    else:
        a("**No viable gate** (train precision ≥ 0.5 and n_gate ≤ 30).")
        a("")
    a("## Holdout")
    a("")
    jm = payload["janjul"]["metrics"]
    dm = payload["dec"]["metrics"]
    a("| Split | E18-H overall | E19-B overall | Oracle wrap overall |")
    a("|---|---:|---:|---:|")
    def cell(blob, k):
        return f"{blob[k]['rmse']:.2f}" if k in blob else "—"
    a(f"| Jan+Jul | {jm['E18-H']['rmse']:.2f} | {cell(jm,'E19-B')} | {cell(jm,'oracle_nlu_bomb_wrap')} |")
    a(f"| December | {dm['E18-H']['rmse']:.2f} | {cell(dm,'E19-B')} | {cell(dm,'oracle_nlu_bomb_wrap')} |")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    path = REP / "E19B_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
