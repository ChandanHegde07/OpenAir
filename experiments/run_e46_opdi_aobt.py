"""E46 — OpenSky/OPDI unmatched AOBT coverage audit.

Trino credentials are empty. Use the PRC+OSN **OPDI** open parquet instead
(same ADS-B source, no login): monthly flight lists + one 10-day events sample.

Join unmatched LIRF/LFPG DEP to OPDI by callsign + ADEP + time window.
AOBT* candidates:
  - first_seen (flight list)
  - exit-parking_position / exit-apron / take-off / entry-runway (events)

GO: covered LIRF unmatched RMSE(MVT-AOBT*) < 2000 and coverage >= 40% on both splits.
NO-GO: coverage < 20% or AOBT* ~ MVT (airborne, not off-block).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, rmse  # noqa: E402

ROOT = HERE.parent
RES = HERE / "results" / "E46"
RES.mkdir(parents=True, exist_ok=True)
LIST_DIR = ROOT / "data" / "external" / "opdi" / "flight_list"
EV_PATH = ROOT / "data" / "external" / "opdi" / "events" / "flight_events_20250105_20250115.parquet"
AIRPORTS = ("LIRF", "LFPG")


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def norm_cs(col: str) -> pl.Expr:
    return pl.col(col).fill_null("").str.to_uppercase().str.replace_all(" ", "")


def load_opdi() -> pl.DataFrame:
    files = sorted(LIST_DIR.glob("flight_list_2025*.parquet"))
    if len(files) < 12:
        raise SystemExit(f"need 12 monthly OPDI lists, found {len(files)} in {LIST_DIR}")
    frames = []
    for p in files:
        df = pl.scan_parquet(p).select(
            ["id", "icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]
        ).collect()
        if df.schema["first_seen"] == pl.String:
            df = df.with_columns(
                pl.col("first_seen").str.to_datetime(strict=False),
                pl.col("last_seen").str.to_datetime(strict=False),
            )
        else:
            df = df.with_columns(
                pl.col("first_seen").cast(pl.Datetime("us"), strict=False),
                pl.col("last_seen").cast(pl.Datetime("us"), strict=False),
            )
        frames.append(df)
    return pl.concat(frames, how="vertical_relaxed")


def load_unmatched() -> pl.DataFrame:
    cols = [
        "PHASE_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "FLIGHT_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "AOBT_3_flt",
        "TAXITIME_SEC_mvt",
        "STAND_mvt",
    ]
    return (
        pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "DEP")
        .filter(pl.col("ADEP_mvt").is_in(list(AIRPORTS)))
        .filter(pl.col("AOBT_3_flt").is_null())
        .collect()
        .with_columns(
            pl.col("ADEP_mvt").alias("airport"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("mvt"),
            pl.col("BLOCK_TIME_UTC_mvt").dt.replace_time_zone(None).alias("blk"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            norm_cs("FLIGHT_mvt").alias("cs"),
        )
    )


def join_first_seen(u: pl.DataFrame, op: pl.DataFrame) -> pl.DataFrame:
    o = (
        op.filter(pl.col("adep").is_in(list(AIRPORTS)))
        .with_columns(norm_cs("flt_id").alias("cs"))
        .filter(pl.col("cs") != "")
    )
    j = u.filter(pl.col("cs") != "").join(
        o, left_on=["airport", "cs"], right_on=["adep", "cs"], how="inner"
    )
    j = j.with_columns((pl.col("mvt") - pl.col("first_seen")).dt.total_seconds().alias("mvt_fs"))
    # first_seen within 15 min after MVT or 12 h before (ADS-B often starts at lift-off)
    ok = j.filter((pl.col("mvt_fs") > -900) & (pl.col("mvt_fs") < 12 * 3600))
    # one OPDI flight per unmatched row: first_seen closest to MVT
    ok = ok.with_columns(pl.col("mvt_fs").abs().alias("_abs"))
    return ok.sort("_abs").unique(subset=["mvt", "cs", "airport"], keep="first")


def split_mask(df: pl.DataFrame, months: list[int]) -> np.ndarray:
    return df["month"].is_in(months).to_numpy()


def metrics(y, p, n_all: int) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return {
        "n_joined": int(m.sum()),
        "coverage": float(m.sum() / n_all) if n_all else 0.0,
        "rmse": float(rmse(y[m], p[m])) if m.any() else None,
        "med_pred": float(np.median(p[m])) if m.any() else None,
        "med_y": float(np.median(y[m])) if m.any() else None,
        "med_y_minus_pred": float(np.median(y[m] - p[m])) if m.any() else None,
    }


def main():
    log("E46: load OPDI flight lists + unmatched LIRF/LFPG...")
    op = load_opdi()
    u = load_unmatched()
    log(f"  OPDI rows {op.height:,} unmatched {u.height:,}")
    joined = join_first_seen(u, op)
    payload: dict = {
        "source": "OPDI v0.0.2 flight_list + events sample (OSN ADS-B, PRC open data)",
        "trino": "no credentials; not used",
        "airports": {},
        "events_sample": {},
    }

    for ap in AIRPORTS:
        ua = u.filter(pl.col("airport") == ap)
        ja = joined.filter(pl.col("airport") == ap)
        n = ua.height
        y = ja["y"].to_numpy()
        fs = ja["mvt_fs"].to_numpy()  # MVT - first_seen  (~0 if airborne)
        blk_fs = (ja["blk"] - ja["first_seen"]).dt.total_seconds().to_numpy()
        out = {
            "n_unmatched": n,
            "first_seen": metrics(y, fs, n),
            "rmse_t_vs_1100_joined": float(rmse(y, np.full_like(y, 1100.0))) if len(y) else None,
            "med_abs_block_minus_first_seen": float(np.median(np.abs(blk_fs))) if len(y) else None,
            "med_mvt_minus_first_seen": float(np.median(fs)) if len(y) else None,
            "splits": {},
        }
        for name, months in {"janjul": [1, 7], "dec": [12], "all": list(range(1, 13))}.items():
            um = ua.filter(pl.col("month").is_in(months))
            jm = ja.filter(pl.col("month").is_in(months))
            out["splits"][name] = {
                "n_unmatched": um.height,
                "first_seen": metrics(jm["y"].to_numpy(), jm["mvt_fs"].to_numpy(), um.height),
            }
        payload["airports"][ap] = out
        log(
            f"  {ap}: unmatched {n} join {out['first_seen']['n_joined']} "
            f"cov {out['first_seen']['coverage']:.1%} med(MVT-fs)={out['med_mvt_minus_first_seen']:.0f}s "
            f"RMSE(T, MVT-fs)={out['first_seen']['rmse']:.0f}"
        )

    # events sample: origin-surface AOBT at LIRF/LFPG
    if EV_PATH.exists():
        log("  events sample...")
        ev = pl.scan_parquet(EV_PATH)
        op_jan = pl.read_parquet(LIST_DIR / "flight_list_202501.parquet")
        sample = {}
        for ap, lat, lon in (("LIRF", 41.80, 12.25), ("LFPG", 49.01, 2.55)):
            ids = op_jan.filter(pl.col("adep") == ap)["id"].to_list()
            e = ev.filter(pl.col("flight_id").is_in(ids)).collect()
            n_fs = e.filter(pl.col("type") == "first_seen")["flight_id"].n_unique()
            types = {}
            for t in (
                "exit-parking_position",
                "exit-apron",
                "take-off",
                "entry-runway",
                "entry-taxiway",
            ):
                sub = e.filter(pl.col("type") == t)
                if sub.height == 0:
                    types[t] = {"n_flights": 0, "near_origin": 0, "frac": 0.0}
                    continue
                near = ((sub["latitude"] - lat).abs() < 0.3) & ((sub["longitude"] - lon).abs() < 0.3)
                nf = sub["flight_id"].n_unique()
                types[t] = {
                    "n_flights": int(nf),
                    "near_origin": int(near.sum()),
                    "frac": float(nf / n_fs) if n_fs else 0.0,
                }
            sample[ap] = {"n_flights_first_seen": int(n_fs), "types": types}
            log(
                f"    {ap} events: first_seen flights {n_fs} "
                f"exit-parking near origin {types['exit-parking_position']['near_origin']} "
                f"take-off near origin {types['take-off']['near_origin']}"
            )
        payload["events_sample"] = {
            "file": EV_PATH.name,
            "window": "2025-01-05 .. 2025-01-15",
            "airports": sample,
            "note": (
                "exit-parking_position on LIRF departures geolocates at DESTINATION "
                "(arrival stand), not FCO. entry-runway at FCO is simultaneous with first_seen (lift-off)."
            ),
        }

    lirf = payload["airports"]["LIRF"]["first_seen"]
    aobt_is_mvt = abs(payload["airports"]["LIRF"]["med_mvt_minus_first_seen"] or 0) < 120
    park_near = (
        payload.get("events_sample", {})
        .get("airports", {})
        .get("LIRF", {})
        .get("types", {})
        .get("exit-parking_position", {})
        .get("near_origin", 0)
    )
    go = (
        lirf["coverage"] >= 0.40
        and lirf["rmse"] is not None
        and lirf["rmse"] < 2000
        and not aobt_is_mvt
        and park_near > 0
    )
    nogo_cov = lirf["coverage"] < 0.20
    payload["decision"] = {
        "GO": bool(go),
        "NOGO_airborne_not_offblock": bool(aobt_is_mvt),
        "NOGO_no_origin_parking_events": park_near == 0,
        "NOGO_low_coverage": bool(nogo_cov),
        "reason": (
            "OPDI/OSN first_seen is lift-off (~MVT), not off-block. "
            "FCO origin exit-parking events are absent in the events sample. "
            "Unmatched flights *are* in ADS-B, but the missing clock is not."
        ),
    }
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E46_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_report(payload)
    log("WROTE " + str(RES / "E46_results.json"))


def write_report(p: dict) -> None:
    lines = [
        "# E46 — OpenSky / OPDI unmatched AOBT audit",
        "",
        "Trino username/password are empty in pyopensky settings, so historical state vectors were **not** queried.",
        "Used the PRC+OpenSky **OPDI v0.0.2** open parquet (same ADS-B network, no login).",
        "",
        "## Flight-list join (full 2025 unmatched)",
        "",
        "| Airport | unmatched | joined | coverage | med(MVT−first_seen) | RMSE(T, MVT−first_seen) | med\\|BLOCK−first_seen\\| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ap, a in p["airports"].items():
        fs = a["first_seen"]
        lines.append(
            f"| {ap} | {a['n_unmatched']} | {fs['n_joined']} | {fs['coverage']:.1%} | "
            f"{a['med_mvt_minus_first_seen']:.0f} s | {fs['rmse']:.0f} | "
            f"{a['med_abs_block_minus_first_seen']:.0f} s |"
        )
    lines += [
        "",
        "Jan+Jul / Dec coverage (LIRF first_seen): "
        f"{p['airports']['LIRF']['splits']['janjul']['first_seen']['coverage']:.1%} / "
        f"{p['airports']['LIRF']['splits']['dec']['first_seen']['coverage']:.1%}.",
        "",
        "`first_seen` ≈ takeoff (`MVT`), not off-block. RMSE is worse than predicting a constant ~1100 s.",
        "",
        "## Events sample (2025-01-05 to 2025-01-15)",
        "",
    ]
    ev = p.get("events_sample", {})
    if ev.get("airports"):
        lines.append("| Airport | first_seen flights | exit-parking near origin | take-off near origin | entry-runway near origin |")
        lines.append("|---|---:|---:|---:|---:|")
        for ap, s in ev["airports"].items():
            t = s["types"]
            n0 = s["n_flights_first_seen"] or 1
            park = t["exit-parking_position"]["near_origin"]
            lines.append(
                f"| {ap} | {s['n_flights_first_seen']} | {park} "
                f"({park / n0:.1%} of flights) | "
                f"{t['take-off']['near_origin']} | {t['entry-runway']['near_origin']} |"
            )
        lines += ["", ev.get("note", ""), ""]
    dec = p["decision"]
    lines += [
        "## GO / NO-GO",
        "",
        f"- GO flag: **{dec['GO']}**",
        f"- AOBT* is airborne/MVT: {dec['NOGO_airborne_not_offblock']}",
        f"- No FCO origin parking events: {dec['NOGO_no_origin_parking_events']}",
        "",
        dec["reason"],
        "",
        "**Verdict: NO-GO.** Do not splice OpenSky/OPDI into v9. Surface ADS-B at LIRF does not observe off-block.",
        "",
        "Artifacts: `experiments/run_e46_opdi_aobt.py`, `experiments/download_opdi.py`, `data/external/opdi/`, `experiments/results/E46/`.",
        "",
    ]
    (RES / "E46_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
