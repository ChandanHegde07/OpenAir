"""E22: METAR weather join (Branch B — no ADS-B trajectories in the bundle).

STEP 0 found only flight-list parquet under data/ (30 columns, no lat/lon,
no surface tracks). air-data/ does not exist. Challenge eligibility allows
open, documented extra datasets.

This experiment joins Iowa Mesonet ASOS/METAR (public METAR/SPECI) to each
DEP at ranking-safe off-block (AOBT if present, else MVT). Features are
decoded visibility/wind/precip/fog flags. Added to the frozen E18-H residual
LightGBM (matched-only geo_mean, LIRF-override rows dropped, always-on
LIRF MVT−SCHED). No loss change, no rolling-stat queue revival.

Training files only for fits. METAR is an open external table, not ranking.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import AIRPORTS, DATA, ROOT, load_dep, mae, rmse, save_result  # noqa: E402
from run_e16a import (  # noqa: E402
    MODEL_DISRUPT_COLS,
    add_arrival_delay_state,
    add_push_disruption,
    add_rolling_quantiles,
    load_arr_delay,
    score_pred,
)
from run_e18_unmatched_specialist import fit_residual, prepare_split, slice_metrics  # noqa: E402
from run_e4_e5 import add_queue, add_traffic, load_arr  # noqa: E402
from common import add_causal_rolling  # noqa: E402

OUT = ROOT / "analysis" / "E22"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
METAR_DIR = DATA / "external" / "metar"
for d in (FIG, TAB, REP, FIG / "dec"):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1
E18H_OVERALL = 372.36
E18H_MATCHED = 250.98
E20_OVERALL = 368.03
E20_MATCHED = 244.76
E20_DEC = 228.45
E20_DEC_MATCHED = 215.90

METAR_NUM = [
    "met_vsby",
    "met_sknt",
    "met_gust",
    "met_tmpf",
    "met_relh",
    "met_p01i",
    "met_age_min",
    "met_low_vis",
    "met_fog",
    "met_precip",
    "met_strong_wind",
    "met_missing",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_metar_csv(p: Path) -> pl.DataFrame | None:
    if not p.exists() or p.stat().st_size < 50:
        return None
    raw = p.read_text(encoding="utf-8", errors="replace")
    if raw.lstrip().lower().startswith("<") or "station,valid" not in raw[:200].lower():
        log(f"  WARN {p.name} does not look like CSV ({p.stat().st_size} bytes)")
        return None
    df = pl.read_csv(p, try_parse_dates=False, infer_schema_length=5000, ignore_errors=True)
    cols = {c.lower(): c for c in df.columns}
    if any(n not in cols for n in ("station", "valid")):
        log(f"  WARN {p.name} columns={df.columns}")
        return None
    df = df.rename({cols[k]: k for k in cols})
    df = df.with_columns(
        pl.col("station").cast(pl.Utf8).str.to_uppercase().alias("station"),
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", strict=False).alias("valid"),
    )
    for c in ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "vsby", "gust"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    if "wxcodes" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("wxcodes"))
    else:
        df = df.with_columns(pl.col("wxcodes").cast(pl.Utf8))
    df = df.select(["station", "valid", "tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "vsby", "gust", "wxcodes"])
    return df.filter(pl.col("valid").is_not_null())


def load_metar() -> pl.DataFrame:
    frames = []
    for st in AIRPORTS:
        paths = sorted({p for p in METAR_DIR.glob(f"{st}*.csv") if p.is_file()})
        if not paths:
            log(f"  WARN missing METAR {st}")
            continue
        n = 0
        for p in paths:
            df = _read_metar_csv(p)
            if df is None or df.height == 0:
                continue
            frames.append(df)
            n += df.height
        log(f"  {st} METAR rows={n:,} files={len(paths)}")
    if not frames:
        raise SystemExit("no METAR files loaded")
    out = pl.concat(frames).unique(["station", "valid"], keep="last").sort(["station", "valid"])
    log(f"  METAR total {out.height:,} from {out['station'].n_unique()} stations")
    return out


def decode_and_join(dep: pl.DataFrame, met: pl.DataFrame) -> pl.DataFrame:
    wx = met.with_columns(
        pl.col("wxcodes").fill_null("").str.to_uppercase().alias("wx"),
    ).with_columns(
        pl.col("vsby").alias("met_vsby"),
        pl.col("sknt").alias("met_sknt"),
        pl.col("gust").alias("met_gust"),
        pl.col("tmpf").alias("met_tmpf"),
        pl.col("relh").alias("met_relh"),
        pl.col("p01i").alias("met_p01i"),
        (pl.col("vsby") < 1.0).cast(pl.Int8).alias("met_low_vis"),
        (
            pl.col("wx").str.contains("FG")
            | ((pl.col("vsby") < 0.5) & pl.col("vsby").is_not_null())
        )
        .cast(pl.Int8)
        .alias("met_fog"),
        (
            pl.col("wx").str.contains(r"RA|SN|DZ|SH|TS|GR|PL|IC")
            | ((pl.col("p01i") > 0) & pl.col("p01i").is_not_null())
        )
        .cast(pl.Int8)
        .alias("met_precip"),
        (pl.col("sknt") >= 20).cast(pl.Int8).alias("met_strong_wind"),
    ).select(
        pl.col("station").alias("airport"),
        "valid",
        "met_vsby",
        "met_sknt",
        "met_gust",
        "met_tmpf",
        "met_relh",
        "met_p01i",
        "met_low_vis",
        "met_fog",
        "met_precip",
        "met_strong_wind",
    )
    dep = dep.with_columns(
        pl.when(pl.col("AOBT_3_flt").is_not_null())
        .then(pl.col("AOBT_3_flt"))
        .otherwise(pl.col("MVT_TIME_UTC_mvt"))
        .alias("wx_t")
    ).sort(["airport", "wx_t"])
    wx = wx.sort(["airport", "valid"])
    j = dep.join_asof(wx, left_on="wx_t", right_on="valid", by="airport", strategy="backward")
    j = j.with_columns(
        ((pl.col("wx_t") - pl.col("valid")).dt.total_seconds() / 60.0).alias("met_age_min"),
    )
    stale = (pl.col("met_age_min") > 180) | pl.col("met_age_min").is_null()
    j = j.with_columns(stale.cast(pl.Int8).alias("met_missing"))
    # do not feed a >3 h-old report as if it were current weather
    for c in ["met_vsby", "met_sknt", "met_gust", "met_tmpf", "met_relh", "met_p01i"]:
        j = j.with_columns(
            pl.when(pl.col("met_missing") == 1).then(None).otherwise(pl.col(c)).alias(c)
        )
    for c in ["met_low_vis", "met_fog", "met_precip", "met_strong_wind"]:
        j = j.with_columns(
            pl.when(pl.col("met_missing") == 1)
            .then(0)
            .otherwise(pl.col(c).fill_null(0))
            .cast(pl.Int8)
            .alias(c)
        )
    return j


def featurize(dep: pl.DataFrame) -> pl.DataFrame:
    dep = add_causal_rolling(dep)
    dep = dep.with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    arr = load_arr()
    dep = add_traffic(dep, arr)
    dep = add_queue(dep)
    dep = add_push_disruption(dep)
    dep = add_rolling_quantiles(dep)
    dep = add_arrival_delay_state(dep, load_arr_delay())
    return dep


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


def tail_block(y, p, um) -> dict:
    s = score_pred(y, p, um)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    ok = np.isfinite(y) & np.isfinite(p)
    matched = ok & ~um
    r2 = (y - p) ** 2
    sse = float(np.sum(r2[matched])) if matched.any() else 0.0
    gt60 = matched & (y > 3600.0)
    sse_gt60 = float(np.sum(r2[gt60])) if gt60.any() else 0.0
    return {
        "overall": s["rmse_overall"],
        "matched": s["rmse_matched"],
        "unmatched": s["rmse_unmatched"],
        "mae_matched": s["mae_matched"],
        "rmse_lt20": s["rmse_lt20_matched"],
        "rmse_gt30": s["rmse_gt30_matched"],
        "rmse_gt60": s["rmse_gt60_matched"],
        "sse_share_gt30": s["sse_share_gt30_matched"],
        "sse_share_gt60": (sse_gt60 / sse) if sse > 0 else float("nan"),
        "n_gt30": s["n_gt30_matched"],
        "n_gt60": s["n_gt60_matched"],
        "n_matched": s["n_matched"],
        "n_unmatched": s["n_unmatched"],
    }


def run_split(dep, months, extra_wx: bool):
    pack = prepare_split(dep, months, matched_geo=True)
    extra = list(MODEL_DISRUPT_COLS) + (METAR_NUM if extra_wx else [])
    _, pred, model = fit_residual(
        pack["tr"], pack["va"], extra=extra, drop_override_from_train=True
    )
    y = pack["va"]["y"].to_numpy().astype(np.float64)
    um = pack["va"]["unmatched"].to_numpy()
    blk = tail_block(y, pred, um)
    blk.update(slice_metrics(y, pred, um, pack["va"]["airport"].to_numpy()))
    if extra_wx and model is not None:
        from run_e12_e9 import CAT_COLS, NUM_COLS as BASE_NUM

        feats = list(BASE_NUM) + [c for c in extra if c not in BASE_NUM] + list(CAT_COLS)
        booster = getattr(model, "booster_", None)
        if booster is not None:
            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            if len(gain) == len(feats):
                imp = sorted(zip(feats, gain.tolist()), key=lambda t: -t[1])
                blk["wx_importance"] = [(n, g) for n, g in imp if n.startswith("met_")]
                blk["top12"] = imp[:12]
    return blk, pack["va"], pred


def main():
    log("E22 STEP 0 recap: no air-data/, training is 30 flight-list columns, no trajectories.")
    log("Branch B: Iowa Mesonet METAR (open). Eligibility allows documented open extra data.")
    log("load METAR...")
    met = load_metar()
    log("featurize training DEP + join METAR...")
    dep = featurize(load_dep())
    # preserve pre-join row order so time_es_split ties match E18-H
    dep = dep.with_row_index("_ord")
    dep = decode_and_join(dep, met)
    dep = dep.sort("_ord").drop("_ord")
    n_miss = int(dep["met_missing"].sum())
    n_fog = int(dep["met_fog"].sum())
    n_precip = int(dep["met_precip"].sum())
    n_low = int(dep["met_low_vis"].sum())
    log(f"  DEP {dep.height:,} met_missing={n_miss:,} fog={n_fog:,} precip={n_precip:,} low_vis={n_low:,}")
    by_ap = (
        dep.group_by("airport")
        .agg(
            pl.len().alias("n"),
            pl.col("met_missing").mean().alias("miss_rate"),
            pl.col("met_fog").mean().alias("fog_rate"),
            pl.col("met_precip").mean().alias("precip_rate"),
            pl.col("met_low_vis").mean().alias("low_vis_rate"),
            pl.col("met_strong_wind").mean().alias("strong_wind_rate"),
            pl.col("met_vsby").median().alias("med_vsby"),
            pl.col("met_age_min").median().alias("med_age_min"),
        )
        .sort("airport")
    )
    by_ap.write_csv(TAB / "e22_metar_coverage.csv")
    log(str(by_ap))

    payload = {
        "branch": "B_metar",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": by_ap.to_dicts(),
    }

    for split_name, months in ("janjul", [1, 7]), ("dec", [12]):
        log(f"===== {split_name} E18-H baseline =====")
        base, va, pred_b = run_split(dep, months, extra_wx=False)
        log(f"  E18-H overall={base['overall']:.2f} matched={base['matched']:.2f} "
            f">30={base['rmse_gt30']:.1f} >60={base['rmse_gt60']:.1f} sse>30={base['sse_share_gt30']:.3f}")
        log(f"===== {split_name} E22 + METAR =====")
        new, va2, pred_n = run_split(dep, months, extra_wx=True)
        log(f"  E22    overall={new['overall']:.2f} matched={new['matched']:.2f} "
            f">30={new['rmse_gt30']:.1f} >60={new['rmse_gt60']:.1f} sse>30={new['sse_share_gt30']:.3f}")
        if new.get("wx_importance"):
            log(f"  METAR gain {new['wx_importance'][:8]}")
        payload[split_name] = {"baseline": base, "e22": new}

        # residual vs fog/precip on matched (frozen E18-H residual, before METAR)
        y = va["y"].to_numpy().astype(np.float64)
        um = va["unmatched"].to_numpy().astype(bool)
        r = y - pred_b
        mm = (~um) & np.isfinite(r)
        fog = va["met_fog"].to_numpy().astype(np.float64)
        prec = va["met_precip"].to_numpy().astype(np.float64)
        lowv = va["met_low_vis"].to_numpy().astype(np.float64)
        wind = va["met_strong_wind"].to_numpy().astype(np.float64)
        payload[split_name]["diag"] = {
            "mean_resid_fog": float(np.nanmean(r[mm & (fog == 1)])) if (mm & (fog == 1)).any() else float("nan"),
            "mean_resid_nofog": float(np.nanmean(r[mm & (fog == 0)])) if (mm & (fog == 0)).any() else float("nan"),
            "mean_resid_precip": float(np.nanmean(r[mm & (prec == 1)])) if (mm & (prec == 1)).any() else float("nan"),
            "mean_resid_noprecip": float(np.nanmean(r[mm & (prec == 0)])) if (mm & (prec == 0)).any() else float("nan"),
            "mean_resid_low_vis": float(np.nanmean(r[mm & (lowv == 1)])) if (mm & (lowv == 1)).any() else float("nan"),
            "mean_resid_strong_wind": float(np.nanmean(r[mm & (wind == 1)])) if (mm & (wind == 1)).any() else float("nan"),
            "n_fog_matched": int((mm & (fog == 1)).sum()),
            "n_precip_matched": int((mm & (prec == 1)).sum()),
            "n_low_vis_matched": int((mm & (lowv == 1)).sum()),
            "n_strong_wind_matched": int((mm & (wind == 1)).sum()),
        }
        make_split_plots(split_name, y, um, pred_b, pred_n, fog, prec)

    j, d = payload["janjul"], payload["dec"]
    dj = j["e22"]["overall"] - j["baseline"]["overall"]
    dd = d["e22"]["overall"] - d["baseline"]["overall"]
    dj_m = j["e22"]["matched"] - j["baseline"]["matched"]
    dd_m = d["e22"]["matched"] - d["baseline"]["matched"]
    beat_e20_j = j["e22"]["overall"] < E20_OVERALL - 0.5
    beat_e20_d = d["e22"]["overall"] < E20_DEC - 0.5
    no_e20_matched_reg = (
        j["e22"]["matched"] <= E20_MATCHED + 0.5 and d["e22"]["matched"] <= E20_DEC_MATCHED + 0.5
    )
    if beat_e20_j and beat_e20_d and no_e20_matched_reg:
        verdict = "ACCEPTED"
    elif dj < -0.5 and dd <= 0.5 and dj_m <= 0.5:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECTED"
    reason = (
        f"E18-H→E22 Jan+Jul {j['baseline']['overall']:.2f}→{j['e22']['overall']:.2f} ({dj:+.2f}), "
        f"matched {j['baseline']['matched']:.2f}→{j['e22']['matched']:.2f} ({dj_m:+.2f}); "
        f">30 {j['baseline']['rmse_gt30']:.1f}→{j['e22']['rmse_gt30']:.1f}; "
        f">60 {j['baseline']['rmse_gt60']:.1f}→{j['e22']['rmse_gt60']:.1f}; "
        f"SSE>30 {100*j['baseline']['sse_share_gt30']:.1f}%→{100*j['e22']['sse_share_gt30']:.1f}%; "
        f"SSE>60 {100*j['baseline']['sse_share_gt60']:.1f}%→{100*j['e22']['sse_share_gt60']:.1f}%. "
        f"Dec {d['baseline']['overall']:.2f}→{d['e22']['overall']:.2f} ({dd:+.2f}), "
        f"matched {d['baseline']['matched']:.2f}→{d['e22']['matched']:.2f} ({dd_m:+.2f}). "
        f"vs E20 {E20_OVERALL:.2f}/{E20_DEC:.2f}: "
        f"{'beats' if beat_e20_j else 'does not beat'} Jan+Jul, "
        f"{'beats' if beat_e20_d else 'does not beat'} Dec."
    )
    payload["decision"] = {"verdict": verdict, "reason": reason, "d_janjul": dj, "d_dec": dd}
    log(f"VERDICT {verdict}")
    log(reason)
    write_report(payload)
    save_result("E22", payload)
    (TAB / "e22_findings.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), OUT / "run_e22.py")
    log("done")


def make_split_plots(split_name, y, um, pred_b, pred_n, fog, prec):
    y = np.asarray(y, dtype=np.float64)
    um = np.asarray(um, dtype=bool)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    pred_n = np.asarray(pred_n, dtype=np.float64)
    fog = np.asarray(fog, dtype=np.float64)
    prec = np.asarray(prec, dtype=np.float64)
    mm = (~um) & np.isfinite(y) & np.isfinite(pred_b) & np.isfinite(pred_n)
    dest = FIG / ("dec" if split_name == "dec" else ".")
    dest.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = ["Overall", "Matched", ">30 min", ">60 min"]
    b = tail_block(y, pred_b, um)
    n = tail_block(y, pred_n, um)
    xb = [b["overall"], b["matched"], b["rmse_gt30"], b["rmse_gt60"]]
    xn = [n["overall"], n["matched"], n["rmse_gt30"], n["rmse_gt60"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, xb, 0.36, label="E18-H", color="#4C78A8")
    ax.bar(x + 0.18, xn, 0.36, label="E22 METAR", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(f"E22 vs E18-H ({split_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "01_rmse_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    rb = y[mm] - pred_b[mm]
    for ax, flag, title in (
        (axes[0], fog[mm] == 1, "fog"),
        (axes[1], prec[mm] == 1, "precip"),
    ):
        groups = [rb[~flag], rb[flag]] if flag.any() and (~flag).any() else [rb]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticklabels([f"no {title}", title] if len(groups) == 2 else [title])
        ax.set_title(f"E18-H residual by {title} ({split_name})")
        ax.set_ylabel("y − pred (s)")
    fig.tight_layout()
    fig.savefig(dest / "02_resid_by_weather.png")
    plt.close(fig)


def write_report(payload):
    j, d = payload["janjul"], payload["dec"]
    lines = []
    a = lines.append
    a("# E22 — METAR weather (Branch B)")
    a("")
    a("**Project:** OpenAir")
    a("**Experiment:** E22")
    a("**Validation:** Jan+Jul 2025 training holdout")
    a("**Stress:** December 2025")
    a("**Data:** 12 `training_*.parquet` + public Iowa Mesonet ASOS/METAR for 2025.")
    a("No ranking/submitting in fits.")
    a("Script: `experiments/run_e22_metar.py`.")
    a("")
    a("## STEP 0 — data audit")
    a("")
    a("### What is on disk")
    a("")
    a("`data/` contains exactly:")
    a("")
    a("- 12 `training_*.parquet` files (calendar 2025)")
    a("- `ranking.parquet`")
    a("- `submitting.parquet`")
    a("- `external/metar/{ICAO}.csv` (this experiment; Iowa Mesonet ASOS 2025)")
    a("")
    a("**`air-data/` does not exist** in the repo, parent `prc/`, or anywhere")
    a("searched. A filename search for ads-b / trajectory / lat-lon files under")
    a("OpenAir returned nothing.")
    a("")
    a("### Training schema (30 columns, confirmed on `training_2025-01-01_…`)")
    a("")
    a("Movement clocks and airport fields: `MVT_ID_mvt`, `FLIGHT_ID_mvt`,")
    a("`FLIGHT_mvt`, `FLIGHT_RULE_mvt`, `ADEP_mvt`, `ADES_mvt`, `PHASE_mvt`,")
    a("`MVT_TIME_UTC_mvt`, `BLOCK_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`,")
    a("`AIRCRAFT_TYPE_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `TAXITIME_SEC_mvt`.")
    a("")
    a("NM flight-list fields: `LOBT_flt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`,")
    a("`ADES_FILED_flt`, `MARKET_SEGMENT_flt`, `IOBT_flt`, `FLIGHT_RULE_flt`,")
    a("`FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`,")
    a("`AIRCRAFT_OPERATOR_flt`, `EOBT_1_flt`, `ARVT_1_flt`, `AOBT_3_flt`,")
    a("`ARVT_3_flt`.")
    a("")
    a("**No lat/lon, no taxiway, no weather, no 1-second ADS-B surface")
    a("trajectories.** This edition is flight-list + movement clocks only, unlike")
    a("past PRC challenges that shipped daily trajectory parquet.")
    a("")
    a("### Extra-data rules (2026 eligibility)")
    a("")
    a("https://prc-data-challenge-2026.netlify.app/eligibility.html — prize-eligible")
    a("solutions require:")
    a("")
    a("- *All used external datasets are openly accessible/usable and documented.*")
    a("- *All additional datasets used are openly available under an open source")
    a("  license.*")
    a("")
    a("Same policy as 2025 (`The use of additional and/or external dataset is")
    a("permitted if open data and documented.`). Iowa Environmental Mesonet")
    a("ASOS/METAR is public observational weather (upstream NWS/NOAA + WMO")
    a("partners). **Branch B selected.**")
    a("")
    a("## Join")
    a("")
    a("Nearest METAR with `valid ≤ AOBT` (else `MVT`) at the same ICAO station.")
    a("Reports older than 3 h are treated as missing and their decoded values are")
    a("nulled. Features: visibility, wind, gust, temp, RH, precip amount, age,")
    a("flags for low vis / fog / precip / strong wind. Frozen E18-H residual")
    a("LightGBM + these columns. No loss change, no E19 rolling-queue revival.")
    a("")
    a("## Coverage")
    a("")
    a("| Airport | n | miss rate | fog | precip | low vis | strong wind | med vis (sm) | med age (min) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["coverage"]:
        med_v = r.get("med_vsby")
        med_a = r.get("med_age_min")
        med_v_s = f"{med_v:.2f}" if med_v is not None else "—"
        med_a_s = f"{med_a:.1f}" if med_a is not None else "—"
        a(
            f"| {r['airport']} | {r['n']:,} | {100*r['miss_rate']:.2f}% | "
            f"{100*r['fog_rate']:.2f}% | {100*r['precip_rate']:.2f}% | "
            f"{100*r.get('low_vis_rate', 0):.2f}% | {100*r.get('strong_wind_rate', 0):.2f}% | "
            f"{med_v_s} | {med_a_s} |"
        )
    a("")
    a("## Holdout vs E18-H (fair feature test) and E20 (current best)")
    a("")
    a("### Jan+Jul 2025")
    a("")
    a("| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE share >30 | SSE share >60 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    b, n = j["baseline"], j["e22"]
    a(
        f"| E18-H | {b['overall']:.2f} | {b['matched']:.2f} | {b['unmatched']:.2f} | "
        f"{b['rmse_gt30']:.1f} | {b['rmse_gt60']:.1f} | {100*b['sse_share_gt30']:.1f}% | "
        f"{100*b['sse_share_gt60']:.1f}% |"
    )
    a(
        f"| E22 METAR | {n['overall']:.2f} | {n['matched']:.2f} | {n['unmatched']:.2f} | "
        f"{n['rmse_gt30']:.1f} | {n['rmse_gt60']:.1f} | {100*n['sse_share_gt30']:.1f}% | "
        f"{100*n['sse_share_gt60']:.1f}% |"
    )
    a(f"| E20 ensemble (ref) | {E20_OVERALL:.2f} | {E20_MATCHED:.2f} | — | — | — | — | — |")
    a("")
    a("### December 2025")
    a("")
    a("| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE share >30 | SSE share >60 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    b, n = d["baseline"], d["e22"]
    a(
        f"| E18-H | {b['overall']:.2f} | {b['matched']:.2f} | {b['unmatched']:.2f} | "
        f"{b['rmse_gt30']:.1f} | {b['rmse_gt60']:.1f} | {100*b['sse_share_gt30']:.1f}% | "
        f"{100*b['sse_share_gt60']:.1f}% |"
    )
    a(
        f"| E22 METAR | {n['overall']:.2f} | {n['matched']:.2f} | {n['unmatched']:.2f} | "
        f"{n['rmse_gt30']:.1f} | {n['rmse_gt60']:.1f} | {100*n['sse_share_gt30']:.1f}% | "
        f"{100*n['sse_share_gt60']:.1f}% |"
    )
    a(f"| E20 ensemble (ref) | {E20_DEC:.2f} | {E20_DEC_MATCHED:.2f} | — | — | — | — | — |")
    a("")
    a("Frozen-E18-H residual vs weather flags (Jan+Jul matched):")
    a("")
    dg = j["diag"]
    a(f"- fog n={dg['n_fog_matched']:,} mean resid {dg['mean_resid_fog']:.1f} vs no-fog {dg['mean_resid_nofog']:.1f}")
    a(f"- precip n={dg['n_precip_matched']:,} mean resid {dg['mean_resid_precip']:.1f} vs none {dg['mean_resid_noprecip']:.1f}")
    a(f"- low vis n={dg['n_low_vis_matched']:,} mean resid {dg['mean_resid_low_vis']:.1f}")
    a(f"- strong wind n={dg['n_strong_wind_matched']:,} mean resid {dg['mean_resid_strong_wind']:.1f}")
    a("")
    wx_imp = j["e22"].get("wx_importance") or []
    if wx_imp:
        a("LightGBM gain on METAR columns (Jan+Jul):")
        a("")
        for name, g in wx_imp:
            a(f"- `{name}`: {g:.1f}")
        a("")
    a("## Decision")
    a("")
    a(f"**Verdict:** {payload['decision']['verdict']}")
    a("")
    a(payload["decision"]["reason"])
    a("")
    a("Accept only if E22 beats E20 on **both** splits without regressing")
    a("matched/overall. Readme current-model is updated only on ACCEPT.")
    a("Beating E18-H without beating E20 is INCONCLUSIVE (feature may still be")
    a("worth stacking later; it is not the new current model).")
    a("")
    a("Artifacts: `analysis/E22/`, `data/external/metar/`, `data/external/README.md`.")
    a("")
    path = REP / "E22_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
