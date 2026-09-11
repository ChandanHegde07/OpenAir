"""E47 — leftover SSE attack: OPDI-in regime + stand-release in LIRF G,
and a cheap EGLL unmatched head.

in_opdi = callsign appears in OPDI flight_list with first_seen near MVT.
Training-only OPDI 2025 lists. Ranking-safe if 2026-01/07 lists are used at
submit time (they exist on the OPDI portal).

Candidates splice onto E20 OOF (v9 = unmatched-only CatBoost G on LIRF_u).
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
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402
from run_e34d_enriched import EXTRA, enrich  # noqa: E402
from run_e45_stand_release import REL, add_stand_release, cdf, e20_blend, fit_cat  # noqa: E402

RES = HERE / "results" / "E47"
RES.mkdir(parents=True, exist_ok=True)
LIST_DIR = HERE.parent / "data" / "external" / "opdi" / "flight_list"
SEED = 1
FEATS_V9 = NUM + EXTRA
FEATS_A = FEATS_V9 + REL + ["in_opdi"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_opdi_lirf() -> pl.DataFrame:
    frames = []
    for p in sorted(LIST_DIR.glob("flight_list_2025*.parquet")):
        df = pl.read_parquet(p).select(["flt_id", "adep", "first_seen"])
        if df.schema["first_seen"] == pl.String:
            df = df.with_columns(pl.col("first_seen").str.to_datetime(strict=False))
        frames.append(df)
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("adep") == "LIRF")
        .with_columns(pl.col("flt_id").fill_null("").str.to_uppercase().str.replace_all(" ", "").alias("cs"))
        .filter(pl.col("cs") != "")
        .select(["cs", "first_seen"])
    )


def attach_in_opdi(dep: pl.DataFrame, op: pl.DataFrame) -> pl.DataFrame:
    # Only unmatched rows are joined (matched G training does not use in_opdi).
    u = dep.filter(pl.col("unmatched")).select(
        [
            "MVT_ID_mvt",
            pl.col("FLIGHT_mvt").fill_null("").str.to_uppercase().str.replace_all(" ", "").alias("cs"),
            pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("_mvt_naive"),
        ]
    )
    j = u.join(op, on="cs", how="left")
    j = j.with_columns((pl.col("_mvt_naive") - pl.col("first_seen")).dt.total_seconds().alias("_dt"))
    hit = (
        j.filter((pl.col("_dt") > -900) & (pl.col("_dt") < 12 * 3600))
        .select("MVT_ID_mvt")
        .unique()
        .with_columns(pl.lit(1).alias("in_opdi"))
    )
    return dep.join(hit, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("in_opdi").fill_null(0).cast(pl.Int8)
    )


def overall_splice(oof: pl.DataFrame, id_va: np.ndarray, pred_lu: np.ndarray):
    e20 = e20_blend(oof)
    y = oof["y"].to_numpy().astype(float)
    pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
    idx = np.array([pos[int(i)] for i in id_va])
    p = e20.copy()
    p[idx] = pred_lu
    return float(rmse(y, p)), float(np.sum((y - p) ** 2)), e20, idx, y


def main():
    log("E47: LIRF frame...")
    dep_all = add_causal_rolling(load_dep()).with_columns(
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix")
    )
    arr_surf = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select([pl.lit("LIRF").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    arr_l = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select(["STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        .collect()
    )
    lirf = dep_all.filter(pl.col("airport") == "LIRF")
    lirf = add_surface(lirf, arr_surf)
    lirf = enrich(lirf, lirf["mvt_sched"].to_numpy().astype(float))
    log("  stand-release + in_opdi...")
    lirf = add_stand_release(lirf, arr_l)
    op = load_opdi_lirf()
    lirf = attach_in_opdi(lirf, op)
    log(f"  LIRF unmatched in_opdi rate {float(lirf.filter(pl.col('unmatched'))['in_opdi'].mean()):.3f}")

    payload = {"splits": {}, "egll": {}}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        log(f"  split {split}")
        tr = lirf.filter(~pl.col("month").is_in(months))
        va = lirf.filter(pl.col("month").is_in(months) & pl.col("unmatched"))
        tr_u = tr.filter(pl.col("unmatched"))
        Dt = tr_u["mvt_sched"].to_numpy().astype(float)
        yt = tr_u["y"].to_numpy().astype(float)
        Gt = Dt - yt
        D = va["mvt_sched"].to_numpy().astype(float)
        y = va["y"].to_numpy().astype(float)
        in_va = va["in_opdi"].to_numpy().astype(bool)

        m_v9 = fit_cat(tr_u, Gt, FEATS_V9)
        t_v9 = np.maximum(D - np.asarray(m_v9.predict(cdf(va, FEATS_V9)), float), 0.0)

        m_a = fit_cat(tr_u, Gt, FEATS_A)
        g_a = np.asarray(m_a.predict(cdf(va, FEATS_A)), float)
        t_a = np.maximum(D - g_a, 0.0)

        # two-head: OUT -> D, IN -> G trained on IN only
        tr_in = tr_u.filter(pl.col("in_opdi") == 1)
        t_b = D.copy()
        if tr_in.height >= 80 and in_va.any():
            m_in = fit_cat(tr_in, tr_in["mvt_sched"].to_numpy().astype(float) - tr_in["y"].to_numpy().astype(float), FEATS_A)
            t_b[in_va] = np.maximum(
                D[in_va] - np.asarray(m_in.predict(cdf(va.filter(pl.col("in_opdi") == 1), FEATS_A)), float),
                0.0,
            )

        # OUT -> D, IN -> v9 G (no refit)
        t_c = t_v9.copy()
        t_c[~in_va] = D[~in_va]

        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        id_va = va["MVT_ID_mvt"].to_numpy()
        cands = {"v9": t_v9, "A_opdi_sr": t_a, "B_twohead": t_b, "C_outD": t_c, "D": D}

        # EGLL unmatched -> train unmatched median (ranking-present)
        egll_tr = dep_all.filter((pl.col("airport") == "EGLL") & (~pl.col("month").is_in(months)) & pl.col("unmatched"))
        egll_va = dep_all.filter((pl.col("airport") == "EGLL") & pl.col("month").is_in(months) & pl.col("unmatched"))
        med = float(np.median(egll_tr["y"].to_numpy())) if egll_tr.height else 1300.0
        egll_ids = egll_va["MVT_ID_mvt"].to_numpy()
        egll_y = egll_va["y"].to_numpy().astype(float)

        e20 = e20_blend(oof)
        y_all = oof["y"].to_numpy().astype(float)
        pos = {int(i): k for k, i in enumerate(oof["MVT_ID_mvt"].to_numpy())}
        eidx = np.array([pos[int(i)] for i in egll_ids])
        e20_eg = e20[eidx]
        t_eg = np.full_like(egll_y, med)
        payload["egll"][split] = {
            "n": int(len(egll_y)),
            "e20_rmse": float(rmse(egll_y, e20_eg)),
            "median_rmse": float(rmse(egll_y, t_eg)),
            "train_median": med,
        }

        out = {}
        for name, pred in cands.items():
            ov, sse, _, _, _ = overall_splice(oof, id_va, pred)
            # + EGLL median splice on top of this LIRF pred
            p_both = e20.copy()
            lidx = np.array([pos[int(i)] for i in id_va])
            p_both[lidx] = pred
            p_both[eidx] = t_eg
            out[name] = {
                "lirf_u_rmse": float(rmse(y, pred)),
                "overall": ov,
                "overall_plus_egll": float(rmse(y_all, p_both)),
                "in_opdi_n": int(in_va.sum()),
                "out_n": int((~in_va).sum()),
            }
            log(
                f"    [{split}] {name}: LIRF_u {out[name]['lirf_u_rmse']:.1f} "
                f"overall {ov:.2f} +EGLL {out[name]['overall_plus_egll']:.2f}"
            )
        payload["splits"][split] = {
            "n": int(va.height),
            "in_opdi_rate": float(in_va.mean()),
            "e20_overall": float(rmse(y_all, e20)),
            "cands": out,
            "egll": payload["egll"][split],
        }

    jj, dc = payload["splits"]["janjul"]["cands"], payload["splits"]["dec"]["cands"]
    v9j, v9d = jj["v9"]["overall"], dc["v9"]["overall"]
    decision = {}
    for name in ("A_opdi_sr", "B_twohead", "C_outD"):
        for tag, key in [("lirf", "overall"), ("plus_egll", "overall_plus_egll")]:
            dj = v9j - jj[name][key]
            dd = v9d - dc[name][key]
            go = dj >= 5.0 and dd >= -1.0 and jj[name]["lirf_u_rmse"] <= jj["v9"]["lirf_u_rmse"]
            decision[f"{name}:{tag}"] = {
                "janjul_drop": dj,
                "dec_drop": dd,
                "GO": bool(go),
            }
            log(f"  DECISION {name}/{tag}: GO={go} jj {dj:+.2f} dec {dd:+.2f}")
    payload["decision"] = decision
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E47_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    lines = [
        "# E47 — OPDI-in regime + EGLL unmatched",
        "",
        "| split | cand | LIRF_u | overall | overall+EGLL med |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("janjul", "dec"):
        s = payload["splits"][split]
        for n, c in s["cands"].items():
            lines.append(f"| {split} | {n} | {c['lirf_u_rmse']:.1f} | {c['overall']:.2f} | {c['overall_plus_egll']:.2f} |")
        e = s["egll"]
        lines.append(f"| {split} | EGLL e20/med | {e['e20_rmse']:.1f}/{e['median_rmse']:.1f} |  |  |")
    lines += ["", "## GO/NO-GO", ""]
    any_go = False
    for k, v in decision.items():
        lines.append(f"- {k}: GO={v['GO']} Jan+Jul {v['janjul_drop']:+.2f} Dec {v['dec_drop']:+.2f}")
        any_go |= v["GO"]
    lines.append(f"\n**Verdict: {'GO' if any_go else 'NO-GO'}**\n")
    (RES / "E47_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("WROTE " + str(RES / "E47_results.json"))


if __name__ == "__main__":
    main()
