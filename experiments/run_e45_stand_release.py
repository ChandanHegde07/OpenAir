"""E45 — delay-window stand-release as a G-regime identifier (LIRF unmatched).

Identity stays T = D - G, D = MVT-SCHED, G = BLOCK-SCHED.

New observable (ranking-safe): whether another movement occupied this
stand during (SCHED, MVT). That is a physical constraint on BLOCK, not a
rolling congestion count.

  t_v    = first other DEP.MVT or ARR.BLOCK at same stand in (SCHED, MVT)
  has_tv = 1{t_v exists}
  G_ub   = t_v - SCHED     # BLOCK <= t_v  =>  T >= MVT - t_v
  T_lb   = MVT - t_v

Variants
  v9    unmatched-only CatBoost G (E35 production, no stand-release)
  A     v9 + stand-release features in the G model
  B     soft mixture P(G<300) -> T = p*T_phys + (1-p)*D
  C     hard has_tv -> D else T_phys  (diagnostic; expected to lose full-year)

Splits: Jan+Jul primary, December stress. Training files only.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, add_causal_rolling, load_dep, rmse  # noqa: E402
from run_e33_delay_decomposition import CAT, NUM, add_surface  # noqa: E402
from run_e34d_enriched import EXTRA, enrich  # noqa: E402

RES = HERE / "results" / "E45"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
FEATS_V9 = NUM + EXTRA
REL = ["has_tv", "G_ub", "T_lb", "n_dep_mvt", "n_arr_blk", "n_int", "tv_hour", "stand_remote"]
FEATS_A = FEATS_V9 + REL
TPHYS_FALLBACK = 1100.0
ALPHAS = [0.85, 0.9, 0.95, 1.0]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _i64(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype("datetime64[ns]").astype(np.int64)


def add_stand_release(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Attach delay-window stand-release features. Open interval (SCHED, MVT)."""
    n = dep.height
    stands = dep["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
    sched = _i64(dep["SCHED_TIME_UTC_mvt"])
    mvt = _i64(dep["MVT_TIME_UTC_mvt"])

    dep_st = stands
    dep_t = mvt
    arr_ok = arr.filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    arr_st = (
        arr_ok["STAND_mvt"].fill_null("").cast(pl.String).to_numpy()
        if arr_ok.height
        else np.array([], dtype=object)
    )
    arr_t = _i64(arr_ok["BLOCK_TIME_UTC_mvt"]) if arr_ok.height else np.array([], dtype=np.int64)

    dep_by: dict[str, np.ndarray] = defaultdict(list)
    arr_by: dict[str, np.ndarray] = defaultdict(list)
    for s, t in zip(dep_st, dep_t):
        dep_by[s].append(t)
    for s, t in zip(arr_st, arr_t):
        arr_by[s].append(t)
    dep_by = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in dep_by.items()}
    arr_by = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in arr_by.items()}

    has = np.zeros(n, dtype=np.int8)
    g_ub = np.full(n, np.nan)
    t_lb = np.full(n, np.nan)
    n_dep = np.zeros(n, dtype=np.float64)
    n_arr = np.zeros(n, dtype=np.float64)
    tv_hour = np.full(n, np.nan)

    for i in range(n):
        s = stands[i]
        a, b = sched[i], mvt[i]
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            continue
        dt = dep_by.get(s)
        at = arr_by.get(s)
        nd = 0
        na = 0
        first = None
        if dt is not None and dt.size:
            lo = int(np.searchsorted(dt, a, side="right"))
            hi = int(np.searchsorted(dt, b, side="left"))
            if hi > lo:
                nd = hi - lo
                first = int(dt[lo])
        if at is not None and at.size:
            lo = int(np.searchsorted(at, a, side="right"))
            hi = int(np.searchsorted(at, b, side="left"))
            if hi > lo:
                na = hi - lo
                t0 = int(at[lo])
                first = t0 if first is None else min(first, t0)
        n_dep[i] = nd
        n_arr[i] = na
        if first is not None:
            has[i] = 1
            g_ub[i] = (first - a) / 1e9
            t_lb[i] = (b - first) / 1e9
            tv_hour[i] = datetime.fromtimestamp(first / 1e9, tz=timezone.utc).hour

    remote = (
        dep["STAND_mvt"].fill_null("").cast(pl.String).str.starts_with("8").cast(pl.Int8).to_numpy()
    )
    d = np.maximum(dep["mvt_sched"].to_numpy().astype(float), 0.0)
    # missing t_v: no tighter bound than takeoff; G_ub := D, T_lb := 0
    g_ub_f = np.where(has == 1, g_ub, d)
    t_lb_f = np.where(has == 1, t_lb, 0.0)
    tv_h_f = np.where(has == 1, tv_hour, dep["hour"].to_numpy().astype(float))
    return dep.with_columns(
        [
            pl.Series("has_tv", has),
            pl.Series("G_ub", g_ub_f),
            pl.Series("T_lb", t_lb_f),
            pl.Series("n_dep_mvt", n_dep),
            pl.Series("n_arr_blk", n_arr),
            pl.Series("n_int", n_dep + n_arr),
            pl.Series("tv_hour", tv_h_f),
            pl.Series("stand_remote", remote),
        ]
    )


def cdf(df: pl.DataFrame, feats: list[str]) -> "object":
    x = df.select(feats + CAT).to_pandas()
    for c in CAT:
        x[c] = x[c].fillna("NA").astype(str)
    return x


def fit_cat(tr: pl.DataFrame, target: np.ndarray, feats: list[str]) -> CatBoostRegressor:
    m = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
        od_type="Iter",
        od_wait=80,
    )
    m.fit(cdf(tr, feats), target, cat_features=CAT)
    return m


def e20_blend(oof: pl.DataFrame) -> np.ndarray:
    return (
        0.456 * oof["pred_C"].to_numpy()
        + 0.053 * oof["pred_D"].to_numpy()
        + 0.491 * oof["pred_E"].to_numpy()
    )


def slice_rmse(y, p, d, lo=4000.0, hi=9000.0):
    m = (d > lo) & (d < hi) & np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return {"n": 0, "rmse": float("nan")}
    return {"n": int(m.sum()), "rmse": float(rmse(y[m], p[m]))}


def topk_sse_share(y, p, k=50):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    tot = float(e2.sum())
    if tot <= 0:
        return 0.0
    idx = np.argsort(e2)[::-1][:k]
    return float(e2[idx].sum() / tot)


def main():
    log("E45: load DEP + LIRF ARR...")
    dep_all = add_causal_rolling(load_dep()).with_columns(
        pl.col("FLIGHT_mvt").fill_null("").str.slice(0, 3).alias("flt_prefix")
    )
    arr_l = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select(["STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        .collect()
    )
    arr_surf = arr_l.select(
        [pl.lit("LIRF").alias("airport"), "MVT_TIME_UTC_mvt", pl.lit("").alias("RUNWAY_mvt")]
    ).sort("MVT_TIME_UTC_mvt")
    # surface needs RUNWAY; reload ARR runway
    arr_surf = (
        pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
        .filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .select([pl.lit("LIRF").alias("airport"), "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )
    lirf = dep_all.filter(pl.col("airport") == "LIRF")
    log(f"  LIRF DEP {lirf.height:,} unmatched {int(lirf['unmatched'].sum()):,} ARR {arr_l.height:,}")
    lirf = add_surface(lirf, arr_surf)
    lirf = enrich(lirf, lirf["mvt_sched"].to_numpy().astype(float))
    log("  stand-release features...")
    lirf = add_stand_release(lirf, arr_l)
    lirf = lirf.with_columns(
        (pl.col("mvt_sched") - pl.col("y")).alias("G"),
    )
    u_all = lirf.filter(pl.col("unmatched"))
    # full-year diagnostic (training description; not a fit)
    G = u_all["G"].to_numpy().astype(float)
    T = u_all["y"].to_numpy().astype(float)
    D = u_all["mvt_sched"].to_numpy().astype(float)
    hv = u_all["has_tv"].to_numpy().astype(bool)
    diag = {
        "n": int(u_all.height),
        "corr_T_D": float(np.corrcoef(T, D)[0, 1]),
        "corr_G_D": float(np.corrcoef(G, D)[0, 1]),
        "has_tv_rate": float(hv.mean()),
        "has": {
            "n": int(hv.sum()),
            "med_G": float(np.median(G[hv])) if hv.any() else None,
            "med_T": float(np.median(T[hv])) if hv.any() else None,
            "med_D": float(np.median(D[hv])) if hv.any() else None,
        },
        "no": {
            "n": int((~hv).sum()),
            "med_G": float(np.median(G[~hv])) if (~hv).any() else None,
            "med_T": float(np.median(T[~hv])) if (~hv).any() else None,
            "med_D": float(np.median(D[~hv])) if (~hv).any() else None,
        },
        "always_D_rmse": float(rmse(T, D)),
        "hard_mix_D_else_1100": float(rmse(T, np.where(hv, D, TPHYS_FALLBACK))),
    }
    sl = (D > 4000) & (D < 9000)
    diag["slice_4_9h"] = {
        "n": int(sl.sum()),
        "has_tv_rate": float(hv[sl].mean()) if sl.any() else None,
        "med_G_has": float(np.median(G[sl & hv])) if (sl & hv).any() else None,
        "med_G_no": float(np.median(G[sl & ~hv])) if (sl & ~hv).any() else None,
        "med_T_has": float(np.median(T[sl & hv])) if (sl & hv).any() else None,
        "med_T_no": float(np.median(T[sl & ~hv])) if (sl & ~hv).any() else None,
        "rmse_D": float(rmse(T[sl], D[sl])) if sl.any() else None,
        "rmse_mix": float(rmse(T[sl], np.where(hv[sl], D[sl], TPHYS_FALLBACK))) if sl.any() else None,
    }
    log(
        f"  diagnostic has_tv={diag['has_tv_rate']:.3f} "
        f"medG has/no {diag['has']['med_G']:.0f}/{diag['no']['med_G']:.0f} "
        f"slice mix {diag['slice_4_9h']['rmse_mix']:.0f} vs D {diag['slice_4_9h']['rmse_D']:.0f}"
    )

    payload = {"diagnostic": diag, "splits": {}}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        log(f"  split {split}...")
        tr = lirf.filter(~pl.col("month").is_in(months))
        va = lirf.filter(pl.col("month").is_in(months) & pl.col("unmatched"))
        tr_u = tr.filter(pl.col("unmatched"))
        Dt = tr_u["mvt_sched"].to_numpy().astype(float)
        yt = tr_u["y"].to_numpy().astype(float)
        Gt = Dt - yt
        D = va["mvt_sched"].to_numpy().astype(float)
        y = va["y"].to_numpy().astype(float)
        hv_va = va["has_tv"].to_numpy().astype(bool)
        t_phys = float(np.median(yt[tr_u["has_tv"].to_numpy() == 0])) if (tr_u["has_tv"] == 0).sum() else TPHYS_FALLBACK
        if not np.isfinite(t_phys):
            t_phys = TPHYS_FALLBACK

        # v9 baseline: unmatched-only CatBoost G, E35 features
        m_v9 = fit_cat(tr_u, Gt, FEATS_V9)
        g_v9 = np.asarray(m_v9.predict(cdf(va, FEATS_V9)), float)
        t_v9 = np.maximum(D - g_v9, 0.0)

        # A: + stand-release features
        m_a = fit_cat(tr_u, Gt, FEATS_A)
        g_a = np.asarray(m_a.predict(cdf(va, FEATS_A)), float)
        t_a = np.maximum(D - g_a, 0.0)
        imp = dict(zip(m_a.feature_names_, m_a.get_feature_importance()))
        rel_imp = {k: float(imp.get(k, 0.0)) for k in REL}

        # B: P(G<300) mixture
        z = (Gt < 300).astype(int)
        pcat = tr_u.select(["has_tv", "mvt_sched", "G_ub", "T_lb", "n_int", "hour", "stand_remote", "flt_prefix", "STAND_mvt"]).to_pandas()
        vcat = va.select(["has_tv", "mvt_sched", "G_ub", "T_lb", "n_int", "hour", "stand_remote", "flt_prefix", "STAND_mvt"]).to_pandas()
        for c in ("flt_prefix", "STAND_mvt"):
            pcat[c] = pcat[c].astype("category")
            vcat[c] = vcat[c].astype("category")
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(pcat, z, categorical_feature=["flt_prefix", "STAND_mvt"])
        p_hold = np.clip(clf.predict_proba(vcat)[:, 1], 0.0, 1.0)
        t_b = p_hold * t_phys + (1.0 - p_hold) * D
        t_b = np.maximum(t_b, 0.0)

        # C: hard gate
        t_c = np.where(hv_va, D, t_phys).astype(float)

        cands = {"v9": t_v9, "A": t_a, "B": t_b, "C": t_c, "D": D, "tphys": np.full_like(D, t_phys)}

        # overall via E20 OOF splice (LIRF unmatched only)
        oof = pl.read_parquet(HERE / "results" / "E20" / f"oof_predictions_{split}.parquet")
        e20 = e20_blend(oof)
        y_all = oof["y"].to_numpy().astype(float)
        um_all = oof["unmatched"].to_numpy().astype(bool)
        # airport may not be in oof — join ids
        id_va = va["MVT_ID_mvt"].to_numpy()
        id_oof = oof["MVT_ID_mvt"].to_numpy()
        pos = {int(i) if i == i else None: k for k, i in enumerate(id_oof)}
        idx = np.array([pos.get(int(i), -1) for i in id_va])
        if (idx < 0).any():
            raise RuntimeError(f"{split}: {int((idx < 0).sum())} val LIRF unmatched missing from E20 OOF")
        e20_overall = float(rmse(y_all, e20))
        # confirm E20 LIRF-u is ~D
        e20_lu = e20[idx]

        def overall_of(pred_lu):
            p = e20.copy()
            p[idx] = pred_lu
            return float(rmse(y_all, p)), float(np.sum((y_all - p) ** 2))

        out_cands = {}
        for name, pred in cands.items():
            ov, sse = overall_of(pred)
            out_cands[name] = {
                "lirf_u_rmse": float(rmse(y, pred)),
                "lirf_u_sse": float(np.sum((y - pred) ** 2)),
                "overall_rmse": ov,
                "overall_sse": sse,
                "slice_4_9h": slice_rmse(y, pred, D),
                "top50_sse_share": topk_sse_share(y, pred, 50),
                "mean_pred": float(np.mean(pred)),
                "med_pred": float(np.median(pred)),
            }
            log(
                f"    [{split}] {name}: LIRF_u {out_cands[name]['lirf_u_rmse']:.1f} "
                f"overall {ov:.2f} slice {out_cands[name]['slice_4_9h']['rmse']:.1f}"
            )

        # A with alpha grid (robustness, not selection on val if we freeze a=1)
        a_grid = {}
        for al in ALPHAS:
            p = np.maximum(D - al * g_a, 0.0)
            a_grid[str(al)] = {"lirf_u_rmse": float(rmse(y, p)), "overall_rmse": overall_of(p)[0]}

        payload["splits"][split] = {
            "n_val": int(va.height),
            "n_train_unm": int(tr_u.height),
            "t_phys": t_phys,
            "has_tv_rate": float(hv_va.mean()),
            "e20_overall": e20_overall,
            "e20_lirf_u": float(rmse(y, e20_lu)),
            "e20_vs_D": float(rmse(y, D)),
            "cands": out_cands,
            "A_alpha": a_grid,
            "A_release_importance": rel_imp,
            "B_p_hold_mean": float(p_hold.mean()),
            "B_p_hold_has": float(p_hold[hv_va].mean()) if hv_va.any() else None,
            "B_p_hold_no": float(p_hold[~hv_va].mean()) if (~hv_va).any() else None,
            "val_has": {
                "n": int(hv_va.sum()),
                "med_y": float(np.median(y[hv_va])) if hv_va.any() else None,
                "med_G": float(np.median((D - y)[hv_va])) if hv_va.any() else None,
            },
            "val_no": {
                "n": int((~hv_va).sum()),
                "med_y": float(np.median(y[~hv_va])) if (~hv_va).any() else None,
                "med_G": float(np.median((D - y)[~hv_va])) if (~hv_va).any() else None,
            },
        }

    # GO / NO-GO vs v9 (this run's v9, should match E35 ~3752 / 2557)
    jj = payload["splits"]["janjul"]["cands"]
    dc = payload["splits"]["dec"]["cands"]
    v9_jj = jj["v9"]["lirf_u_rmse"]
    v9_dc = dc["v9"]["lirf_u_rmse"]
    decision = {}
    for name in ("A", "B", "C"):
        d_jj = jj["v9"]["overall_rmse"] - jj[name]["overall_rmse"]
        d_lu_jj = v9_jj - jj[name]["lirf_u_rmse"]
        d_lu_dc = v9_dc - dc[name]["lirf_u_rmse"]
        sl_v9 = jj["v9"]["slice_4_9h"]["rmse"]
        sl_c = jj[name]["slice_4_9h"]["rmse"]
        sl_cut = (sl_v9 - sl_c) / sl_v9 if sl_v9 and sl_v9 == sl_v9 else float("nan")
        go = (
            jj[name]["lirf_u_rmse"] < v9_jj
            and dc[name]["lirf_u_rmse"] < v9_dc
            and d_jj >= 8.0
            and sl_cut >= 0.10
        )
        nogo_dec = dc[name]["lirf_u_rmse"] >= v9_dc
        nogo_small = d_jj < 5.0
        decision[name] = {
            "janjul_overall_drop": d_jj,
            "janjul_lirf_u_drop": d_lu_jj,
            "dec_lirf_u_drop": d_lu_dc,
            "slice_rel_cut": sl_cut,
            "GO": bool(go),
            "NOGO_dec_up": bool(nogo_dec),
            "NOGO_small_jj": bool(nogo_small),
        }
        log(
            f"  DECISION {name}: GO={go} jj_overall {d_jj:+.2f}s "
            f"LIRF_u {d_lu_jj:+.1f}/{d_lu_dc:+.1f} slice_cut {sl_cut:.1%}"
        )

    payload["decision"] = decision
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (RES / "E45_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_report(payload)
    log("WROTE " + str(RES / "E45_results.json"))


def write_report(p: dict) -> None:
    d = p["diagnostic"]
    sl = d["slice_4_9h"]
    lines = [
        "# E45 — stand-release G-regime (LIRF unmatched)",
        "",
        "Baseline = this run's unmatched-only CatBoost G (E35 / v9 recipe).",
        "Production v9 LB **288.9003**. Identity `T = D − G`.",
        "",
        "## Diagnostic (full-year unmatched LIRF, not a fit)",
        "",
        f"- n={d['n']}, has_tv rate={d['has_tv_rate']:.3f}, corr(T,D)={d['corr_T_D']:.3f}, corr(G,D)={d['corr_G_D']:.3f}",
        f"- reused: n={d['has']['n']}, med G={d['has']['med_G']:.0f}, med T={d['has']['med_T']:.0f}, med D={d['has']['med_D']:.0f}",
        f"- not reused: n={d['no']['n']}, med G={d['no']['med_G']:.0f}, med T={d['no']['med_T']:.0f}, med D={d['no']['med_D']:.0f}",
        f"- always D RMSE={d['always_D_rmse']:.0f}; hard mix D-if-reused else 1100 RMSE={d['hard_mix_D_else_1100']:.0f}",
        f"- D in (4000,9000): n={sl['n']}, med G has/no={sl['med_G_has']:.0f}/{sl['med_G_no']:.0f}, "
        f"med T has/no={sl['med_T_has']:.0f}/{sl['med_T_no']:.0f}, RMSE D={sl['rmse_D']:.0f}, mix={sl['rmse_mix']:.0f}",
        "",
        "## Holdout results",
        "",
        "| split | cand | LIRF_u RMSE | overall RMSE | 4–9h D slice | top-50 SSE share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("janjul", "dec"):
        s = p["splits"][split]
        for name, c in s["cands"].items():
            slr = c["slice_4_9h"]["rmse"]
            lines.append(
                f"| {split} | {name} | {c['lirf_u_rmse']:.1f} | {c['overall_rmse']:.2f} | "
                f"{slr:.1f} | {c['top50_sse_share']:.3f} |"
            )
    lines += ["", "## Feature importance (A, stand-release only)", ""]
    for split in ("janjul", "dec"):
        lines.append(f"- {split}: `{p['splits'][split]['A_release_importance']}`")
    lines += ["", "## GO / NO-GO", ""]
    any_go = False
    for name, dec in p["decision"].items():
        mark = "GO" if dec["GO"] else "NO-GO"
        if dec["GO"]:
            any_go = True
        lines.append(
            f"- **{name}: {mark}** — Jan+Jul overall {dec['janjul_overall_drop']:+.2f} s, "
            f"LIRF_u {dec['janjul_lirf_u_drop']:+.1f} / Dec {dec['dec_lirf_u_drop']:+.1f}, "
            f"slice cut {dec['slice_rel_cut']:.1%}"
        )
    lines += [
        "",
        "Rule: GO if LIRF_u drops on both splits, Jan+Jul overall drop ≥8 s, and the 4–9h slice "
        "RMSE cut vs v9 is ≥10%. NO-GO if December LIRF_u rises or Jan+Jul overall drop <5 s.",
        "",
        f"**Verdict: {'GO — splice LIRF unmatched into v9' if any_go else 'NO-GO — do not submit; Direction 2 (OpenSky) is next.'}**",
        "",
        "Artifacts: `experiments/run_e45_stand_release.py`, `experiments/results/E45/`.",
        "",
    ]
    (RES / "E45_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
