"""E0 baseline verification, E1 clock relationships, E2 AOBT calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    add_causal_rolling,
    airport_mean_fallback,
    fmt,
    load_dep,
    metrics_block,
    ols_predict,
    save_result,
    split_by_months,
    fill_with_fallback,
    SPLITS,
)


def evaluate(name, y, p, unmatched, airport, fh):
    m = metrics_block(y, p, unmatched, airport)
    print(f"  {name:40s} {fmt(m)}", flush=True)
    fh.write(f"  {name:40s} {fmt(m)}\n")
    return m


def clip_pred(p, lo=0.0, hi=7200.0):
    p = np.asarray(p, dtype=np.float64)
    out = p.copy()
    m = np.isfinite(out)
    out[m] = np.clip(out[m], lo, hi)
    return out


def run_e0(df, fh):
    print("\n========== E0 BASELINE VERIFICATION ==========", flush=True)
    fh.write("\n========== E0 BASELINE VERIFICATION ==========\n")
    payload = {"id": "E0", "experiments": {}}

    for split_name, months in SPLITS.items():
        tr, va = split_by_months(df, months)
        y = va["y"].to_numpy()
        unmatched = va["unmatched"].to_numpy()
        airport = va["airport"].to_numpy()
        ap_fb = airport_mean_fallback(tr, va)
        glob = np.full_like(y, float(tr["y"].mean()))
        proxy = va["mvt_aobt"].to_numpy().astype(np.float64)
        roll = va["roll10_mean_mvt_aobt"].to_numpy().astype(np.float64)

        print(f"\n--- split={split_name} train={tr.height:,} val={va.height:,} ---", flush=True)
        fh.write(f"\n--- split={split_name} train={tr.height:,} val={va.height:,} ---\n")
        block = {}
        block["global_mean"] = evaluate("global_mean", y, glob, unmatched, airport, fh)
        block["airport_mean"] = evaluate("airport_mean", y, ap_fb, unmatched, airport, fh)
        block["aobt_proxy_matched_only"] = evaluate("aobt_proxy (nan dropped)", y, proxy, unmatched, airport, fh)
        block["aobt_proxy_plus_apfb"] = evaluate(
            "aobt_proxy else airport_mean", y, fill_with_fallback(proxy, ap_fb), unmatched, airport, fh
        )
        block["aobt_proxy_clip_apfb"] = evaluate(
            "clip(aobt_proxy,0,7200) else ap_mean",
            y,
            fill_with_fallback(clip_pred(proxy), ap_fb),
            unmatched,
            airport,
            fh,
        )

        pred_ols, coef = ols_predict(tr.filter(~pl.col("unmatched"))["mvt_aobt"].to_numpy(), tr.filter(~pl.col("unmatched"))["y"].to_numpy(), proxy)
        block["ols_aobt"] = evaluate("OLS aobt else ap_mean", y, fill_with_fallback(pred_ols, ap_fb), unmatched, airport, fh)
        block["ols_aobt_coef"] = [float(c) for c in coef]

        trm = tr.filter((~pl.col("unmatched")) & pl.col("roll10_mean_mvt_aobt").is_not_null())
        xtr = np.column_stack(
            [
                trm["mvt_aobt"].to_numpy().astype(float),
                trm["roll10_mean_mvt_aobt"].to_numpy().astype(float),
            ]
        )
        pred2, coef2 = ols_predict(xtr, trm["y"].to_numpy(), np.column_stack([proxy, roll]))
        block["ols_aobt_roll10"] = evaluate(
            "OLS aobt+roll10 else ap_mean", y, fill_with_fallback(pred2, ap_fb), unmatched, airport, fh
        )
        block["ols_aobt_roll10_coef"] = [float(c) for c in coef2]
        block["ols_aobt_roll10_matched"] = evaluate("OLS aobt+roll10 matched-only", y, pred2, unmatched, airport, fh)

        payload["experiments"][split_name] = block
        print(f"  OLS aobt coef={coef}", flush=True)
        print(f"  OLS aobt+roll10 coef={coef2}", flush=True)
        fh.write(f"  OLS aobt coef={coef}\n")
        fh.write(f"  OLS aobt+roll10 coef={coef2}\n")

    save_result("E0", payload)
    return payload


def run_e1(df, fh):
    print("\n========== E1 CLOCK RELATIONSHIPS ==========", flush=True)
    fh.write("\n========== E1 CLOCK RELATIONSHIPS ==========\n")
    clock_cols = [
        "mvt_aobt",
        "mvt_eobt",
        "mvt_iobt",
        "mvt_lobt",
        "mvt_sched",
        "aobt_eobt",
        "aobt_iobt",
        "aobt_lobt",
        "aobt_sched",
        "eobt_iobt",
        "eobt_lobt",
        "iobt_lobt",
        "clock_std",
        "clock_range",
        "abs_aobt_eobt",
        "abs_aobt_lobt",
        "n_clocks",
    ]
    payload = {"id": "E1", "univariate_raw": {}, "univariate_ols": {}, "combos": {}}

    # Primary split
    tr, va = split_by_months(df, SPLITS["janjul"])
    y = va["y"].to_numpy()
    unmatched = va["unmatched"].to_numpy()
    airport = va["airport"].to_numpy()
    ap_fb = airport_mean_fallback(tr, va)

    print("\n--- Jan+Jul univariate RAW clock as prediction ---", flush=True)
    fh.write("\n--- Jan+Jul univariate RAW clock as prediction ---\n")
    for c in clock_cols:
        raw = va[c].to_numpy().astype(float)
        payload["univariate_raw"][c] = evaluate(f"RAW {c}", y, raw, unmatched, airport, fh)

    print("\n--- Jan+Jul univariate OLS clock -> y (matched fit, ap fallback) ---", flush=True)
    fh.write("\n--- Jan+Jul univariate OLS clock -> y ---\n")
    for c in clock_cols:
        trc = tr.filter(pl.col(c).is_not_null())
        pred, coef = ols_predict(trc[c].to_numpy(), trc["y"].to_numpy(), va[c].to_numpy().astype(float))
        m = evaluate(f"OLS {c}", y, fill_with_fallback(pred, ap_fb), unmatched, airport, fh)
        m["coef"] = [float(x) for x in coef]
        payload["univariate_ols"][c] = m

    print("\n--- Jan+Jul clock combinations (OLS) ---", flush=True)
    fh.write("\n--- Jan+Jul clock combinations (OLS) ---\n")
    combos = {
        "aobt+aobt_eobt": ["mvt_aobt", "aobt_eobt"],
        "aobt+aobt_lobt": ["mvt_aobt", "aobt_lobt"],
        "aobt+clock_std": ["mvt_aobt", "clock_std"],
        "aobt+clock_range": ["mvt_aobt", "clock_range"],
        "aobt+abs_aobt_eobt": ["mvt_aobt", "abs_aobt_eobt"],
        "aobt+mvt_eobt": ["mvt_aobt", "mvt_eobt"],
        "aobt+aobt_eobt+clock_std": ["mvt_aobt", "aobt_eobt", "clock_std"],
        "aobt+roll10": ["mvt_aobt", "roll10_mean_mvt_aobt"],
        "aobt+roll10+clock_std": ["mvt_aobt", "roll10_mean_mvt_aobt", "clock_std"],
        "aobt+roll10+aobt_eobt": ["mvt_aobt", "roll10_mean_mvt_aobt", "aobt_eobt"],
    }
    for name, cols in combos.items():
        trc = tr
        for c in cols:
            trc = trc.filter(pl.col(c).is_not_null())
        xtr = np.column_stack([trc[c].to_numpy().astype(float) for c in cols])
        xva = np.column_stack([va[c].to_numpy().astype(float) for c in cols])
        pred, coef = ols_predict(xtr, trc["y"].to_numpy(), xva)
        m = evaluate(name, y, fill_with_fallback(pred, ap_fb), unmatched, airport, fh)
        m["coef"] = [float(x) for x in coef]
        payload["combos"][name] = m
        print(f"    coef={coef}", flush=True)
        fh.write(f"    coef={coef}\n")

    # December sanity for the winning combo vs aobt-only
    print("\n--- December sanity for selected combos ---", flush=True)
    fh.write("\n--- December sanity ---\n")
    payload["dec"] = {}
    trd, vad = split_by_months(df, SPLITS["dec"])
    yd = vad["y"].to_numpy()
    umd = vad["unmatched"].to_numpy()
    apd = vad["airport"].to_numpy()
    fb = airport_mean_fallback(trd, vad)
    for name, cols in {
        "ols_aobt": ["mvt_aobt"],
        "aobt+roll10": ["mvt_aobt", "roll10_mean_mvt_aobt"],
        "aobt+roll10+aobt_eobt": ["mvt_aobt", "roll10_mean_mvt_aobt", "aobt_eobt"],
        "aobt+clock_std": ["mvt_aobt", "clock_std"],
    }.items():
        trc = trd
        for c in cols:
            trc = trc.filter(pl.col(c).is_not_null())
        pred, coef = ols_predict(
            np.column_stack([trc[c].to_numpy().astype(float) for c in cols]),
            trc["y"].to_numpy(),
            np.column_stack([vad[c].to_numpy().astype(float) for c in cols]),
        )
        payload["dec"][name] = evaluate(name, yd, fill_with_fallback(pred, fb), umd, apd, fh)

    save_result("E1", payload)
    return payload


def _calibrated_group(train, val, keys, min_n, k_shrink, parent_pred_val, parent_pred_tr=None):
    """Fit y ~ a + b * mvt_aobt per group with shrinkage toward parent predictions."""
    trm = train.filter(~pl.col("unmatched"))
    stats = trm.group_by(keys).agg(
        pl.len().alias("n"),
        pl.col("y").mean().alias("y_mean"),
        pl.col("mvt_aobt").mean().alias("x_mean"),
        pl.col("y").std().alias("y_std"),
        pl.col("mvt_aobt").std().alias("x_std"),
        (pl.col("y") * pl.col("mvt_aobt")).mean().alias("xy_mean"),
    )
    # slope = cov(x,y)/var(x)
    stats = stats.with_columns(
        pl.when((pl.col("n") >= min_n) & (pl.col("x_std") > 1e-6))
        .then(
            (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean"))
            / (pl.col("x_std") ** 2 * (pl.col("n") - 1) / pl.col("n"))
        )
        .otherwise(None)
        .alias("b_raw")
    )
    # Use numpy per-group OLS for correctness
    yv = val["y"].to_numpy()
    xv = val["mvt_aobt"].to_numpy().astype(float)
    pred = np.array(parent_pred_val, dtype=float).copy()
    # map group key -> (a,b,n)
    key_df = trm.select(keys + ["mvt_aobt", "y"]).drop_nulls(["mvt_aobt", "y"])
    groups = {}
    if len(keys) == 1:
        grouped = key_df.partition_by(keys, as_dict=True)
        for k, g in grouped.items():
            kk = k[0] if isinstance(k, tuple) else k
            groups[(kk,)] = g
    else:
        grouped = key_df.partition_by(keys, as_dict=True)
        for k, g in grouped.items():
            groups[k if isinstance(k, tuple) else (k,)] = g

    coefs = {}
    for k, g in groups.items():
        x = g["mvt_aobt"].to_numpy().astype(float)
        yy = g["y"].to_numpy().astype(float)
        n = len(x)
        if n < min_n or np.std(x) < 1e-6:
            continue
        X = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        coefs[k] = (float(coef[0]), float(coef[1]), n)

    # apply
    val_keys = [val[c].to_numpy() for c in keys]
    local = np.full(len(yv), np.nan)
    ns = np.zeros(len(yv))
    for i in range(len(yv)):
        k = tuple(val_keys[j][i] for j in range(len(keys)))
        if k in coefs and np.isfinite(xv[i]):
            a, b, n = coefs[k]
            local[i] = a + b * xv[i]
            ns[i] = n
    w = ns / (ns + k_shrink)
    w = np.where(np.isfinite(local) & (ns >= min_n), w, 0.0)
    out = w * local + (1.0 - w) * parent_pred_val
    out = np.where(np.isfinite(xv), out, np.nan)
    return out, {"n_groups": len(coefs), "min_n": min_n, "k_shrink": k_shrink}


def run_e2(df, fh):
    print("\n========== E2 AOBT CALIBRATION ==========", flush=True)
    fh.write("\n========== E2 AOBT CALIBRATION ==========\n")
    payload = {"id": "E2", "splits": {}}

    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr, va = split_by_months(df, months)
        y = va["y"].to_numpy()
        unmatched = va["unmatched"].to_numpy()
        airport = va["airport"].to_numpy()
        ap_fb = airport_mean_fallback(tr, va)
        proxy = va["mvt_aobt"].to_numpy().astype(float)
        trm = tr.filter(~pl.col("unmatched"))

        print(f"\n--- split={split_name} ---", flush=True)
        fh.write(f"\n--- split={split_name} ---\n")
        block = {}

        # global OLS
        gpred, gcoef = ols_predict(trm["mvt_aobt"].to_numpy(), trm["y"].to_numpy(), proxy)
        block["global_ols"] = evaluate("global OLS", y, fill_with_fallback(gpred, ap_fb), unmatched, airport, fh)
        block["global_ols"]["coef"] = [float(c) for c in gcoef]

        # airport OLS (no shrink)
        ap_pred, meta = _calibrated_group(tr, va, ["airport"], min_n=100, k_shrink=0, parent_pred_val=gpred)
        block["airport_ols_noshink"] = evaluate(
            "airport OLS no-shrink", y, fill_with_fallback(ap_pred, ap_fb), unmatched, airport, fh
        )
        block["airport_ols_noshink"]["meta"] = meta

        ap_pred_s, meta = _calibrated_group(tr, va, ["airport"], min_n=100, k_shrink=200, parent_pred_val=gpred)
        block["airport_ols_k200"] = evaluate(
            "airport OLS k=200", y, fill_with_fallback(ap_pred_s, ap_fb), unmatched, airport, fh
        )

        # airport additive bias only: y_hat = proxy + mean(y-proxy)_airport
        bias = trm.group_by("airport").agg((pl.col("y") - pl.col("mvt_aobt")).mean().alias("bias"))
        j = va.join(bias, on="airport", how="left")
        bias_pred = proxy + j["bias"].to_numpy().astype(float)
        block["airport_additive_bias"] = evaluate(
            "airport additive bias", y, fill_with_fallback(bias_pred, ap_fb), unmatched, airport, fh
        )

        # airport x WTC
        wtc_pred, meta = _calibrated_group(
            tr, va, ["airport", "WK_TBL_CAT_flt"], min_n=200, k_shrink=300, parent_pred_val=ap_pred
        )
        block["airport_wtc"] = evaluate(
            "airport x WTC shrink to airport", y, fill_with_fallback(wtc_pred, ap_fb), unmatched, airport, fh
        )
        block["airport_wtc"]["meta"] = meta

        # airport x ac_family
        fam_pred, meta = _calibrated_group(
            tr, va, ["airport", "ac_family"], min_n=200, k_shrink=300, parent_pred_val=ap_pred
        )
        block["airport_family"] = evaluate(
            "airport x ac_family shrink to airport", y, fill_with_fallback(fam_pred, ap_fb), unmatched, airport, fh
        )
        block["airport_family"]["meta"] = meta

        # airport x runway
        rwy_pred, meta = _calibrated_group(
            tr, va, ["airport", "RUNWAY_mvt"], min_n=200, k_shrink=300, parent_pred_val=ap_pred
        )
        block["airport_rwy"] = evaluate(
            "airport x runway shrink to airport", y, fill_with_fallback(rwy_pred, ap_fb), unmatched, airport, fh
        )
        block["airport_rwy"]["meta"] = meta

        # airport OLS + roll10
        roll = va["roll10_mean_mvt_aobt"].to_numpy().astype(float)
        xtr = np.column_stack(
            [
                tr.filter((~pl.col("unmatched")) & pl.col("roll10_mean_mvt_aobt").is_not_null())["mvt_aobt"].to_numpy().astype(float),
                tr.filter((~pl.col("unmatched")) & pl.col("roll10_mean_mvt_aobt").is_not_null())["roll10_mean_mvt_aobt"].to_numpy().astype(float),
            ]
        )
        ytr = tr.filter((~pl.col("unmatched")) & pl.col("roll10_mean_mvt_aobt").is_not_null())["y"].to_numpy()
        # per-airport two-feature OLS
        ap_roll_pred = np.full(len(y), np.nan)
        n_ap = 0
        for ap in sorted(set(tr["airport"].unique().to_list())):
            tr_ap = tr.filter(
                (pl.col("airport") == ap)
                & (~pl.col("unmatched"))
                & pl.col("roll10_mean_mvt_aobt").is_not_null()
            )
            if tr_ap.height < 200:
                continue
            pred_ap, _ = ols_predict(
                np.column_stack(
                    [
                        tr_ap["mvt_aobt"].to_numpy().astype(float),
                        tr_ap["roll10_mean_mvt_aobt"].to_numpy().astype(float),
                    ]
                ),
                tr_ap["y"].to_numpy(),
                np.column_stack([proxy, roll]),
            )
            mask = airport == ap
            ap_roll_pred[mask] = pred_ap[mask]
            n_ap += 1
        # fallback to global aobt+roll10
        glob_roll, coef = ols_predict(xtr, ytr, np.column_stack([proxy, roll]))
        mixed = fill_with_fallback(ap_roll_pred, glob_roll)
        block["airport_ols_plus_roll10"] = evaluate(
            "per-airport OLS aobt+roll10", y, fill_with_fallback(mixed, ap_fb), unmatched, airport, fh
        )
        block["global_ols_plus_roll10"] = evaluate(
            "global OLS aobt+roll10", y, fill_with_fallback(glob_roll, ap_fb), unmatched, airport, fh
        )

        payload["splits"][split_name] = block

    save_result("E2", payload)
    return payload


def main():
    out_path = Path(__file__).resolve().parent / "results" / "E0_E2.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        print("loading DEP...", flush=True)
        df = load_dep()
        print(f"loaded {df.height:,} DEP", flush=True)
        df = add_causal_rolling(df)
        print("rolling attached", flush=True)
        e0 = run_e0(df, fh)
        e1 = run_e1(df, fh)
        e2 = run_e2(df, fh)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
