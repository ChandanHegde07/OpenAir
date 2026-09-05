"""E2b: winning clocks + airport calibration. E3: stand/runway geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AIRPORTS,
    add_causal_rolling,
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
    load_dep,
    metrics_block,
    ols_predict,
    save_result,
    split_by_months,
)


def evaluate(name, y, p, unmatched, airport, fh):
    m = metrics_block(y, p, unmatched, airport)
    print(f"  {name:48s} {fmt(m)}", flush=True)
    fh.write(f"  {name:48s} {fmt(m)}\n")
    return m


def per_airport_ols(train, val, cols, min_n=200):
    """Per-airport OLS on cols; fallback to global OLS on same cols."""
    trm = train
    for c in cols:
        trm = trm.filter(pl.col(c).is_not_null())
    trm = trm.filter(~pl.col("unmatched")) if "mvt_aobt" in cols else trm
    xg = np.column_stack([trm[c].to_numpy().astype(float) for c in cols])
    glob, gcoef = ols_predict(xg, trm["y"].to_numpy(), np.column_stack([val[c].to_numpy().astype(float) for c in cols]))
    pred = glob.copy()
    used = 0
    airport = val["airport"].to_numpy()
    for ap in AIRPORTS:
        tr_ap = trm.filter(pl.col("airport") == ap)
        if tr_ap.height < min_n:
            continue
        xva = np.column_stack([val[c].to_numpy().astype(float) for c in cols])
        p_ap, _ = ols_predict(
            np.column_stack([tr_ap[c].to_numpy().astype(float) for c in cols]),
            tr_ap["y"].to_numpy(),
            xva,
        )
        mask = airport == ap
        pred[mask] = p_ap[mask]
        used += 1
    return pred, glob, gcoef, used


def geometry_tables(train: pl.DataFrame) -> dict[str, pl.DataFrame]:
    y = pl.col("y")
    aggs = [
        y.len().alias("n"),
        y.mean().alias("mean"),
        y.median().alias("med"),
        y.std().alias("std"),
        y.quantile(0.25).alias("p25"),
        y.quantile(0.75).alias("p75"),
        y.quantile(0.90).alias("p90"),
    ]
    sr = train.group_by(["airport", "STAND_mvt", "RUNWAY_mvt"]).agg(aggs)
    st = train.group_by(["airport", "STAND_mvt"]).agg(aggs)
    rw = train.group_by(["airport", "RUNWAY_mvt"]).agg(aggs)
    ap = train.group_by(["airport"]).agg(aggs)
    return {"sr": sr, "st": st, "rw": rw, "ap": ap}


def attach_geometry(val: pl.DataFrame, tabs: dict, min_n: int) -> pl.DataFrame:
    def prep(t, prefix):
        cols = {c: f"{prefix}_{c}" for c in t.columns if c not in ("airport", "STAND_mvt", "RUNWAY_mvt")}
        return t.rename(cols)

    out = (
        val.join(prep(tabs["sr"], "sr"), on=["airport", "STAND_mvt", "RUNWAY_mvt"], how="left")
        .join(prep(tabs["st"], "st"), on=["airport", "STAND_mvt"], how="left")
        .join(prep(tabs["rw"], "rw"), on=["airport", "RUNWAY_mvt"], how="left")
        .join(prep(tabs["ap"], "ap"), on=["airport"], how="left")
    )
    # hierarchical expected taxi: sr if n>=min_n else stand else rwy else airport
    sr_ok = (pl.col("sr_n") >= min_n) & pl.col("sr_mean").is_not_null()
    st_ok = (pl.col("st_n") >= min_n) & pl.col("st_mean").is_not_null()
    rw_ok = (pl.col("rw_n") >= min_n) & pl.col("rw_mean").is_not_null()
    out = out.with_columns(
        pl.when(sr_ok)
        .then(pl.col("sr_mean"))
        .when(st_ok)
        .then(pl.col("st_mean"))
        .when(rw_ok)
        .then(pl.col("rw_mean"))
        .otherwise(pl.col("ap_mean"))
        .alias("geo_mean"),
        pl.when(sr_ok)
        .then(pl.col("sr_med"))
        .when(st_ok)
        .then(pl.col("st_med"))
        .when(rw_ok)
        .then(pl.col("rw_med"))
        .otherwise(pl.col("ap_med"))
        .alias("geo_med"),
        pl.when(sr_ok).then(pl.col("sr_p90")).when(st_ok).then(pl.col("st_p90")).otherwise(pl.col("ap_p90")).alias("geo_p90"),
        pl.when(sr_ok).then(pl.col("sr_std")).otherwise(pl.col("ap_std")).alias("geo_std"),
        pl.when(sr_ok).then(pl.col("sr_n")).otherwise(0).alias("sr_n_used"),
    )
    out = out.with_columns(
        (pl.col("sr_mean") - pl.col("st_mean")).alias("sr_minus_stand"),
        (pl.col("sr_mean") - pl.col("rw_mean")).alias("sr_minus_rwy"),
        (pl.col("sr_mean") - pl.col("ap_mean")).alias("sr_minus_ap"),
        (pl.col("mvt_aobt") - pl.col("geo_mean")).alias("aobt_minus_geo"),
        (pl.col("sr_p75") - pl.col("sr_p25")).alias("sr_iqr"),
    )
    return out


def main():
    out_path = Path(__file__).resolve().parent / "results" / "E2b_E3.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        print("loading...", flush=True)
        df = add_causal_rolling(load_dep())
        print(f"loaded {df.height:,}", flush=True)

        payload = {"E2b": {}, "E3": {}}

        print("\n========== E2b CLOCKS + AIRPORT CALIBRATION ==========", flush=True)
        fh.write("========== E2b CLOCKS + AIRPORT CALIBRATION ==========\n")
        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr, va = split_by_months(df, months)
            y = va["y"].to_numpy()
            um = va["unmatched"].to_numpy()
            ap = va["airport"].to_numpy()
            fb = airport_mean_fallback(tr, va)
            print(f"\n--- {split_name} ---", flush=True)
            fh.write(f"\n--- {split_name} ---\n")
            block = {}

            for label, cols in [
                ("mvt_aobt", ["mvt_aobt"]),
                ("aobt+eobt", ["mvt_aobt", "aobt_eobt"]),
                ("aobt+eobt+roll10", ["mvt_aobt", "aobt_eobt", "roll10_mean_mvt_aobt"]),
                ("aobt+eobt+family_as_dummy_skip", None),
            ]:
                if cols is None:
                    continue
                pred_ap, pred_g, coef, nused = per_airport_ols(tr, va, cols)
                block[f"global_{label}"] = evaluate(
                    f"global {label}", y, fill_with_fallback(pred_g, fb), um, ap, fh
                )
                block[f"global_{label}"]["coef"] = [float(c) for c in coef]
                block[f"airport_{label}"] = evaluate(
                    f"per-airport {label}", y, fill_with_fallback(pred_ap, fb), um, ap, fh
                )
                print(f"    global coef={coef}  airports_used={nused}", flush=True)
                fh.write(f"    global coef={coef}  airports_used={nused}\n")

            # fallback chain: 3feat if roll10 finite else 2feat else ap mean
            p3, g3, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "roll10_mean_mvt_aobt"])
            p2, g2, _, _ = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt"])
            chained = fill_with_fallback(p3, p2)
            block["airport_eobt_roll_chain"] = evaluate(
                "airport aobt+eobt+roll10 chain to 2feat", y, fill_with_fallback(chained, fb), um, ap, fh
            )
            chained_g = fill_with_fallback(g3, g2)
            block["global_eobt_roll_chain"] = evaluate(
                "global aobt+eobt+roll10 chain to 2feat", y, fill_with_fallback(chained_g, fb), um, ap, fh
            )
            payload["E2b"][split_name] = block

        print("\n========== E3 STAND/RUNWAY GEOMETRY ==========", flush=True)
        fh.write("\n========== E3 STAND/RUNWAY GEOMETRY ==========\n")
        for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
            tr, va = split_by_months(df, months)
            tabs = geometry_tables(tr)
            va_g = attach_geometry(va, tabs, min_n=30)
            tr_g = attach_geometry(tr, tabs, min_n=30)
            y = va_g["y"].to_numpy()
            um = va_g["unmatched"].to_numpy()
            ap = va_g["airport"].to_numpy()
            fb = airport_mean_fallback(tr, va_g)
            print(f"\n--- {split_name} ---", flush=True)
            fh.write(f"\n--- {split_name} ---\n")
            block = {}

            geo = va_g["geo_mean"].to_numpy().astype(float)
            geo_med = va_g["geo_med"].to_numpy().astype(float)
            sr_raw = va_g["sr_mean"].to_numpy().astype(float)
            block["geo_mean_hier"] = evaluate("geo mean hierarchy n>=30", y, fill_with_fallback(geo, fb), um, ap, fh)
            block["geo_med_hier"] = evaluate("geo median hierarchy n>=30", y, fill_with_fallback(geo_med, fb), um, ap, fh)
            block["sr_mean_raw"] = evaluate("raw stand-rwy mean (nan if unseen)", y, sr_raw, um, ap, fh)
            # unmatched-only geometry
            if um.any():
                block["geo_on_unmatched"] = evaluate(
                    "geo mean on unmatched only", y[um], geo[um], um[um], ap[um], fh
                )

            # OLS y ~ geo
            trc = tr_g.filter(pl.col("geo_mean").is_not_null())
            pgeo, coef = ols_predict(trc["geo_mean"].to_numpy(), trc["y"].to_numpy(), geo)
            block["ols_geo"] = evaluate("OLS geo_mean", y, fill_with_fallback(pgeo, fb), um, ap, fh)

            # OLS y ~ mvt_aobt + geo
            for label, cols in [
                ("aobt+geo", ["mvt_aobt", "geo_mean"]),
                ("aobt+eobt+geo", ["mvt_aobt", "aobt_eobt", "geo_mean"]),
                ("aobt+eobt+geo+roll10", ["mvt_aobt", "aobt_eobt", "geo_mean", "roll10_mean_mvt_aobt"]),
                ("aobt+eobt+sr_minus_ap", ["mvt_aobt", "aobt_eobt", "sr_minus_ap"]),
                ("aobt+eobt+aobt_minus_geo", ["mvt_aobt", "aobt_eobt", "aobt_minus_geo"]),
                ("aobt+eobt+geo+geo_p90", ["mvt_aobt", "aobt_eobt", "geo_mean", "geo_p90"]),
                ("aobt+eobt+geo+sr_minus_rwy", ["mvt_aobt", "aobt_eobt", "geo_mean", "sr_minus_rwy"]),
            ]:
                trc = tr_g
                for c in cols:
                    trc = trc.filter(pl.col(c).is_not_null())
                pred, coef = ols_predict(
                    np.column_stack([trc[c].to_numpy().astype(float) for c in cols]),
                    trc["y"].to_numpy(),
                    np.column_stack([va_g[c].to_numpy().astype(float) for c in cols]),
                )
                # chain: if combo nan (unmatched), use geo (available without AOBT)
                chained = fill_with_fallback(pred, geo)
                block[label] = evaluate(label, y, fill_with_fallback(chained, fb), um, ap, fh)
                block[label]["coef"] = [float(c) for c in coef]
                print(f"    coef={coef}", flush=True)
                fh.write(f"    coef={coef}\n")

            # per-airport OLS aobt+eobt+geo
            p_ap, p_g, coef, nused = per_airport_ols(
                tr_g, va_g, ["mvt_aobt", "aobt_eobt", "geo_mean"]
            )
            chained = fill_with_fallback(p_ap, geo)
            block["airport_aobt_eobt_geo"] = evaluate(
                "per-airport aobt+eobt+geo", y, fill_with_fallback(chained, fb), um, ap, fh
            )
            print(f"    global-of-this-spec coef={coef}", flush=True)

            payload["E3"][split_name] = block

        save_result("E2b_E3", payload)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
