"""AOBT residual, leak-free taxi-in rolling, geometry, unmatched flights, time-split validation."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data")
OUT = Path("analysis/output")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_FILES = sorted(glob.glob(str(DATA / "training_*.parquet")))
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def log(fh, *args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    fh.write(msg + "\n")


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), int(m.sum())


def mae(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))), int(m.sum())


def load():
    cols = [
        "MVT_ID_mvt",
        "FLIGHT_ID_mvt",
        "PHASE_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "RUNWAY_mvt",
        "STAND_mvt",
        "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt",
        "MARKET_SEGMENT_flt",
        "AIRCRAFT_OPERATOR_flt",
        "AOBT_3_flt",
        "EOBT_1_flt",
        "LOBT_flt",
        "ARVT_3_flt",
        "CALLSIGN_flt",
    ]
    return pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES]).collect()


def airport_of(df):
    return df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("airport")
    )


def main():
    out = OUT / "04_residual_validation.txt"
    with open(out, "w", encoding="utf-8") as fh:
        log(fh, "loading")
        df = load()
        dep = airport_of(df.filter(pl.col("PHASE_mvt") == "DEP"))
        arr = airport_of(df.filter(pl.col("PHASE_mvt") == "ARR"))

        dep = dep.with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("aobt_proxy"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("block_minus_aobt"),
            (pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).alias("y"),
            pl.col("AOBT_3_flt").is_null().alias("unmatched"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
            pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("sched_hour"),
        )
        dep = dep.with_columns((pl.col("y") - pl.col("aobt_proxy")).alias("resid_aobt"))

        y = dep["y"].to_numpy()
        proxy = dep["aobt_proxy"].to_numpy().astype(float)
        resid = dep["resid_aobt"].to_numpy().astype(float)
        log(fh, "AOBT residual (y - (MVT-AOBT)) = BLOCK-AOBT actually wait: y=MVT-BLOCK, proxy=MVT-AOBT, resid=AOBT-BLOCK")
        r = resid[np.isfinite(resid)]
        log(fh, f"resid n={r.size:,} mean={r.mean():.2f} med={np.median(r):.2f} std={r.std():.2f} mae={np.mean(np.abs(r)):.2f}")
        log(fh, f"RMSE of proxy {rmse(y, proxy)[0]:.2f}")

        # unmatched
        log(fh, "\n===== UNMATCHED NM FLIGHTS =====")
        u = dep.filter(pl.col("unmatched"))
        m = dep.filter(~pl.col("unmatched"))
        log(fh, f"unmatched n={u.height:,} ({100*u.height/dep.height:.2f}%)")
        uy = u["y"].to_numpy()
        my = m["y"].to_numpy()
        log(fh, f"unmatched taxi mean={uy.mean():.1f} med={np.median(uy):.1f} std={uy.std():.1f} p95={np.quantile(uy,0.95):.1f} p99={np.quantile(uy,0.99):.1f} max={uy.max():.1f}")
        log(fh, f"matched   taxi mean={my.mean():.1f} med={np.median(my):.1f} std={my.std():.1f} p95={np.quantile(my,0.95):.1f} p99={np.quantile(my,0.99):.1f}")
        log(fh, "unmatched by airport", u.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y").mean().alias("mean"), pl.col("y").median().alias("med"), pl.col("y").std().alias("std")).sort("n", descending=True))
        # RMSE if we predict airport median for unmatched
        log(fh, f"global mean RMSE on unmatched {rmse(uy, np.full_like(uy, my.mean()))[0]:.1f}")
        log(fh, f"global mean RMSE on matched {rmse(my, np.full_like(my, my.mean()))[0]:.1f}")
        log(fh, f"matched-only AOBT RMSE {rmse(m['y'].to_numpy(), m['aobt_proxy'].to_numpy().astype(float))[0]:.2f}")

        # geometry: same stand, different runways
        log(fh, "\n===== STAND-RUNWAY GEOMETRY =====")
        sr = (
            dep.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
            .agg(
                pl.len().alias("n"),
                pl.col("y").mean().alias("mean"),
                pl.col("y").median().alias("med"),
                pl.col("y").std().alias("std"),
                pl.col("aobt_proxy").mean().alias("proxy_mean"),
            )
            .filter(pl.col("n") >= 30)
        )
        log(fh, f"stand-rwy pairs with n>=30: {sr.height}")
        # stands that use 2+ runways
        st = sr.group_by(["ADEP_mvt", "STAND_mvt"]).agg(pl.len().alias("n_rwy"), pl.col("mean").min().alias("min_m"), pl.col("mean").max().alias("max_m"))
        multi = st.filter(pl.col("n_rwy") >= 2)
        log(fh, f"stands with >=2 runways (n>=30 each): {multi.height}")
        spread = (multi["max_m"] - multi["min_m"]).to_numpy()
        log(fh, f"mean taxi spread across runways for same stand: mean={spread.mean():.1f} med={np.median(spread):.1f} p90={np.quantile(spread,0.9):.1f} max={spread.max():.1f}")

        # within vs between variance
        # ANOVA-like: SS_between stand-rwy / SS_total
        overall = dep["y"].mean()
        joined = dep.join(sr.select(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "mean"]), on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        yy = joined["y"].to_numpy()
        mu = joined["mean"].to_numpy()
        msk = np.isfinite(mu)
        ss_tot = np.sum((yy[msk] - yy[msk].mean()) ** 2)
        ss_bet = np.sum((mu[msk] - yy[msk].mean()) ** 2)
        log(fh, f"stand-rwy (n>=30 groups) eta^2 = SS_between/SS_tot = {ss_bet/ss_tot:.4f}  (in-sample R2 of group means on those rows)")

        # airport eta2
        apm = dep.group_by("ADEP_mvt").agg(pl.col("y").mean().alias("apm"))
        j2 = dep.join(apm, on="ADEP_mvt")
        ss_b2 = np.sum((j2["apm"].to_numpy() - overall) ** 2)
        ss_t2 = np.sum((j2["y"].to_numpy() - overall) ** 2)
        log(fh, f"airport eta^2 = {ss_b2/ss_t2:.4f}")

        # rolling taxi-in of recent arrivals (LEAK-FREE at ranking)
        log(fh, "\n===== RECENT TAXI-IN (available in ranking) =====")
        arr_s = arr.sort(["airport", "MVT_TIME_UTC_mvt"]).select(
            "airport",
            pl.col("MVT_TIME_UTC_mvt").alias("arr_t"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("arr_block"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("taxiin"),
            "RUNWAY_mvt",
        )
        arr_s = arr_s.with_columns(pl.int_range(0, pl.len()).over("airport").alias("arr_idx"))
        # rolling mean of last 10 taxi-ins per airport using polars rolling
        arr_roll = arr.sort(["airport", "MVT_TIME_UTC_mvt"])
        arr_roll = arr_roll.with_columns(
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).rolling_mean(window_size=10).over("airport").alias("roll10_taxiin"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).rolling_median(window_size=10).over("airport").alias("roll10med_taxiin"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).shift(1).over("airport").alias("lag1_taxiin"),
        )
        # asof join onto dep by airport, using ARR MVT_TIME <= DEP MVT_TIME
        dep_k = dep.select("MVT_ID_mvt", "airport", "MVT_TIME_UTC_mvt", "RUNWAY_mvt").sort(["airport", "MVT_TIME_UTC_mvt"])
        arr_j = arr_roll.select(
            "airport",
            pl.col("MVT_TIME_UTC_mvt").alias("arr_t"),
            "roll10_taxiin",
            "roll10med_taxiin",
            "lag1_taxiin",
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("last_taxiin"),
            pl.col("RUNWAY_mvt").alias("arr_rwy"),
        ).sort(["airport", "arr_t"])
        joined_arr = dep_k.join_asof(arr_j, left_on="MVT_TIME_UTC_mvt", right_on="arr_t", by="airport", strategy="backward")
        dep = dep.join(joined_arr.select("MVT_ID_mvt", "roll10_taxiin", "lag1_taxiin", "last_taxiin"), on="MVT_ID_mvt", how="left")

        # rolling AOBT-proxy taxi of previous DEPs (available in ranking)
        dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
        dep = dep.with_columns(
            pl.col("aobt_proxy").shift(1).over("airport").alias("lag1_aobt"),
            pl.col("aobt_proxy").shift(1).rolling_mean(window_size=10).over("airport").alias("roll10_aobt"),
            pl.col("aobt_proxy").shift(1).rolling_median(window_size=10).over("airport").alias("roll10med_aobt"),
            pl.col("aobt_proxy").shift(1).rolling_mean(window_size=5).over("airport").alias("roll5_aobt"),
            pl.col("aobt_proxy").shift(1).rolling_mean(window_size=20).over("airport").alias("roll20_aobt"),
            pl.col("y").shift(1).over("airport").alias("lag1_y"),  # train-only
            pl.col("y").shift(1).rolling_mean(window_size=10).over("airport").alias("roll10_y"),
            pl.col("aobt_proxy").shift(1).over(["airport", "RUNWAY_mvt"]).alias("lag1_aobt_rwy"),
            pl.col("aobt_proxy").shift(1).rolling_mean(window_size=10).over(["airport", "RUNWAY_mvt"]).alias("roll10_aobt_rwy"),
        )

        log(fh, "\ncorrelations with y and resid_aobt:")
        feats = [
            "aobt_proxy",
            "lag1_aobt",
            "roll5_aobt",
            "roll10_aobt",
            "roll20_aobt",
            "roll10med_aobt",
            "lag1_aobt_rwy",
            "roll10_aobt_rwy",
            "lag1_y",
            "roll10_y",
            "lag1_taxiin",
            "roll10_taxiin",
            "last_taxiin",
        ]
        for c in feats:
            x = dep[c].to_numpy().astype(float)
            log(fh, f"  {c:22s} corr_y={corr(x,y):7.4f} corr_resid={corr(x,resid):7.4f}")

        # per airport roll10_aobt and queue-like
        log(fh, "\nper-airport corr(roll10_aobt, y) and corr(roll10_taxiin, y):")
        for ap in AIRPORTS:
            sub = dep.filter(pl.col("airport") == ap)
            log(
                fh,
                f"  {ap} roll10_aobt={corr(sub['roll10_aobt'].to_numpy().astype(float), sub['y'].to_numpy()):6.3f} "
                f"roll10_taxiin={corr(sub['roll10_taxiin'].to_numpy().astype(float), sub['y'].to_numpy()):6.3f} "
                f"aobt_proxy={corr(sub['aobt_proxy'].to_numpy().astype(float), sub['y'].to_numpy()):6.3f}",
            )

        # TIME SPLIT VALIDATION
        log(fh, "\n===== TIME-SPLIT VALIDATION =====")
        # split A: train months 1-11, val month 12
        # split B: train all except 1 and 7, val 1 and 7 (ranking months analogue)
        # split C: train 1-6, val 7 (summer)
        for name, val_months in [
            ("val_Dec", [12]),
            ("val_JanJul", [1, 7]),
            ("val_Jul", [7]),
            ("val_Jan", [1]),
        ]:
            tr = dep.filter(~pl.col("month").is_in(val_months))
            va = dep.filter(pl.col("month").is_in(val_months))
            log(fh, f"\n--- {name} train={tr.height:,} val={va.height:,} ---")
            yv = va["y"].to_numpy()
            # global mean
            p = np.full_like(yv, tr["y"].mean())
            log(fh, f"  global_mean          RMSE={rmse(yv,p)[0]:7.2f} MAE={mae(yv,p)[0]:7.2f}")
            # airport mean
            apm = tr.group_by("ADEP_mvt").agg(pl.col("y").mean().alias("p"))
            j = va.join(apm, on="ADEP_mvt", how="left")
            log(fh, f"  airport_mean         RMSE={rmse(j['y'].to_numpy(), j['p'].to_numpy())[0]:7.2f}")
            # stand-rwy mean, fallback airport
            sr = tr.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg(pl.col("y").mean().alias("p_sr"))
            apm2 = tr.group_by("ADEP_mvt").agg(pl.col("y").mean().alias("p_ap"))
            j = va.join(sr, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").join(apm2, on="ADEP_mvt")
            p = np.where(np.isfinite(j["p_sr"].to_numpy().astype(float)), j["p_sr"].to_numpy().astype(float), j["p_ap"].to_numpy().astype(float))
            log(fh, f"  stand_rwy_fb_ap      RMSE={rmse(yv,p)[0]:7.2f} MAE={mae(yv,p)[0]:7.2f}")
            # AOBT proxy (no training)
            log(fh, f"  aobt_proxy           RMSE={rmse(yv, va['aobt_proxy'].to_numpy().astype(float))[0]:7.2f} MAE={mae(yv, va['aobt_proxy'].to_numpy().astype(float))[0]:7.2f}")
            # clip aobt
            pc = np.clip(va["aobt_proxy"].to_numpy().astype(float), 60, 7200)
            log(fh, f"  aobt_proxy_clip      RMSE={rmse(yv,pc)[0]:7.2f}")
            # airport + wtc mean
            wt = tr.group_by(["ADEP_mvt", "WK_TBL_CAT_flt"]).agg(pl.col("y").mean().alias("p_w"))
            j = va.join(wt, on=["ADEP_mvt", "WK_TBL_CAT_flt"], how="left").join(apm2, on="ADEP_mvt")
            p = np.where(np.isfinite(j["p_w"].to_numpy().astype(float)), j["p_w"].to_numpy().astype(float), j["p_ap"].to_numpy().astype(float))
            log(fh, f"  airport_wtc          RMSE={rmse(yv,p)[0]:7.2f}")
            # OLS on train: y ~ aobt_proxy  (matched only)
            trm = tr.filter(pl.col("aobt_proxy").is_not_null())
            vam = va.filter(pl.col("aobt_proxy").is_not_null())
            X = np.column_stack([np.ones(trm.height), trm["aobt_proxy"].to_numpy().astype(float)])
            coef, *_ = np.linalg.lstsq(X, trm["y"].to_numpy(), rcond=None)
            pv = coef[0] + coef[1] * vam["aobt_proxy"].to_numpy().astype(float)
            log(fh, f"  OLS aobt             RMSE={rmse(vam['y'].to_numpy(), pv)[0]:7.2f} coef={coef}")
            # OLS aobt + roll10_aobt
            tr2 = trm.filter(pl.col("roll10_aobt").is_not_null())
            va2 = vam.filter(pl.col("roll10_aobt").is_not_null())
            X = np.column_stack([np.ones(tr2.height), tr2["aobt_proxy"].to_numpy().astype(float), tr2["roll10_aobt"].to_numpy().astype(float)])
            coef, *_ = np.linalg.lstsq(X, tr2["y"].to_numpy(), rcond=None)
            pv = coef[0] + coef[1] * va2["aobt_proxy"].to_numpy().astype(float) + coef[2] * va2["roll10_aobt"].to_numpy().astype(float)
            log(fh, f"  OLS aobt+roll10aobt  RMSE={rmse(va2['y'].to_numpy(), pv)[0]:7.2f} coef={coef}")
            # + roll10 taxiin
            tr3 = tr2.filter(pl.col("roll10_taxiin").is_not_null())
            va3 = va2.filter(pl.col("roll10_taxiin").is_not_null())
            X = np.column_stack(
                [
                    np.ones(tr3.height),
                    tr3["aobt_proxy"].to_numpy().astype(float),
                    tr3["roll10_aobt"].to_numpy().astype(float),
                    tr3["roll10_taxiin"].to_numpy().astype(float),
                ]
            )
            coef, *_ = np.linalg.lstsq(X, tr3["y"].to_numpy(), rcond=None)
            pv = (
                coef[0]
                + coef[1] * va3["aobt_proxy"].to_numpy().astype(float)
                + coef[2] * va3["roll10_aobt"].to_numpy().astype(float)
                + coef[3] * va3["roll10_taxiin"].to_numpy().astype(float)
            )
            log(fh, f"  OLS aobt+roll10a+ti  RMSE={rmse(va3['y'].to_numpy(), pv)[0]:7.2f} coef={coef}")
            # stand-rwy of residual after aobt, add back
            trm2 = trm.with_columns((pl.col("y") - pl.col("aobt_proxy")).alias("r"))
            sr_r = trm2.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg(pl.col("r").mean().alias("r_sr"))
            ap_r = trm2.group_by("ADEP_mvt").agg(pl.col("r").mean().alias("r_ap"))
            j = vam.join(sr_r, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").join(ap_r, on="ADEP_mvt")
            rhat = np.where(np.isfinite(j["r_sr"].to_numpy().astype(float)), j["r_sr"].to_numpy().astype(float), j["r_ap"].to_numpy().astype(float))
            p = j["aobt_proxy"].to_numpy().astype(float) + rhat
            log(fh, f"  aobt + standrwy resid RMSE={rmse(j['y'].to_numpy(), p)[0]:7.2f} MAE={mae(j['y'].to_numpy(), p)[0]:7.2f}")
            # blend aobt with stand-rwy mean
            sr_y = tr.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg(pl.col("y").mean().alias("p_sr"))
            j = va.join(sr_y, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").join(apm2, on="ADEP_mvt")
            psr = np.where(np.isfinite(j["p_sr"].to_numpy().astype(float)), j["p_sr"].to_numpy().astype(float), j["p_ap"].to_numpy().astype(float))
            pa = j["aobt_proxy"].to_numpy().astype(float)
            # unmatched: pa nan -> psr
            blend = np.where(np.isfinite(pa), 0.7 * pa + 0.3 * psr, psr)
            log(fh, f"  blend 0.7 aobt+0.3 sr RMSE={rmse(yv, blend)[0]:7.2f}")
            blend2 = np.where(np.isfinite(pa), pa, psr)
            log(fh, f"  aobt else standrwy    RMSE={rmse(yv, blend2)[0]:7.2f}")

        # LIRF deep dive
        log(fh, "\n===== LIRF =====")
        lirf = dep.filter(pl.col("ADEP_mvt") == "LIRF")
        log(fh, f"n={lirf.height:,} unmatched={lirf['unmatched'].sum()}")
        ly = lirf["y"].to_numpy()
        log(fh, f"mean={ly.mean():.1f} med={np.median(ly):.1f} std={ly.std():.1f} p99={np.quantile(ly,0.99):.1f} >1h={(ly>3600).sum()}")
        log(fh, "by runway", lirf.group_by("RUNWAY_mvt").agg(pl.len().alias("n"), pl.col("y").mean().alias("mean"), pl.col("y").median().alias("med"), pl.col("y").std().alias("std"), (pl.col("y") > 3600).sum().alias("gt1h")).sort("n", descending=True))
        log(fh, "by month", lirf.group_by("month").agg(pl.len().alias("n"), pl.col("y").mean().alias("mean"), pl.col("y").std().alias("std"), (pl.col("y") > 3600).sum().alias("gt1h")).sort("month"))
        log(fh, f"LIRF AOBT RMSE {rmse(lirf['y'].to_numpy(), lirf['aobt_proxy'].to_numpy().astype(float))[0]:.1f}")
        log(fh, f"LIRF unmatched taxi mean={lirf.filter(pl.col('unmatched'))['y'].mean()}")

        # EGLL
        log(fh, "\n===== EGLL =====")
        eg = dep.filter(pl.col("ADEP_mvt") == "EGLL")
        log(fh, f"AOBT RMSE {rmse(eg['y'].to_numpy(), eg['aobt_proxy'].to_numpy().astype(float))[0]:.1f}")
        log(fh, "by runway", eg.group_by("RUNWAY_mvt").agg(pl.len().alias("n"), pl.col("y").mean().alias("mean"), pl.col("y").median().alias("med"), pl.col("y").std().alias("std")).sort("n", descending=True))

        # ranking: unmatched rate
        log(fh, "\n===== RANKING UNMATCHED =====")
        rank = pl.scan_parquet(DATA / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
        n = rank.select(pl.len()).collect().item()
        nu = rank.select(pl.col("AOBT_3_flt").is_null().sum()).collect().item()
        log(fh, f"ranking DEP unmatched AOBT {nu:,}/{n:,} ({100*nu/n:.2f}%)")
        log(fh, rank.with_columns(pl.col("AOBT_3_flt").is_null().alias("u")).group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("u").sum().alias("unmatched")).collect())

        # SCHED vs BLOCK delay as feature of residual
        log(fh, "\n===== delay vs residual =====")
        dep = dep.with_columns(
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("aobt_minus_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("aobt_minus_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("mvt_minus_sched"),
        )
        for c in ["aobt_minus_eobt", "aobt_minus_sched", "mvt_minus_sched"]:
            x = dep[c].to_numpy().astype(float)
            log(fh, f"  {c:22s} corr_y={corr(x,y):7.4f} corr_resid={corr(x,resid):7.4f}")

        # AOBT residual by airport
        log(fh, "\nAOBT residual by airport (y - proxy):")
        for ap in AIRPORTS:
            sub = dep.filter(pl.col("ADEP_mvt") == ap)
            r = sub["resid_aobt"].to_numpy().astype(float)
            r = r[np.isfinite(r)]
            log(fh, f"  {ap} mean={r.mean():7.1f} med={np.median(r):7.1f} std={r.std():7.1f} mae={np.mean(np.abs(r)):7.1f} rmse={np.sqrt(np.mean(r**2)):7.1f}")

    print("WROTE", out)


if __name__ == "__main__":
    main()
