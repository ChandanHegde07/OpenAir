"""E71c — correct previous-DEP-at-stand features (self-excluded) + oracle/student."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, rmse

def log(m): print(m, flush=True)
dep = (pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="DEP")
       .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
                "TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"])
       .collect()).rename({"ADEP_mvt":"airport"})
dep = dep.with_columns([
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("T"),
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
    pl.col("AOBT_3_flt").is_null().alias("um"),
])
d=dep.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
d=d.with_columns([
    pl.col("MVT_TIME_UTC_mvt").shift(1).over(["airport","STAND_mvt"]).alias("pmvt"),
    pl.col("BLOCK_TIME_UTC_mvt").shift(1).over(["airport","STAND_mvt"]).alias("pblock"),
    pl.col("T").shift(1).over(["airport","STAND_mvt"]).alias("pTaxi"),
    pl.col("MVT_ID_mvt").shift(1).over(["airport","STAND_mvt"]).alias("pid"),
])
d=d.with_columns([
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("pmvt")).dt.total_seconds().alias("gap_mvt"),
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("pblock")).dt.total_seconds().alias("gap_block"),
])
d=d.filter(pl.col("pid").is_not_null())
print("=== PREVIOUS DEP AT SAME STAND (correct, self-excluded) ===")
for nm,m in (("unmatched",pl.col("um")),("nonLIRF-u",pl.col("um")&(pl.col("airport")!="LIRF")),("LIRF-u",pl.col("um")&(pl.col("airport")=="LIRF"))):
    s=d.filter(m)
    T=s["T"].to_numpy().astype(float); gb=s["gap_block"].to_numpy().astype(float)
    gm=s["gap_mvt"].to_numpy().astype(float); pT=s["pTaxi"].to_numpy().astype(float)
    print(f"\n  [{nm}] n={s.height}")
    for lab,x in (("prev_Taxi",pT),("gap_block(prev BLOCK)",gb),("gap_mvt",gm)):
        mm=np.isfinite(x)&np.isfinite(T)
        print(f"    corr(T,{lab:22s})={np.corrcoef(T[mm],x[mm])[0,1]:7.3f}  rmse(T,{lab:22s})={rmse(T[mm],x[mm]):11.1f}  med={np.median(x[mm]):9.0f}")
    mm=np.isfinite(gb)&np.isfinite(T)
    print(f"    frac T<=gap_block: {np.mean(T[mm]<=gb[mm]):.3f}   med(T/gap_block)={np.median(T[mm]/gb[mm]):.3f}")
    # oracle: T_hat = clip(prev_Taxi, 0, gap_block)? or use prev Taxi as direct structural guess
    print(f"    direct T_hat=prev_Taxi: rmse={rmse(T[mm],np.clip(pT[mm],0,None)):9.1f}")
    print(f"    cap median_T at gap_block: {rmse(T[mm], np.minimum(np.median(T),gb[mm])):9.1f}")

print("\n=== LIRF-ALL: D-G decomposition + prev-block oracle on holdout months ===")
for split,months in (("janjul",[1,7]),("dec",[12])):
    s=d.filter(pl.col("airport")=="LIRF")
    tr=s.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    va=s.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    for nm,x in (("all",va),("unmatched",va.filter(pl.col("um")))):
        T=x["T"].to_numpy().astype(float); D=x["D"].to_numpy().astype(float)
        gb=x["gap_block"].to_numpy().astype(float); pT=x["pTaxi"].to_numpy().astype(float)
        base_med=np.median(tr["T"].to_numpy())
        print(f"  [{split}/{nm}] n={x.height} med_base={rmse(T,np.full_like(T,base_med)):.0f} D={rmse(T,np.clip(D,0,None)):.0f} "
              f"D-Gmean={rmse(T,np.clip(D-np.mean(tr['D'].to_numpy()-tr['T'].to_numpy()),0,None)):.0f} "
              f"prevTaxi={rmse(T,np.clip(pT,0,None)):.0f} gapBlock={rmse(T,np.clip(gb,0,None)):.0f}")
