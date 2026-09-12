"""E71b — structural decomposition search + previous-DEP-BLOCK oracle."""
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
       .select(["MVT_ID_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt",
                "BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
                "MARKET_SEGMENT_flt","AOBT_3_flt"]).collect())
dep = dep.with_columns([
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("T"),
    pl.col("AOBT_3_flt").is_null().alias("um"),
    pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
]).rename({"ADEP_mvt":"airport"})

print("=== PER-AIRPORT UNMATCHED STRUCTURE: T vs D ===")
for a in sorted(dep["airport"].unique().to_list()):
    s=dep.filter((pl.col("airport")==a)&pl.col("um"))
    if s.height<50: continue
    T=s["T"].to_numpy().astype(float); D=s["D"].to_numpy().astype(float)
    ok=np.isfinite(T)&np.isfinite(D)
    T,D=T[ok],D[ok]
    c=np.corrcoef(T,D)[0,1] if len(T)>2 else np.nan
    # E33 style: T = D - G, G = D - T ; how much does D alone help vs median?
    med=np.median(T)
    print(f"  {a:5s} n={len(T):5d} corr(T,D)={c:6.3f} T_med={med:8.0f} D_med={np.median(D):8.0f} "
          f"rmse(T,med)={rmse(T,np.full_like(T,med)):9.1f} rmse(T,D)={rmse(T,D):10.1f} "
          f"rmse(T,D-Gmean)={rmse(T,D-np.mean(D-T)):9.1f}")

print("\n=== PREVIOUS DEP AT SAME STAND (oracle BLOCK) ===")
prev=dep.select([pl.col("airport"),"STAND_mvt",pl.col("MVT_TIME_UTC_mvt").alias("pmvt"),
                 pl.col("BLOCK_TIME_UTC_mvt").alias("pblock"),pl.col("T").alias("pTaxi"),
                 pl.col("SCHED_TIME_UTC_mvt").alias("psched"),pl.col("um").alias("pum")]).sort(["airport","STAND_mvt","pmvt"])
d=dep.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
j=d.join_asof(prev, left_on="MVT_TIME_UTC_mvt", right_on="pmvt", by=["airport","STAND_mvt"], strategy="backward")
j=j.with_columns([
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("pmvt")).dt.total_seconds().alias("gap_mvt"),
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("pblock")).dt.total_seconds().alias("gap_block"),
])
# keep only genuinely previous (strict) and plausible turnaround <48h
j=j.with_columns([pl.when(pl.col("gap_mvt")<=0).then(None).otherwise(pl.col("gap_mvt")).alias("gap_mvt"),
                  pl.when(pl.col("gap_block")<=0).then(None).otherwise(pl.col("gap_block")).alias("gap_block")])
for nm,m in (("unmatched",pl.col("um")),("nonLIRF-u",pl.col("um")&(pl.col("airport")!="LIRF")),("LIRF-u",pl.col("um")&(pl.col("airport")=="LIRF"))):
    s=j.filter(m)
    T=s["T"].to_numpy().astype(float); gb=s["gap_block"].to_numpy().astype(float); gm=s["gap_mvt"].to_numpy().astype(float)
    pb=s["pblock"].to_numpy().astype(object)  # datetime
    pT=s["pTaxi"].to_numpy().astype(float); gb_ok=np.isfinite(gb)
    print(f"\n  [{nm}] n={s.height} gap_block finite={gb_ok.mean():.3f}")
    for lab,x in (("gap_block",gb),("gap_mvt",gm),("prev_Taxi",pT)):
        mm=np.isfinite(x)&np.isfinite(T)
        if mm.sum()<10: continue
        print(f"    corr(T,{lab})={np.corrcoef(T[mm],x[mm])[0,1]:.3f}  rmse(T,{lab})={rmse(T[mm],x[mm]):10.1f}  med {lab}={np.median(x[mm]):8.0f}")
    # oracle structural: T_hat = gap_block (hard upper bound if stand reused); clipped
    mm=gb_ok&np.isfinite(T)
    if mm.sum()>10:
        cap=np.minimum(np.median(T),gb[mm])
        print(f"    oracle cap min(median_T, gap_block): rmse={rmse(T[mm],cap):9.1f} vs base rmse(T,med)={rmse(T[mm],np.full(mm.sum(),np.median(T))):9.1f}")
        print(f"    frac T<=gap_block: {np.mean(T[mm]<=gb[mm]):.3f}  (should be ~1 if hard bound holds)")
print("\n=== TOP non-LIRF-u SSE rows (E20 OOF) ===")
sys.path.insert(0,str(HERE))
from matched_submit import add_extra, e20_col
from common import load_dep
d2=add_extra(load_dep())
oof=pl.read_parquet(HERE/"results"/"E20"/"oof_predictions_janjul.parquet")
e=e20_col(oof).join(oof.select(["MVT_ID_mvt","unmatched"]),on="MVT_ID_mvt")
f=d2.join(e,on="MVT_ID_mvt",how="inner").filter(pl.col("unmatched")&(pl.col("airport")!="LIRF"))
f=f.with_columns(((pl.col("y")-pl.col("e20_pred"))**2).alias("sse"))
print(f"n={f.height} rmse={rmse(f['y'].to_numpy(),f['e20_pred'].to_numpy()):.1f} totalSSE={f['sse'].sum():.3e} top1%share="
      f"{f.sort('sse',descending=True).head(max(1,f.height//100))['sse'].sum()/f['sse'].sum():.3f}")
print(f.sort("sse",descending=True).select(["airport","y","e20_pred","mvt_sched","month"]).head(12).to_pandas().to_string())
