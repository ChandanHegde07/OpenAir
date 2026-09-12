"""E73 — LIRF E33 G-model variants (all-LIRF training): inbound/stand features + seed avg.
Holdout Jan+Jul / Dec, LIRF-unmatched RMSE + SSE."""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,HERE)
from common import TRAIN_FILES, rmse
from run_e33_delay_decomposition import CAT as GCAT, NUM as GNUM, add_surface, log
from run_e34d_enriched import EXTRA, enrich
from catboost import CatBoostRegressor

RES=HERE/"results"/"E73"; RES.mkdir(parents=True,exist_ok=True)

dep=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="DEP")
     .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
              "TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"])
     .collect()).rename({"ADEP_mvt":"airport"})
dep=dep.with_columns([pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("T"),
                      (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
                      pl.col("AOBT_3_flt").is_null().alias("um")])
arr=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="ARR")
     .select([pl.col("ADES_mvt").alias("airport"),"STAND_mvt",pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
              pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
              pl.col("SCHED_TIME_UTC_mvt").alias("asched")])
     .drop_nulls(["airport","STAND_mvt","aibt"]).collect().sort(["airport","STAND_mvt","aibt"]))
arr=arr.with_columns((pl.col("aibt")-pl.col("asched")).dt.total_seconds().alias("arr_delay"),
                     pl.col("arr_taxiin").shift(1).over(["airport","STAND_mvt"]).alias("arr_taxiin_2"))
d=dep.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
j=d.join_asof(arr.rename({"arr_taxiin_2":"arr_taxiin2"}), left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport","STAND_mvt"], strategy="backward")
j=j.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("aibt")).dt.total_seconds().alias("since_arr"))
j=j.with_columns(pl.when((pl.col("since_arr")>0)&(pl.col("since_arr")<36*3600)).then(pl.col("since_arr")).otherwise(None).alias("since_arr"))
# stand release: has another DEP or ARR at same stand within (SCHED,MVT)
arr2=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="ARR")
      .select([pl.col("ADES_mvt").alias("airport"),"MVT_TIME_UTC_mvt","RUNWAY_mvt"]).collect().sort(["airport","MVT_TIME_UTC_mvt"]))
INB=["arr_taxiin","arr_taxiin2","since_arr","arr_delay"]

def prep(x, dref):
    x=x.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0,3).alias("flt_prefix"),
                     pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
                     pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
                     pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
                     pl.col("D").alias("mvt_sched"))
    x=add_surface(x,arr2)
    return enrich(x, dref)

def run(split, months, feats, seeds=(1,), tag=""):
    tr=j.filter((pl.col("airport")=="LIRF")&(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)))
    va=j.filter((pl.col("airport")=="LIRF")&(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))&pl.col("um"))
    trf=prep(tr, tr["D"].to_numpy().astype(float)); vaf=prep(va, tr["D"].to_numpy().astype(float))
    cats=GCAT+["arr_type"] if "arr_type" in trf.columns else GCAT
    def cdf(x):
        z=x.select(feats+cats).to_pandas()
        for c in cats: z[c]=z[c].fillna("NA").astype(str)
        return z
    gsum=np.zeros(vaf.height)
    for s in seeds:
        m=CatBoostRegressor(iterations=1200,learning_rate=0.04,depth=8,l2_leaf_reg=5.0,loss_function="RMSE",
                            random_seed=s,verbose=False,thread_count=-1,od_type="Iter",od_wait=80)
        m.fit(cdf(trf), trf["D"].to_numpy().astype(float)-trf["T"].to_numpy().astype(float), cat_features=cats)
        gsum+=np.asarray(m.predict(cdf(vaf)),float)
    g=gsum/len(seeds)
    D=vaf["D"].to_numpy().astype(float); T=vaf["T"].to_numpy().astype(float)
    base=np.maximum(D-g,0)
    return T, base

plans={
 "base": (GNUM+EXTRA, (1,)),
 "inb": (GNUM+EXTRA+INB, (1,)),
 "inb_seed3": (GNUM+EXTRA+INB, (1,2,3)),
 "base_seed3": (GNUM+EXTRA, (1,2,3)),
}
out={}
for split,months in (("janjul",[1,7]),("dec",[12])):
    out[split]={}
    for tag,(feats,seeds) in plans.items():
        T,base=run(split,months,feats,seeds,tag)
        out[split][tag]={"n":int(len(T)),"rmse":float(rmse(T,base)),"sse":float(np.sum((T-base)**2)),"mean":float(np.mean(base))}
        log(f"[{split}] {tag:12s} n={len(T):4d} rmse={rmse(T,base):8.0f} sse={np.sum((T-base)**2):.3e}")
(RES/"E73_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
log("WROTE E73")
