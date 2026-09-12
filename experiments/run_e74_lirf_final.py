"""E74 — finalize LIRF G recipe (self-contained): base vs inbound vs G-ensemble."""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,HERE)
from common import TRAIN_FILES, rmse
from run_e33_delay_decomposition import CAT as GCAT, NUM as GNUM, add_surface
from run_e34d_enriched import EXTRA, enrich
from catboost import CatBoostRegressor

def log(m): print(m, flush=True)
RES=HERE/"results"/"E74"; RES.mkdir(parents=True,exist_ok=True)
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
              pl.col("SCHED_TIME_UTC_mvt").alias("asched"),
              pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type")])
     .drop_nulls(["airport","STAND_mvt","aibt"]).collect().sort(["airport","STAND_mvt","aibt"]))
arr=arr.with_columns((pl.col("aibt")-pl.col("asched")).dt.total_seconds().alias("arr_delay"),
                     pl.col("arr_taxiin").shift(1).over(["airport","STAND_mvt"]).alias("arr_taxiin_2"))
d=dep.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
j=d.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport","STAND_mvt"], strategy="backward")
j=j.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("aibt")).dt.total_seconds().alias("since_arr"))
j=j.with_columns(pl.when((pl.col("since_arr")>0)&(pl.col("since_arr")<36*3600)).then(pl.col("since_arr")).otherwise(None).alias("since_arr"))
arr2=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="ARR")
      .select([pl.col("ADES_mvt").alias("airport"),"MVT_TIME_UTC_mvt","RUNWAY_mvt"]).collect().sort(["airport","MVT_TIME_UTC_mvt"]))
INB=["arr_taxiin","arr_taxiin_2","since_arr","arr_delay"]
def prep(x,dref):
    x=x.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0,3).alias("flt_prefix"),
                     pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
                     pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
                     pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
                     pl.col("D").alias("mvt_sched"))
    return enrich(add_surface(x,arr2), dref)

payload={}
for split,months in (("janjul",[1,7]),("dec",[12])):
    tr=j.filter((pl.col("airport")=="LIRF")&(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)))
    va=j.filter((pl.col("airport")=="LIRF")&(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))&pl.col("um"))
    trf=prep(tr, tr["D"].to_numpy().astype(float)); vaf=prep(va, tr["D"].to_numpy().astype(float))
    cats=list(dict.fromkeys(GCAT+[c for c in ["arr_type"] if c in trf.columns]))
    INBset=list(dict.fromkeys(GNUM+EXTRA+INB))
    BASEset=list(dict.fromkeys(GNUM+EXTRA))
    def mkdf(x, featset):
        z=x.select(featset+cats).to_pandas()
        for c in cats: z[c]=z[c].fillna("NA").astype(str)
        return z
    D=vaf["D"].to_numpy().astype(float); T=vaf["T"].to_numpy().astype(float)
    def fit_g(featset,seeds):
        g=np.zeros(vaf.height)
        for s in seeds:
            m=CatBoostRegressor(iterations=1200,learning_rate=0.04,depth=8,l2_leaf_reg=5.0,loss_function="RMSE",
                                random_seed=s,verbose=False,thread_count=-1,od_type="Iter",od_wait=80)
            m.fit(mkdf(trf,featset), trf["D"].to_numpy().astype(float)-trf["T"].to_numpy().astype(float), cat_features=cats)
            g+=np.asarray(m.predict(mkdf(vaf,featset)),float)
        return g/len(seeds)
    gb=fit_g(BASEset,(1,)); gi=fit_g(INBset,(1,)); gi3=fit_g(INBset,(1,2,3))
    cands={"base":gb,"inb":gi,"inb3":gi3,"ens_gb_gi":0.5*gb+0.5*gi,"ens_gb_gi3":0.5*gb+0.5*gi3,"ens_gi_gi3":0.5*gi+0.5*gi3}
    payload[split]={}
    for nm,g in cands.items():
        p=np.maximum(D-g,0)
        payload[split][nm]={"n":int(len(T)),"rmse":float(rmse(T,p)),"sse":float(np.sum((T-p)**2))}
        log(f"[{split}] {nm:12s} rmse={rmse(T,p):8.1f} sse={np.sum((T-p)**2):.3e}")
    pl.DataFrame({"MVT_ID_mvt":vaf["MVT_ID_mvt"].to_numpy(),"T":T,"D":D,"gb":gb,"gi":gi,"gi3":gi3}).write_parquet(RES/f"holdout_{split}.parquet")
(RES/"E74_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
log("WROTE E74")
