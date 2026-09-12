"""E72 — stand-release hard bound: T <= since_arr (current aircraft's own arrival
at the stand). Tests capping v8 (LIRF-u) and E20 (non-LIRF-u) predictions."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, TRAIN_FILES as TF, rmse
from matched_submit import add_extra, e20_col
from common import load_dep

def log(m): print(m, flush=True)

dep = (pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="DEP")
       .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
                "TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"])
       .collect()).rename({"ADEP_mvt":"airport"})
dep = dep.with_columns([pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("T"),
                        (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
                        pl.col("AOBT_3_flt").is_null().alias("um")])
arr = (pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="ARR")
       .select([pl.col("ADES_mvt").alias("airport"),"STAND_mvt",pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
                pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type")])
       .drop_nulls(["airport","STAND_mvt","aibt"]).collect().sort(["airport","STAND_mvt","aibt"]))
d=dep.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
j=d.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport","STAND_mvt"], strategy="backward")
j=j.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("aibt")).dt.total_seconds().alias("since_arr"))
j=j.with_columns(pl.when((pl.col("since_arr")>0)&(pl.col("since_arr")<36*3600)).then(pl.col("since_arr")).otherwise(None).alias("since_arr"))
cov=lambda x: np.isfinite(x)

# E20 OOF baseline for holdout months
oofs={}
for split in ("janjul","dec"):
    o=pl.read_parquet(HERE/"results"/"E20"/f"oof_predictions_{split}.parquet")
    oofs[split]=o.with_columns(pl.Series("e20",0.456*o["pred_C"].to_numpy()+0.053*o["pred_D"].to_numpy()+0.491*o["pred_E"].to_numpy()))

# ---- LIRF G model (E34 recipe, CatBoost all-LIRF) for holdout ----
from run_e33_delay_decomposition import CAT as G_CAT, NUM as G_NUM, add_surface
from run_e34d_enriched import EXTRA, enrich
from catboost import CatBoostRegressor
def lirf_g(split, months):
    tr=j.filter((pl.col("airport")=="LIRF")&(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)))
    va=j.filter((pl.col("airport")=="LIRF")&(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))&pl.col("um"))
    arr2=(pl.concat([pl.scan_parquet(p) for p in TF]).filter(pl.col("PHASE_mvt")=="ARR")
          .select([pl.col("ADES_mvt").alias("airport"),"MVT_TIME_UTC_mvt","RUNWAY_mvt"]).collect().sort(["airport","MVT_TIME_UTC_mvt"]))
    def featurize(x, dref):
        x=x.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0,3).alias("flt_prefix"),
                         pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
                         pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
                         pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
                         pl.col("D").alias("mvt_sched"))
        x=add_surface(x,arr2)
        return enrich(x, dref)
    trf=featurize(tr, tr["D"].to_numpy().astype(float)); vaf=featurize(va, tr["D"].to_numpy().astype(float))
    feats=G_NUM+EXTRA
    def cdf(x):
        z=x.select(feats+G_CAT).to_pandas()
        for c in G_CAT: z[c]=z[c].fillna("NA").astype(str)
        return z
    m=CatBoostRegressor(iterations=1200,learning_rate=0.04,depth=8,l2_leaf_reg=5.0,loss_function="RMSE",
                        random_seed=1,verbose=False,thread_count=-1,od_type="Iter",od_wait=80)
    m.fit(cdf(trf), trf["D"].to_numpy().astype(float)-trf["T"].to_numpy().astype(float), cat_features=G_CAT)
    g=np.asarray(m.predict(cdf(vaf)),float)
    D=vaf["D"].to_numpy().astype(float); T=vaf["T"].to_numpy().astype(float)
    base=np.maximum(D-g,0); sa=vaf["since_arr"].to_numpy().astype(float)
    return T, base, sa

print("=== LIRF-unmatched: v8 G + stand-release cap ===")
for split,months in (("janjul",[1,7]),("dec",[12])):
    T,base,sa=lirf_g(split,months)
    print(f"  [{split}] n={len(T)} base(D-G) rmse={rmse(T,base):.0f} sse={np.sum((T-base)**2):.3e}")
    for lo,hi,lab in ((0,36*3600,"all"),(0,12*3600,"<12h"),(0,6*3600,"<6h")):
        msk=cov(sa)&(sa>lo)&(sa<hi)
        capped=np.minimum(base, np.where(msk,sa,np.inf))
        print(f"     cap {lab:5s} rows={msk.sum():4d} rmse={rmse(T,capped):.0f} sse={np.sum((T-capped)**2):.3e} "
              f"cap_binds={int((base>sa)[msk].sum())}")

print("\n=== non-LIRF-unmatched: E20 + stand-release cap ===")
for split,months in (("janjul",[1,7]),("dec",[12])):
    o=oofs[split]
    f=j.join(o.select(["MVT_ID_mvt","e20","unmatched"]),on="MVT_ID_mvt",how="inner")
    f=f.filter(pl.col("unmatched")&(pl.col("airport")!="LIRF"))
    T=f["T"].to_numpy().astype(float); p=f["e20"].to_numpy().astype(float); sa=f["since_arr"].to_numpy().astype(float)
    print(f"  [{split}] n={len(T)} base rmse={rmse(T,p):.1f} sse={np.sum((T-p)**2):.3e} since_arr cov={cov(sa).mean():.3f}")
    for lo,hi,lab in ((0,36*3600,"all"),(0,12*3600,"<12h"),(0,6*3600,"<6h")):
        msk=cov(sa)&(sa>lo)&(sa<hi)
        capped=np.minimum(p, np.where(msk,sa,np.inf))
        print(f"     cap {lab:5s} rows={msk.sum():5d} rmse={rmse(T,capped):.1f} sse={np.sum((T-capped)**2):.3e} cap_binds={int((p>sa)[msk].sum())}")
