"""E74 production builder — likable-eagle_v20.parquet.

Structural change vs v19: the LIRF-unmatched slice (previously the frozen v8
T = D - G_hat) is replaced by an E33 decomposition whose latent gate component G
is learned on ALL LIRF departures (matched+unmatched) including causal ARR
turnaround features (arr_taxiin, arr_taxiin_2, since_arr, arr_delay) at the same
stand. Everything else (v19 matched, non-LIRF unmatched) is preserved from v19.

Causal: all ARR features use in-block/taxi-in strictly before the current MVT.
Ranking never enters the fit. No previous-BLOCK/oracle features used.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,HERE)
from common import ROOT, TRAIN_FILES
from matched_submit import _v8_path
from run_e33_delay_decomposition import CAT as GCAT, NUM as GNUM, add_surface
from run_e34d_enriched import EXTRA, enrich
from catboost import CatBoostRegressor

def log(m): print(m, flush=True)
INB=["arr_taxiin","arr_taxiin_2","since_arr","arr_delay"]
V19=ROOT/"submissions"/"likable-eagle_v19.parquet"
OUT=["experiments/results/submit/likable-eagle_v20.parquet","likable-eagle_v20.parquet","submissions/likable-eagle_v20.parquet"]

def arr_frame(src, airport_col):
    a=(pl.concat([pl.scan_parquet(p) for p in src]).filter(pl.col("PHASE_mvt")=="ARR")
       .select([pl.col(airport_col).alias("airport"),"STAND_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
                pl.col("SCHED_TIME_UTC_mvt").alias("asched"),
                pl.col("MVT_TIME_UTC_mvt"),
                pl.col("RUNWAY_mvt"),
                pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type")])
       .drop_nulls(["airport","STAND_mvt","aibt"]).collect().sort(["airport","STAND_mvt","aibt"]))
    return a.with_columns((pl.col("aibt")-pl.col("asched")).dt.total_seconds().alias("arr_delay"),
                          pl.col("arr_taxiin").shift(1).over(["airport","STAND_mvt"]).alias("arr_taxiin_2"))

def prep(dep, arr, dref):
    x=dep.with_columns(pl.col("FLIGHT_mvt").fill_null("").str.slice(0,3).alias("flt_prefix"),
                       pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int64).alias("hour"),
                       pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
                       pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
                       pl.col("D").alias("mvt_sched"))
    x=x.sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"]).join_asof(
        arr, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport","STAND_mvt"], strategy="backward")
    x=x.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("aibt")).dt.total_seconds().alias("since_arr"))
    x=x.with_columns(pl.when((pl.col("since_arr")>0)&(pl.col("since_arr")<36*3600)).then(pl.col("since_arr")).otherwise(None).alias("since_arr"))
    arr2=arr.select([pl.col("airport"),pl.col("MVT_TIME_UTC_mvt"),pl.col("RUNWAY_mvt")]).sort(["airport","MVT_TIME_UTC_mvt"])
    x=add_surface(x, arr2)
    return enrich(x, dref)

# ---- training LIRF (all) ----
dep_tr=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="DEP")
        .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
                 "TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"])
        .collect()).rename({"ADEP_mvt":"airport"}).filter(pl.col("airport")=="LIRF")
dep_tr=dep_tr.with_columns([pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("T"),
                            (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D")])
arr_tr=arr_frame(TRAIN_FILES,"ADES_mvt")
dref_tr=dep_tr["D"].to_numpy().astype(float)
tr=prep(dep_tr, arr_tr, dref_tr)
cats=list(dict.fromkeys(GCAT+["arr_type"]))
feats=list(dict.fromkeys(GNUM+EXTRA+INB))
def mkdf(x):
    z=x.select(feats+cats).to_pandas()
    for c in cats: z[c]=z[c].fillna("NA").astype(str)
    return z
log(f"training LIRF rows {tr.height}")
m=CatBoostRegressor(iterations=1200,learning_rate=0.04,depth=8,l2_leaf_reg=5.0,loss_function="RMSE",
                    random_seed=1,verbose=False,thread_count=-1,od_type="Iter",od_wait=80)
m.fit(mkdf(tr), tr["D"].to_numpy().astype(float)-tr["T"].to_numpy().astype(float), cat_features=cats)

# ---- ranking LIRF ----
rraw=pl.scan_parquet(ROOT/"data"/"ranking.parquet")
dep_rk=(rraw.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF"))
        .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","RUNWAY_mvt","STAND_mvt",
                 "AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"]).collect()).rename({"ADEP_mvt":"airport"})
dep_rk=dep_rk.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"))
arr_rk=arr_frame([ROOT/"data"/"ranking.parquet"],"ADES_mvt")
rk=prep(dep_rk, arr_rk, dref_tr)
g_rk=np.asarray(m.predict(mkdf(rk)),float)
D_rk=rk["D"].to_numpy().astype(float)
t_new=np.clip(D_rk-g_rk, 0, None)
um_rk=rk["AOBT_3_flt"].is_null().to_numpy()
log(f"ranking LIRF rows {rk.height} unmatched {int(um_rk.sum())} new mean {t_new[um_rk].mean():.1f} med {np.median(t_new[um_rk]):.1f}")

# ---- splice onto v19 for LIRF-unmatched only ----
v19=pl.read_parquet(V19)
repl=pl.DataFrame({"MVT_ID_mvt":rk["MVT_ID_mvt"].to_numpy()[um_rk],"new":t_new[um_rk]})
j=v19.rename({"TAXITIME_SEC_mvt":"old"}).join(repl,on="MVT_ID_mvt",how="left")
final=np.where(j["new"].is_not_null().to_numpy(), j["new"].to_numpy(), j["old"].to_numpy())
final=np.maximum(final.astype(float),0.0)
out=j.select("MVT_ID_mvt").with_columns(pl.Series("TAXITIME_SEC_mvt",final))
sub=pl.read_parquet(ROOT/"data"/"submitting.parquet")
assert out.height==344841 and bool((out["MVT_ID_mvt"]==sub["MVT_ID_mvt"]).all()), "id/order check failed"
assert out["TAXITIME_SEC_mvt"].null_count()==0 and int(out["TAXITIME_SEC_mvt"].is_nan().sum())==0, "null/nan"
assert float(out["TAXITIME_SEC_mvt"].min())>=0
nchg=int((np.abs(final-j["old"].to_numpy())>1e-6).sum())
for p in OUT:
    pp=ROOT/p; pp.parent.mkdir(parents=True,exist_ok=True); out.write_parquet(pp); log(f"WROTE {pp}")
log(f"v20 rows {out.height} changed_vs_v19 {nchg} mean {final.mean():.2f} median {np.median(final):.2f}")
log("VALIDATION_OK")
