"""E74 internal check — confirm the corrected production prep (surface ARR stream
from the true ARR frame) reproduces the validated E74 LIRF-unmatched gain."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,HERE)
import run_e74_make_v20 as B
from common import TRAIN_FILES, rmse
from catboost import CatBoostRegressor

def build_train_eval(train_dep, pred_dep, arr, dref, seed=1):
    tr=B.prep(train_dep, arr, dref)
    va=B.prep(pred_dep, arr, dref)
    m=CatBoostRegressor(iterations=1200,learning_rate=0.04,depth=8,l2_leaf_reg=5.0,loss_function="RMSE",
                        random_seed=seed,verbose=False,thread_count=-1,od_type="Iter",od_wait=80)
    m.fit(B.mkdf(tr), tr["D"].to_numpy().astype(float)-tr["T"].to_numpy().astype(float), cat_features=B.cats)
    g=np.asarray(m.predict(B.mkdf(va)),float)
    return va["T"].to_numpy().astype(float), np.clip(va["D"].to_numpy().astype(float)-g,0,None)

base_dep=(pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="DEP")
          .select(["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
                   "TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","AOBT_3_flt","FLIGHT_mvt"])
          .collect()).rename({"ADEP_mvt":"airport"}).filter(pl.col("airport")=="LIRF")
base_dep=base_dep.with_columns([pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("T"),
                                (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D")])
arr=B.arr_frame(TRAIN_FILES,"ADES_mvt")
for split,months in (("janjul",[1,7]),("dec",[12])):
    trm=base_dep.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    vam=base_dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)&pl.col("AOBT_3_flt").is_null())
    T,p=build_train_eval(trm, vam, arr, trm["D"].to_numpy().astype(float))
    print(f"[{split}] corrected-prep LIRF-u n={len(T)} rmse={rmse(T,p):.0f} sse={np.sum((T-p)**2):.3e}")