"""E71 step 0 — quantify high-SSE unmatched population and probe the
stand-release identity T = since_arr - turnaround on unmatched rows."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAIN_FILES, load_dep, rmse, add_causal_rolling
from matched_submit import add_extra, e20_col
from run_e55_inbound_v2 import load_arr, attach  # attach used matched; here custom

def log(m): print(m, flush=True)

# ---------- load training DEP with target + identity fields ----------
dep = (pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES])
       .filter(pl.col("PHASE_mvt")=="DEP")
       .select(["MVT_ID_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt",
                "BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
                "MARKET_SEGMENT_flt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt","FLIGHT_mvt"])
       .collect())
dep = dep.with_columns([
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
    (pl.col("MVT_TIME_UTC_mvt")-pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("y_from_block"),
    pl.col("AOBT_3_flt").is_null().alias("unmatched"),
    pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"),
])
# sanity: y == y_from_block?
dd=(dep["y"]-dep["y_from_block"]).abs()
print("T vs MVT-BLOCK: max abs diff", float(dd.max()), "n mismatch>1", int((dd>1).sum()))

# ---------- previous ARR at same stand ----------
arr = (pl.concat([pl.scan_parquet(p) for p in TRAIN_FILES]).filter(pl.col("PHASE_mvt")=="ARR")
       .select([pl.col("ADES_mvt").alias("airport"),"STAND_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
                pl.col("MVT_TIME_UTC_mvt").alias("arr_mvt")])
       .drop_nulls(["airport","STAND_mvt","aibt"]).collect().sort(["airport","STAND_mvt","aibt"]))
d = dep.rename({"ADEP_mvt":"airport"}).sort(["airport","STAND_mvt","MVT_TIME_UTC_mvt"])
j = d.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="aibt", by=["airport","STAND_mvt"], strategy="backward")
j = j.with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("aibt")).dt.total_seconds().alias("since_arr"))
# causal validity: aibt must be <= MVT
j = j.with_columns(pl.when(pl.col("since_arr")<0).then(None).otherwise(pl.col("since_arr")).alias("since_arr"))
um=j["unmatched"].to_numpy().astype(bool); ap=j["airport"].to_numpy()
sa=j["since_arr"].to_numpy().astype(float); T=j["y"].to_numpy().astype(float)
cov=np.isfinite(sa)
print(f"\ncoverage since_arr: all={cov.mean():.3f} unmatched={cov[um].mean():.3f} matched={cov[~um].mean():.3f}")
print(f"coverage since_arr nonLIRF-u={cov[um&(ap!='LIRF')].mean():.3f} LIRF-u={cov[um&(ap=='LIRF')].mean():.3f}")
# identity T = since_arr - turnaround  => turnaround = since_arr - T
turn = sa - T
for nm,m in (("all",np.ones(len(T),bool)),("unmatched",um),("nonLIRF-u",um&(ap!="LIRF")),("LIRF-u",um&(ap=="LIRF")),("matched",~um)):
    mm=m&cov&np.isfinite(T)
    if mm.sum()==0: continue
    t=turn[mm]
    print(f"{nm:10s} n={mm.sum():7d} since_arr med={np.median(sa[mm]):8.0f} T med={np.median(T[mm]):7f} turnaround med={np.median(t):8.0f} "
          f"p10={np.quantile(t,.1):7.0f} p90={np.quantile(t,.9):8.0f} neg_frac={np.mean(t<0):.3f}")
    # how well does since_arr alone predict T?
    print(f"           corr(since_arr,T)={np.corrcoef(sa[mm],T[mm])[0,1]:.3f}  RMSE(since_arr,T)={rmse(T[mm],sa[mm]):8.1f}")
