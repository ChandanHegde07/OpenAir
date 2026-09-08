# E22 — METAR weather (Branch B)

**Project:** OpenAir
**Experiment:** E22
**Validation:** Jan+Jul 2025 training holdout
**Stress:** December 2025
**Data:** 12 `training_*.parquet` + public Iowa Mesonet ASOS/METAR for 2025.
No ranking/submitting in fits.
Script: `experiments/run_e22_metar.py`.

## STEP 0 — data audit

### What is on disk

`data/` contains exactly:

- 12 `training_*.parquet` files (calendar 2025)
- `ranking.parquet`
- `submitting.parquet`
- `external/metar/{ICAO}.csv` (this experiment; Iowa Mesonet ASOS 2025)

**`air-data/` does not exist** in the repo, parent `prc/`, or anywhere
searched. A filename search for ads-b / trajectory / lat-lon files under
OpenAir returned nothing.

### Training schema (30 columns, confirmed on `training_2025-01-01_…`)

Movement clocks and airport fields: `MVT_ID_mvt`, `FLIGHT_ID_mvt`,
`FLIGHT_mvt`, `FLIGHT_RULE_mvt`, `ADEP_mvt`, `ADES_mvt`, `PHASE_mvt`,
`MVT_TIME_UTC_mvt`, `BLOCK_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`,
`AIRCRAFT_TYPE_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `TAXITIME_SEC_mvt`.

NM flight-list fields: `LOBT_flt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`,
`ADES_FILED_flt`, `MARKET_SEGMENT_flt`, `IOBT_flt`, `FLIGHT_RULE_flt`,
`FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`,
`AIRCRAFT_OPERATOR_flt`, `EOBT_1_flt`, `ARVT_1_flt`, `AOBT_3_flt`,
`ARVT_3_flt`.

**No lat/lon, no taxiway, no weather, no 1-second ADS-B surface
trajectories.** This edition is flight-list + movement clocks only, unlike
past PRC challenges that shipped daily trajectory parquet.

### Extra-data rules (2026 eligibility)

https://prc-data-challenge-2026.netlify.app/eligibility.html — prize-eligible
solutions require:

- *All used external datasets are openly accessible/usable and documented.*
- *All additional datasets used are openly available under an open source
  license.*

Same policy as 2025 (`The use of additional and/or external dataset is
permitted if open data and documented.`). Iowa Environmental Mesonet
ASOS/METAR is public observational weather (upstream NWS/NOAA + WMO
partners). **Branch B selected.**

## Join

Nearest METAR with `valid ≤ AOBT` (else `MVT`) at the same ICAO station.
Reports older than 3 h are treated as missing and their decoded values are
nulled. Features: visibility, wind, gust, temp, RH, precip amount, age,
flags for low vis / fog / precip / strong wind. Frozen E18-H residual
LightGBM + these columns. No loss change, no E19 rolling-queue revival.

## Coverage

| Airport | n | miss rate | fog | precip | low vis | strong wind | med vis (sm) | med age (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 230,141 | 0.00% | 1.04% | 11.93% | 0.81% | 0.58% | 6.21 | 28.0 |
| EDDM | 167,334 | 0.00% | 3.13% | 15.01% | 2.69% | 0.92% | 6.21 | 28.0 |
| EGLL | 239,546 | 0.00% | 0.32% | 8.65% | 0.27% | 0.85% | 6.21 | 30.0 |
| EHAM | 247,951 | 0.00% | 2.63% | 15.28% | 1.27% | 3.26% | 6.21 | 29.7 |
| LEBL | 179,705 | 0.00% | 0.31% | 6.74% | 0.38% | 0.82% | 6.21 | 29.0 |
| LEMD | 212,242 | 0.00% | 0.38% | 6.39% | 0.23% | 0.17% | 6.21 | 30.0 |
| LFPG | 239,552 | 0.00% | 1.82% | 7.87% | 1.44% | 1.25% | 6.21 | 29.0 |
| LIRF | 160,704 | 0.00% | 0.87% | 5.40% | 0.11% | 0.42% | 6.21 | 30.0 |
| LSZH | 134,907 | 0.00% | 4.88% | 16.74% | 2.44% | 0.17% | 6.21 | 29.0 |
| LTFM | 272,965 | 0.00% | 1.84% | 10.41% | 1.28% | 10.73% | 6.21 | 28.0 |

## Holdout vs E18-H (fair feature test) and E20 (current best)

### Jan+Jul 2025

| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE share >30 | SSE share >60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E18-H | 372.36 | 250.98 | 2216.59 | 798.6 | 2402.4 | 45.2% | 20.9% |
| E22 METAR | 371.81 | 250.11 | 2216.80 | 796.5 | 2398.3 | 45.3% | 21.0% |
| E20 ensemble (ref) | 368.03 | 244.76 | — | — | — | — | — |

### December 2025

| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE share >30 | SSE share >60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E18-H | 238.01 | 223.95 | 838.40 | 716.2 | 2114.6 | 39.6% | 10.3% |
| E22 METAR | 229.04 | 215.27 | 812.81 | 651.1 | 2038.3 | 35.4% | 10.4% |
| E20 ensemble (ref) | 228.45 | 215.90 | — | — | — | — | — |

Frozen-E18-H residual vs weather flags (Jan+Jul matched):

- fog n=6,312 mean resid 47.1 vs no-fog -7.4
- precip n=40,738 mean resid 1.4 vs none -7.5
- low vis n=4,934 mean resid 53.3
- strong wind n=7,870 mean resid -17.2

LightGBM gain on METAR columns (Jan+Jul), share of the METAR family:

- `met_tmpf` 53%, `met_relh` 17%, `met_precip` 17%, `met_vsby` 5%, `met_sknt` 4%, `met_age_min` 3%
- flags (`met_fog` / `met_low_vis` / `met_strong_wind` / `met_gust`) < 0.3% combined
- `met_p01i` and `met_missing` unused (IEM `p01i` is almost always 0; miss rate ≈ 0)

Trees use temperature / humidity / precip presence, not the rare fog flag, even though fog rows have a +47 s E18-H residual.

## Reading

E18-H reproduced exactly after restoring pre-join row order (`time_es_split` ties): Jan+Jul **372.36 / 250.98**, December **238.01 / 223.95**.

The weather residual is real: fog and low-vis matched flights are under-predicted by ~50 s. That does **not** move the contest metric on the ranking analogue:

- Jan+Jul overall −0.56 s, matched −0.87 s. >30 min RMSE 798.6→796.5; >60 min 2402→2398; SSE share of the tail **unchanged** (45.2% / 20.9%). Same failure mode as E16-A / E19: the >30 min tail is not weather-identified on this split.
- December is the winter split: overall −8.98 s, matched −8.67 s, >30 min 716→651, SSE>30 39.6%→35.4%. That is a real seasonal effect, not a split artifact.
- vs current best E20 ensemble: 371.81 vs **368.03** Jan+Jul (does not beat); 229.04 vs **228.45** December (does not beat). December matched 215.27 is slightly better than E20’s 215.90, but overall is not.

Hourly ASOS (median age 28–30 min, ~1 report/hour) is ranking-safe and legal. It is not a 1-second surface trajectory, and it does not replace the ensemble.

## Decision

**Verdict:** INCONCLUSIVE

E18-H→E22 Jan+Jul 372.36→371.81 (−0.56), matched 250.98→250.11 (−0.87); >30 798.6→796.5; >60 2402.4→2398.3; SSE>30 45.2%→45.3%; SSE>60 20.9%→21.0%. Dec 238.01→229.04 (−8.98), matched 223.95→215.27 (−8.67). vs E20 368.03/228.45: does not beat Jan+Jul, does not beat Dec.

Accept only if E22 beats E20 on **both** splits without regressing
matched/overall. Readme current-model is updated only on ACCEPT.
Beating E18-H without beating E20 is INCONCLUSIVE (feature may still be
worth stacking later; it is not the new current model).

Do not run another E18-H residual-only METAR variant. Optional follow-up is
to add the same columns to E20’s CatBoost / airport experts, not to reopen
E15 or E19.

Artifacts: `analysis/E22/`, `data/external/metar/`, `data/external/README.md`.

