# PRC 2026 Discovery Report — Aircraft Taxi-Out Time

**Status:** discovery only. No final model trained.  
**Data:** 12 monthly training files (2025), `ranking.parquet` (Jan + Jul 2026), `submitting.parquet`.  
**Metric:** RMSE on taxi-out seconds for ranking departures.

---

## 1. Dataset summary

The files are a left join of airport **movement** columns (`*_mvt`) and EUROCONTROL Network Manager **flight** columns (`*_flt`).

| File | Rows | Cols | Period |
|---|---:|---:|---|
| 12 × `training_YYYY-MM-...parquet` | **4,167,797** | 30 | 2025-01-01 → 2025-12-31 |
| `ranking.parquet` | **689,534** | 30 (same schema) | Jan 2026 (305,590) + Jul 2026 (383,944) |
| `submitting.parquet` | **344,841** | 2 | template: `MVT_ID_mvt`, `TAXITIME_SEC_mvt` (all null) |

Monthly training schemas are **identical**. Ranking schema is **identical** to training. Submitting has only the ID and the target to fill. No columns appear or disappear across months. Missingness of NM fields is stable (~0.8–1.1% per month).

Training is balanced by phase: **2,085,047 DEP** and **2,082,750 ARR**. Ranking: **344,841 DEP** (exactly the submitting IDs) and **344,693 ARR**.

Ten airports, both arrivals and departures:

| Airport | Training DEP | Training ARR | Ranking DEP |
|---|---:|---:|---:|
| LTFM Istanbul | 272,965 | 273,886 | 47,581 |
| EHAM Schiphol | 247,951 | 247,706 | 38,182 |
| LFPG CDG | 239,552 | 238,835 | 39,872 |
| EGLL Heathrow | 239,546 | 239,511 | 39,840 |
| EDDF Frankfurt | 230,141 | 230,122 | 36,317 |
| LEMD Madrid | 212,242 | 211,091 | 36,954 |
| LEBL Barcelona | 179,705 | 179,081 | 30,080 |
| EDDM Munich | 167,334 | 167,323 | 25,966 |
| LIRF Fiumicino | 160,704 | 160,504 | 26,899 |
| LSZH Zurich | 134,907 | 134,691 | 23,150 |

There are **no coordinates, taxiways, positions, weather, or trajectories** in the provided files.

### Schema (training / ranking)

| Column | Type | Meaning | Missing % | Unique (train) | Usefulness |
|---|---|---|---:|---:|---|
| `MVT_ID_mvt` | float | Unique movement ID | 0 | 4,167,797 | Join / submission key |
| `FLIGHT_ID_mvt` | float | NM flight ID (if matched) | 0.74 | 3,821,596 | Link DEP↔ARR of same city-pair flight |
| `FLIGHT_mvt` | str | Flight number | 0.01 | 74,533 | Weak ID; not a feature by itself |
| `FLIGHT_RULE_mvt` | str | I / V | 0.02 | 3 | Almost all IFR; low value |
| `ADEP_mvt` | str | Departure airport | ~0 | 1,582 | **Airport of interest if PHASE=DEP** |
| `ADES_mvt` | str | Destination airport | ~0 | 1,569 | City-pair / long-haul proxy |
| `PHASE_mvt` | str | DEP / ARR | 0 | 2 | Filter: we predict DEP only |
| `MVT_TIME_UTC_mvt` | ts | Takeoff (DEP) or landing (ARR) | **0** | 3,588,175 | **Known in ranking. Second-level.** |
| `BLOCK_TIME_UTC_mvt` | ts | Off-block (DEP) or in-block (ARR) | ~0 train; **100% of ranking DEP** | 3,337,095 | **Target ingredient. Leak if used.** |
| `SCHED_TIME_UTC_mvt` | ts | Scheduled dep/arr | 0 | 108,149 | Known; mostly 5-min slots |
| `AIRCRAFT_TYPE_mvt` | str | ICAO type | 0.04 | 272 | Geometry + wake proxy |
| `RUNWAY_mvt` | str | Runway used | **0** | 58 | **Known in ranking. Core geometry.** |
| `STAND_mvt` | str | Stand/gate | ~0 | 1,924 | **Known in ranking. Core geometry.** |
| `TAXITIME_SEC_mvt` | i32 | Taxi-out (DEP) or taxi-in (ARR), seconds | ~0 train; **100% of ranking DEP** | 5,346 | **TARGET for DEP** |
| `LOBT_flt` | ts | Last known off-block (NM) | 0.91 | 315,483 | Minute-level; weaker than AOBT |
| `CALLSIGN_flt` | str | Callsign | 0.91 | 73,055 | ID |
| `ADEP_flt` / `ADES_flt` | str | NM origin/dest | 0.91 | ~1,450 | Cross-check; DEP ADEP always matches |
| `ADES_FILED_flt` | str | Filed destination | 0.91 | 1,437 | Diversion flag (~2k rows) |
| `MARKET_SEGMENT_flt` | str | Mainline / Lowcost / … | 0.91 | 9 | Operational behaviour |
| `IOBT_flt` | ts | Initial off-block | 0.91 | 239,576 | Flight-plan; minute-level |
| `FLIGHT_RULE_flt` | str | I/V/Y/Z | 0.91 | 5 | Almost all I |
| `FLIGHT_TYPE_flt` | str | S/N/G/M/X | 0.91 | 6 | Mostly scheduled |
| `AIRCRAFT_TYPE_flt` | str | NM type | 0.91 | 243 | Prefer `*_mvt` (more complete) |
| `WK_TBL_CAT_flt` | str | L/M/H/J | 0.91 | 5 | Strong mean shift |
| `AIRCRAFT_OPERATOR_flt` | str | Anonymized airline | 0.91 | 683 | Airline procedures |
| `EOBT_1_flt` | ts | Estimated off-block (FPL M1) | 0.91 | 417,744 | Planned; weaker than AOBT |
| `ARVT_1_flt` | ts | Planned arrival (M1) | 0.91 | 3,551,227 | Planned airborne duration |
| `AOBT_3_flt` | ts | **NM actual off-block (M3)** | 0.91 train; **1.53% of ranking DEP** | 753,327 | **Best measurement of off-block in ranking** |
| `ARVT_3_flt` | ts | NM actual arrival (M3) | 0.91 | 2,708,729 | After takeoff; future for DEP |

On disk: ~277 MB training + 44 MB ranking. Fully loadable in Polars.

---

## 2. Target analysis

**Target:** `TAXITIME_SEC_mvt` for `PHASE_mvt == "DEP"`.  
**Unit:** seconds.  
**Identity (verified on every training DEP row):**

```
TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt
```

100% exact. Taxi-out is airport-reported takeoff minus airport-reported off-block.

For ARR, taxi-in is `BLOCK - MVT` (in-block minus landing), not `MVT - BLOCK`. Arrivals are **not** scored. They are context.

### DEP taxi-out distribution (n = 2,085,047)

| Stat | Seconds | Minutes |
|---|---:|---:|
| Mean | 991.2 | 16.5 |
| Median | 912 | 15.2 |
| Std | 546.4 | 9.1 |
| Min | −12 | — |
| Max | 131,167 | 36.4 h |
| P5 | 480 | 8.0 |
| P25 | 719 | 12.0 |
| P75 | 1,188 | 19.8 |
| P90 | 1,471 | 24.5 |
| P95 | 1,684 | 28.1 |
| P99 | 2,339 | 39.0 |
| P99.9 | 4,501 | 75.0 |

**Shape:** right-skewed, heavy-tailed, unimodal around 12–18 minutes. Not normal. Not obviously multimodal globally (airport mixture produces some bumpiness). Mean−median = 79 s, (mean−median)/std = 0.15.

**RMSE is dominated by the tail.** Against a mean predictor:

- 3.56% of flights > 30 min produce **66.9%** of SSE
- 0.20% of flights > 1 h produce **49.6%** of SSE
- 1.08% unmatched-to-NM flights produce **42.2%** of SSE
- LIRF is 7.7% of rows and **46.9%** of SSE
- LIRF unmatched alone is **37.4%** of SSE

A model that is merely “good on typical 15-minute taxis” will lose the leaderboard to a model that handles LIRF / unmatched / >1 h cases.

369 negative values (mostly LSZH) and 19 zeros: data glitches, RMSE-irrelevant. Clip at 0 for training labels if desired; ranking will have a few too.

---

## 3. Prediction-time information set

Official ranking construction (competition data page):

> `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` for departures have been blanked out.

Verified: ranking DEP has **0%** `BLOCK_TIME` and **0%** `TAXITIME`. Ranking ARR has **both fully present**. `MVT_TIME` (takeoff) is present for **every** ranking departure. `AOBT_3_flt` is present for **98.47%** of ranking departures.

The rationale page says the model is for **post-operations analysis** (constrained intervals, excess fuel), not a gate-push real-time tool.

**Prediction moment:** after the airport movement is observed, with takeoff time, runway, and stand known, reconstruct taxi-out. Equivalently: reconstruct off-block, then `taxi = takeoff - off-block`.

### PREDICTION INFORMATION SET

For scored flight *i* (DEP), everything non-null in ranking for *i*, plus all other ranking/training rows **except other DEP `BLOCK_TIME` / `TAXITIME`**.

**Safe for flight i**

- Takeoff time `MVT_TIME_UTC_mvt`
- Scheduled time
- Runway actually used, stand actually used, aircraft type
- NM fields: `AOBT_3`, `EOBT_1`, `IOBT`, `LOBT`, callsign, operator, WTC, market segment, planned/actual arrival
- All **arrival** movements at the same airport: landing, in-block, taxi-in, stand, runway
- All **other departures’ takeoff times, runways, stands, NM AOBT**

**Not available**

- Airport off-block (`BLOCK_TIME`) of *i* or of any other ranking DEP
- Airport taxi-out of *i* or of any other ranking DEP
- Anything after takeoff that is not already in the row (`ARVT_3` is after takeoff; do not use it as if it were known at pushback — it is in the file, but it is the flight’s own arrival at destination)

This is the rule for all future feature engineering.

---

## 4. Leakage audit

| Column | Class | Why |
|---|---|---|
| `TAXITIME_SEC_mvt` (DEP) | **C** | The target. Blank in ranking. |
| `BLOCK_TIME_UTC_mvt` (DEP) | **C** | `target = MVT − BLOCK`. Blank in ranking. Using it is 100% leak. |
| `MVT_TIME_UTC_mvt` (DEP) | **A** | Present in ranking. Needed to form `MVT − AOBT`. |
| `SCHED_TIME`, stand, runway, type, airline, WTC | **A** | Present in ranking. |
| `AOBT_3_flt` | **A** (competition) / **B** (real-time pushback) | NM actual off-block, **not** equal to airport BLOCK (MAE 238 s, exact match 0.65%). Organizers blanked BLOCK and left AOBT. Using it is the intended measurement-reconciliation problem. |
| `LOBT_flt`, `EOBT_1_flt`, `IOBT_flt` | **A** | Weaker off-block estimates. Present in ranking. |
| `ARVT_3_flt` of this DEP | **C** as a taxi feature | Arrival at destination, after taxi-out and airborne. Present in ranking but post-event. Planned `ARVT_1` is safer as a long-haul flag. |
| Other DEP `TAXITIME` / `BLOCK` | **C** at test | Blank for **all** ranking DEP. Rolling actual taxi-out of neighbours is **not** reproducible. |
| Other DEP `MVT_TIME`, `AOBT_3`, stand, runway | **A** (batch) | Present. Rolling `MVT−AOBT` of previous departures **is** reproducible. |
| All ARR fields including taxi-in | **A** | Ranking ARR is complete. |
| `FLIGHT_ID` ARR at destination | **C** if used as “this flight has landed” | That ARR is after this DEP. |
| Previous unrelated ARR at same stand | **A** if timestamp < this takeoff | Turnaround proxy. |
| IDs as numeric values | **C** | `MVT_ID` looks like epoch-ish floats; do not use as a feature. |
| Rolling stats that include future rows | **C** | For flight *t*, only movements with time `< t` (or `≤ t` excluding self). |

**Hidden-target check:** `AOBT_3` is not a disguised copy of `BLOCK`. Median `BLOCK − AOBT` = +51 s, MAE = 238 s, RMSE if we predict `MVT − AOBT` = **384.9 s** vs mean-predictor **546.4 s**. It is a noisy, biased measurement of the same event, from a different system, at **minute** resolution (97.7% of AOBT values sit on `:00`). Airport BLOCK/MVT are second-level.

---

## 5. Temporal structure

Yes: per airport, movements are a second-resolution event stream.

- Peak hourly movements: ~70–90/h at big hubs (EDDF, EGLL, LTFM, EHAM).
- Daily departures: ~370 (LSZH) to ~750 (LTFM).
- Median gap between successive takeoffs: **64–119 s** depending on airport. 26% of gaps < 1 min, 69% < 2 min.
- Minute-level airport state is reconstructable from takeoff/landing timestamps.

**How to compute leak-free rolling features for flight *i* at airport *a*, reference time `t = MVT_TIME_i`:**

Count events with timestamp in `(t − W, t]` excluding *i*. Use only fields present in ranking.

| Feature | Event time | Ranking-safe? |
|---|---|---|
| Takeoffs in last 1/5/10/15/30/60 min | other DEP `MVT_TIME` | Yes |
| Same-runway takeoffs in those windows | other DEP `MVT_TIME` + `RUNWAY` | Yes |
| Landings in those windows | ARR `MVT_TIME` | Yes |
| Pushbacks in those windows | other DEP `AOBT_3` | Yes |
| Rolling mean of `MVT − AOBT` of previous DEPs | other DEP AOBT+MVT | **Yes — this is the usable “recent taxi”** |
| Rolling mean of actual `TAXITIME` of previous DEPs | other DEP TAXITIME | **No** |
| Rolling mean of recent taxi-in | ARR TAXITIME | Yes, but empirically weak |

Reference time alternatives: `AOBT_3` (start of taxi, more operationally correct) or `EOBT`/`SCHED` (if AOBT missing). Do **not** count takeoffs *during* `[AOBT_i, MVT_i]` as a feature: that count scales with the target (endogenous).

---

## 6. Airport-state reconstruction

What we **can** reconstruct immediately before / at takeoff of *i*:

| Component | Feasible? | How |
|---|---|---|
| Traffic volume | Yes | takeoffs/landings in 5–60 min |
| Traffic composition | Partial | WTC / type of recent movers |
| Queue pressure | Partial | count of other DEP with `AOBT_j < MVT_i ≤ MVT_j` (still taxiing at *i*’s takeoff). Uses other flights’ takeoff times (available in batch ranking). Corr with target **0.26**. |
| Runway pressure | Yes | same-runway takeoff counts; corr up to **0.17** at 60 min |
| Recent taxi-time behaviour | Yes, via NM | rolling `MVT−AOBT` of previous DEPs. Per-airport corr with target **0.43–0.45** at EGLL/EHAM |
| Recent taxi-in | Yes | available; weak / unstable (EHAM −0.28, others ~0.1) |
| Exact aircraft positions | **No** | no lat/lon, no taxiway, no ADS-B in this dump |
| Gate/stand occupancy | Partial | last ARR in-block vs next DEP at same stand |
| Weather / ATFM regulations | **Not in files** | would be external |

Proposed `S_t` (feasible):

```
S_t = [
  n_dep(1,5,15,30,60),
  n_dep_same_rwy(5,15,30),
  n_arr(15,30),
  queue_at_takeoff,          # overlapping [AOBT, MVT]
  roll_mean/median/p90 of (MVT-AOBT) over last 5/10/20 DEPs at airport and at runway,
  n_unmatched_recent,
  dep_rate_15 / dep_rate_60  # regime
]
```

This is feasible. It is **not** a full surface snapshot.

---

## 7. Baseline / residual decomposition

`Taxi ≈ T_geometry(airport, stand, runway, WTC) + T_measurement(MVT − AOBT) + D_congestion + D_disruption`

### Naive predictors (training DEP)

| Predictor | RMSE | MAE | Notes |
|---|---:|---:|---|
| Global mean | 546.4 | 302 | |
| Airport mean (in-sample) | 514.8 | 266 | |
| Stand×runway mean (in-sample) | 469.0 | 206 | overfit-prone; 12,053 groups |
| **`MVT − AOBT_3`** | **384.9** | **238** | no training |
| `MVT − EOBT` | 676.7 | 465 | worse than mean |
| `MVT − LOBT` | 740.1 | 515 | worse than mean |
| `MVT − SCHED` | 2,413 | 1,035 | useless as taxi |

AOBT is the only NM timestamp that beats the mean. EOBT/LOBT/SCHED are **not** taxi.

OLS `y ~ aobt_proxy` shrinks the proxy (slope ≈ 0.59, intercept ≈ 390). That is errors-in-variables: AOBT is a noisy minute-level measurement.

### Time-split (not in-sample)

Train all 2025 months except the holdout.

| Holdout | Mean | Stand×rwy | AOBT proxy | OLS AOBT | OLS AOBT + roll10(MVT−AOBT) |
|---|---:|---:|---:|---:|---:|
| Dec | 515 | 449 | 375 | 340 | **328** |
| Jan | 605 | 554 | 377 | 347 | **337** |
| Jul | 746 | 684 | 466 | 433 | **412** |
| **Jan+Jul (ranking analogue)** | **687** | **628** | **428** | **396** | **378** |

Adding recent taxi-in to the OLS changes RMSE by < 1 s. Do not chase it.

**Residual of AOBT is airport-systematic, not just noise:**

| Airport | mean(y − proxy) | RMSE(proxy) |
|---|---:|---:|
| LTFM | −236 s (proxy too long) | 531 |
| EHAM | −152 | 330 |
| EDDF | −53 | 255 |
| EGLL | −23 | 419 |
| LEMD | +34 | 276 |
| LFPG | +35 | 380 |
| LSZH | +59 | 268 |
| LEBL | +68 | 311 |
| EDDM | +82 | 349 |
| **LIRF** | **+214** | **557** |

Calibrate per airport. LIRF and LTFM are the worst measurement-reconciliation airports.

Congestion counts have **weak** correlation with raw taxi (0.05–0.17) and only slightly stronger with stand–runway residual. They help a little **after** AOBT is in the model, not before. The residual is **not** “mostly congestion.” A large part is measurement bias + LIRF/unmatched disruptions.

---

## 8. Congestion-regime analysis

Regimes exist but they are **modest and airport-specific**. They are not five clean worlds.

Global quintiles of takeoffs in previous 15 min: Q1 mean taxi 903 s vs higher quintiles a few tens of seconds more. After stand–runway subtraction: Q1 residual −66 s, Q5 +33 s.

Locally:

- **EGLL:** Q1 mean 1,265 vs Q5 1,399 (≈ 2 min). Queue corr **0.34**.
- **LFPG:** 944 → 1,096.
- **EDDF:** 818 → 900.
- **EHAM:** essentially flat (781–788). Volume does not explain taxi there.
- **LIRF:** volume barely related to the insane tail.

`P(taxi | high volume)` is shifted, not a different family. This supports **conditional features** and maybe **airport-specific models**, not a five-expert MoE. Quantile regression could help the EGLL/LFPG upper tail; it will not fix LIRF unmatched 2-hour errors.

---

## 9. Graph feasibility

**Cannot reconstruct an actual taxi route from this dataset.** Missing: taxiway IDs, intersections, holding points, coordinates, trajectories, segment times, path sequences.

What **can** be derived without external maps:

- Bipartite graph: stand → runway at each airport (12,053 observed pairs; 6,436 with n≥30).
- Empirical traversal time per pair: this **is** `T_geometry`.
- Same stand, different runways: mean taxi differs by **302 s on average** (p90 563 s). Runway choice is a real distance/path effect.
- Stand×runway group means explain **η² ≈ 0.25** of taxi variance (in-sample, n≥30 groups). Airport alone: 0.11.

With **external** airport maps (open OSM/AIP) one could add great-circle or taxiway-network distance stand→runway. That is a later add-on, not in the parquet files.

GNN: **not supported by the dataset alone.** A 2-node path (stand, runway) does not need a GNN.

---

## 10. Segment-level time-to-traverse

**Not achievable from these files.** We observe only two timestamps per departure (off-block, takeoff). No intermediate points. Cannot split Gate→Apron→Taxiway→Hold→Runway.

The only honest decomposition is:

```
taxi = unimpeded(stand, runway, type) + queue_hold + measurement_error + disruption
```

That is a **two- or three-component** model, not N segments.

---

## 11. Temporal-model feasibility

- Per airport: ~400–750 DEP/day, gaps ~1 min, 365 days.
- Sequences of 30–60 minutes contain 10–40 departures: enough for a TCN/LSTM **in principle**.
- Ranking-safe sequence features are counts and `MVT−AOBT`, not actual taxi-out of neighbours.
- Linear `AOBT + roll10(MVT−AOBT)` already captures a large fraction of that signal (Jan+Jul RMSE 378 vs 428).

A TCN on raw sequences is unlikely to beat a tree on well-built rolling features, and is much easier to leak with. **Do not start here.**

---

## 12. MoE feasibility

Congestion experts: **not justified.** Mean shifts of 1–3 minutes, EHAM has no volume effect, LIRF tail is not a congestion regime.

What **is** justified as separate functions:

1. **Matched vs unmatched** (null NM / null AOBT) — different data generating process. Unmatched LIRF median 3,922 s vs matched LIRF ~1,000 s.
2. **Airport groups:** EGLL (structurally long), LIRF (heavy tail + data quality), LTFM (AOBT bias), compact airports (LSZH, EHAM, EDDM).
3. Optional: **WTC J/H vs M/L**.

That is closer to “two-stage / airport models” than a learned five-expert MoE. A MoE would be complexity without evidence.

---

## 13. Hard-case analysis

| Slice | n | Who | What |
|---|---:|---|---|
| > 30 min | 74,167 (3.6%) | EGLL 25k, LIRF 13k, LTFM 12k, LFPG 9k | EGLL is structural (median already 22 min). |
| > 1 h | 4,126 (0.20%) | **LIRF 1,967**, LTFM 794, EGLL 728 | 26% have **null NM**. Mean dep-15m is *lower* than average. Not a rush-hour story. |
| > 2 h | 584 | LIRF 480 | **481/584 unmatched** |
| Unmatched NM | 22,470 (1.08%) | mean 1,384, std **3,398** | **42% of mean-model SSE** |
| Unmatched LIRF | 1,488 | median **3,922**, mean **6,531** | **37% of mean-model SSE** |

Callsigns on the extremes look like ferry/positioning/GA (`NOSOS739`, `CHHHH438`, `BAWW561D`). Rome unmatched is a different population, or BLOCK/MVT from different events. Ranking has **383 unmatched LIRF** (similar rate).

**Dynamic surface-state will not fix these.** They are not “queue of 15 jets.” They need:

- an `unmatched` flag,
- LIRF-specific unmatched predictor (high),
- possibly clipping / Huber training **plus** a dedicated tail model, because RMSE still scores the raw seconds.

EGLL > 30 min **is** the case surface-state can help (queue corr 0.34, roll10 AOBT corr 0.43).

---

## 14. Top 20 most promising features

All ranking-safe unless noted.

1. **`MVT_TIME − AOBT_3`** (clip e.g. 0–7200) — by far #1  
2. **Airport ID** — calibration of AOBT bias  
3. **Stand × runway** empirical mean / count (fallback airport×runway)  
4. **`unmatched_nm` flag** (null AOBT)  
5. **Airport × unmatched interaction** (LIRF unmatched!)  
6. **Rolling mean of previous 10 `MVT−AOBT` at airport**  
7. **Rolling mean of previous 10 `MVT−AOBT` at same runway**  
8. **Queue at takeoff** (count still-taxiing via AOBT/MVT overlap)  
9. **Wake category** (J 1,416 / H 1,173 / M 937 / L 700 s)  
10. **Aircraft type** (A388 1,418; E75L 686)  
11. **Same-runway takeoffs in 15–60 min**  
12. **Hour of takeoff (UTC) × airport**  
13. **Airline operator**  
14. **Market segment** (Business/Regional shorter; Mainline/Cargo longer)  
15. **Runway ID** (LTFM 34L/16R much longer than others)  
16. **Destination / long-haul flag** (`ARVT_1 − EOBT` or heavy + distant ADES)  
17. **Takeoffs in 15 and 60 min** (regime, airport-dependent)  
18. **Stand ID** (even without runway)  
19. **Month / season** (weak globally, keep for Jan vs Jul ranking)  
20. **AOBT − EOBT** (NM delay) — weak linear corr, may help trees on disruptions  

**Do not use:** `MVT−EOBT` as a taxi estimate, `MVT−SCHED` as a taxi estimate, other DEP actual taxi-out, this flight’s `ARVT_3`.

---

## 15. Top 5 modeling strategies

1. **Calibrated NM off-block + tree residual** (recommended core)  
   Predict `taxi ≈ f(MVT−AOBT, airport, stand, runway, …)` with LightGBM/CatBoost. Fallback stand–runway mean when AOBT is missing.

2. **Explicit unmatched / LIRF tail model**  
   Separate predictor for `AOBT is null`, especially LIRF. This is an RMSE strategy, not a neural architecture.

3. **Tree + ranking-safe surface state**  
   Add roll10 `MVT−AOBT`, queue, same-runway counts. Time-split already shows ~18–20 s RMSE from roll10 on top of OLS AOBT.

4. **Airport-specific trees or strong airport categorical**  
   EGLL vs EHAM vs LIRF are different problems. One global tree with airport as a key categorical is enough at first; split LIRF if needed.

5. **Stand–runway geometry table + residual**  
   Lookup unimpeded time, model the residual with AOBT error + congestion. Same as (1) with a more interpretable baseline.

---

## 16. Recommended first experiment

**Do not train a GNN. Do not sweep 200 hyperparameters.**

Validation: train on 2025 **except January and July**; validate on **January + July 2025**. That is the closest ranking analogue (and it is harsher than a random month). Also report Dec as a sanity check.

Four predictions, same clip `[0, 7200]`:

| ID | Method |
|---|---|
| B0 | Airport × stand × runway mean, fallback airport mean |
| B1 | `clip(MVT − AOBT)` + per-airport residual mean; B0 if unmatched |
| B2 | LightGBM on **flight-level columns only** (airport, stand, runway, type, WTC, hour, airline, sched). **No AOBT proxy, no rolling.** This is the “conventional CatBoost” competitor. |
| B3 | LightGBM on B2 columns **plus** AOBT proxy, unmatched flag, roll5/10/20 `MVT−AOBT` (airport and runway), queue_at_takeoff, same-rwy counts 15/30, airport×unmatched |

Report RMSE overall / matched / unmatched / LIRF / EGLL / >30 min.

**Success criterion for the edge:** B3 beats B2 by a clear margin on Jan+Jul, and B1 already beats B0/B2. If B3 ≈ B1, stop adding architecture and work on unmatched/LIRF.

---

## 17. Recommended competition architecture

```
inputs (ranking-safe)
        │
        ├─ if AOBT present: aobt_proxy = clip(MVT - AOBT)
        │     surface = rolling(MVT-AOBT), queue, rwy counts
        │     tree / calibrated residual
        │
        └─ if unmatched: LIRF-aware unmatched model
              (stand/rwy/hour/type only; high LIRF intercept)
        │
        └─ clip to [0, 7200] (or airport-specific cap)
```

One gradient-boosted tree is enough for the matched 98.5%. A tiny second model or a large leaf for `unmatched × LIRF` is the RMSE insurance policy.

CatBoost vs LightGBM vs XGBoost: irrelevant at this stage. Use whichever handles categoricals (stand, runway, airport, type) with least fuss. **The edge is the information set, not the booster.**

---

## 18. What information gives the biggest potential edge

1. **NM actual off-block (`AOBT_3`) as a noisy measurement of the airport off-block**, calibrated per airport.  
2. **Recent other departures’ `MVT−AOBT`**, because actual neighbour taxi-out is blank in ranking.  
3. **Unmatched-NM / LIRF extremes**, which dominate SSE and that a generic flight-level model will predict as “normal ~15 min.”  
4. **Stand×runway unimpeded time** (geometry without a map).

---

## 19. What NOT to waste time on

- GNN / taxiway graph from this dataset  
- Segment-level time-to-traverse  
- TCN / LSTM / TFT as the first model  
- Mixture-of-experts with congestion experts  
- Using `EOBT`/`LOBT`/`SCHED` as if they were taxi  
- Rolling **actual** taxi-out of other DEPs (not in ranking)  
- Hyperparameter marathons before B0–B3  
- Predicting with future `ARVT_3`  
- Assuming high traffic is the main source of large errors (it is not; LIRF unmatched is)

External METAR / OSM distances are optional later, not a substitute for 1–3 above.

---

# THE COMPETITION EDGE

If another team trains CatBoost/XGBoost/LightGBM on the provided **flight-level columns** (airport, stand, runway, aircraft type, airline, scheduled time, hour, WTC) they will capture geometry and average airport behaviour. Stand×runway means already get in-sample RMSE ~469, and a time-split stand–runway model sits around **628 RMSE on Jan+Jul**.

They are likely to miss three things that are **in this dataset** and **present in ranking**:

### 1. The second clock

Ranking still contains NM actual off-block `AOBT_3_flt` for 98.5% of departures. Airport taxi-out is `takeoff − airport_offblock`. We also observe `takeoff − NM_offblock`. Those two clocks disagree by **MAE 238 s**, with **airport-specific bias** (LTFM −4 min, LIRF +3.5 min).

`clip(MVT − AOBT)` alone has time-split RMSE **428 on Jan+Jul** vs **628** for stand–runway. OLS calibration drops that to **396**. A flight-level tree that never forms this difference is throwing away the best single number in the file.

This is allowed: the organisers blanked `BLOCK_TIME` and `TAXITIME` and left `AOBT_3`. It is not a copy of the target.

### 2. Neighbour taxi inferred from the same two clocks

Other ranking departures also have takeoff and `AOBT_3`, but **not** taxi-out. A conventional model cannot put “mean taxi of the last 10 departures” in the feature vector at test time.

We can put “mean of `(MVT − AOBT)` of the last 10 departures at this airport / runway.” That series has per-airport correlation with the target of **0.43–0.45 at EGLL and EHAM**. On the Jan+Jul split, adding it to OLS AOBT moves RMSE **396 → 378**.

That is the only ranking-safe version of “what is taxi-out doing right now?”

### 3. The unmatched-LIRF generating process

1.08% of training DEPs have no NM match. They produce **42%** of mean-model squared error. Unmatched LIRF (1,488 rows) have **median 3,922 s** and produce **37%** of that SSE. Ranking has 5,290 unmatched DEPs including **383 at LIRF**.

A standard tree will see stand/runway/hour and predict ~15–20 minutes. If ranking unmatched LIRF behave like training, those rows explode RMSE. Surface-state will not save them. An explicit `unmatched × airport` path will.

### Smallest experiment to test the edge

On **train = 2025 except Jan+Jul, val = Jan+Jul 2025**, compare:

- **B2:** LightGBM on flight-level columns only (the conventional model).  
- **B1:** `clip(MVT − AOBT)` + airport residual; stand–runway fallback if unmatched.  
- **B3:** B2 + AOBT proxy + unmatched flag + roll10 `(MVT−AOBT)` + queue.

If B1 already beats B2, the second clock is real. If B3 beats B1, surface-state from neighbour NM clocks is real. If unmatched RMSE stays huge, build the LIRF unmatched model next — not a neural net.

**Do not train the final deep model yet.** The first question is whether these three reconstructed quantities move Jan+Jul RMSE. The measurements above say they will.
