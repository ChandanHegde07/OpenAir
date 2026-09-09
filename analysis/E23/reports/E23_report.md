# E23 — METAR stacked into E20 experts

**Project:** OpenAir
**Experiment:** E23
**Validation:** Jan+Jul 2025 training holdout
**Stress:** December 2025
**Data:** 12 `training_*.parquet` + Iowa Mesonet ASOS/METAR (same join as E22).
No ranking/submitting in fits.
Script: `experiments/run_e23_metar_ensemble.py`.

## Question

E22 added METAR to the E18-H LightGBM residual and got Jan+Jul −0.56 s
(tail unchanged) vs December −9 s / >30 min 716→651. E20's winning blend
does not use that LightGBM (weight 0). This experiment puts the **same
METAR columns** into the three experts that actually carry weight:
CatBoost residual (C, 0.456), per-airport LightGBM (E, 0.491), XGBoost
residual (D, 0.053). Not a fourth model. NNLS is refit on the
METAR-augmented C/D/E. Frozen E20 weights are the control.

Risk to catch: 12 extra columns overfit Jan+Jul. December >30 min gain
must still be visible **inside the ensemble**, not only on a single LGB.

## Setup

- Join: E22 `decode_and_join` (AOBT else MVT, stale >3 h nulled). Row
  order restored so `time_es_split` ties match E20.
- Hygiene: matched-only `geo_mean`, LIRF-override rows dropped from
  every expert, always-on LIRF `MVT−SCHED`.
- Same hparams as E20 for C/D/E. A/A2/B are not trained and not blended.
- No-wx experts are **E20 OOF** (not retrained). METAR C/D/E are fit here.
- Frozen weights: C=0.456, D=0.053, E=0.491.
- NNLS fit on Jan+Jul holdout only; December is transfer.

## Experts (with vs without METAR)

### Jan+Jul 2025

| Expert | no METAR overall | +METAR overall | Δ | no METAR matched | +METAR matched | >30 no | >30 wx |
|---|---:|---:|---:|---:|---:|---:|---:|
| C CatBoost | 370.37 | 371.19 | +0.82 | 248.25 | 249.00 | 782.6 | 792.5 |
| D XGB | 385.50 | 385.00 | -0.50 | 266.24 | 264.59 | 852.9 | 846.8 |
| E airport | 370.52 | 370.16 | -0.36 | 247.76 | 247.73 | 781.7 | 786.0 |

### December 2025

| Expert | no METAR overall | +METAR overall | Δ | no METAR matched | +METAR matched | >30 no | >30 wx |
|---|---:|---:|---:|---:|---:|---:|---:|
| C CatBoost | 232.37 | 225.49 | -6.89 | 218.56 | 211.86 | 689.7 | 635.6 |
| D XGB | 246.17 | 236.86 | -9.31 | 232.70 | 223.25 | 738.2 | 682.4 |
| E airport | 229.94 | 224.03 | -5.91 | 215.90 | 209.89 | 665.2 | 615.2 |

Airport expert E, matched RMSE by airport (does weather land locally?):

| Airport | Jan+Jul E | Jan+Jul E+wx | Δ | Dec E | Dec E+wx | Δ |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 196.13 | 196.33 | +0.20 | 179.05 | 173.17 | -5.88 |
| EDDM | 173.18 | 171.95 | -1.23 | 224.43 | 202.41 | -22.02 |
| EGLL | 259.93 | 260.75 | +0.82 | 231.17 | 229.99 | -1.19 |
| EHAM | 181.77 | 180.31 | -1.46 | 167.25 | 154.72 | -12.53 |
| LEBL | 224.35 | 223.88 | -0.47 | 181.75 | 182.42 | +0.67 |
| LEMD | 178.82 | 178.75 | -0.06 | 166.52 | 166.00 | -0.52 |
| LFPG | 280.25 | 278.19 | -2.06 | 259.63 | 245.01 | -14.62 |
| LIRF | 444.79 | 446.97 | +2.18 | 316.59 | 317.40 | +0.81 |
| LSZH | 201.37 | 200.32 | -1.05 | 180.08 | 168.41 | -11.68 |
| LTFM | 253.60 | 254.41 | +0.82 | 221.47 | 221.04 | -0.43 |

## Blends

### Jan+Jul 2025

| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE>30 | SSE>60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E20 frozen (oof) | 368.03 | 244.76 | 2214.08 | 774.8 | 2361.4 | 44.7% | 21.3% |
| C/D/E no-wx + frozen w | 368.03 | 244.76 | 2214.08 | 774.8 | 2361.4 | 44.7% | 21.3% |
| C/D/E +METAR + frozen w | 368.12 | 244.99 | 2213.45 | 781.3 | 2384.3 | 45.4% | 21.6% |
| C/D/E +METAR + NNLS | 368.07 | 244.96 | 2213.11 | 781.0 | 2385.6 | 45.4% | 21.7% |

### December 2025

| Model | Overall | Matched | Unmatched | >30 RMSE | >60 RMSE | SSE>30 | SSE>60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E20 frozen (oof) | 228.69 | 214.78 | 816.18 | 672.6 | 2031.7 | 37.9% | 10.4% |
| C/D/E no-wx + frozen w | 228.69 | 214.78 | 816.18 | 672.6 | 2031.7 | 37.9% | 10.4% |
| C/D/E +METAR + frozen w | 221.78 | 208.00 | 798.81 | 619.4 | 1983.7 | 34.3% | 10.5% |
| C/D/E +METAR + NNLS | 221.71 | 207.93 | 798.93 | 618.5 | 1981.4 | 34.2% | 10.5% |

NNLS weights on METAR experts (C, D, E), fit Jan+Jul only: **C=0.368, D=0.075, E=0.557** (E20 was C=0.456, D=0.053, E=0.491).

## METAR gain (Jan+Jul)

CatBoost C:

- `met_tmpf`: 7.6
- `met_relh`: 2.0
- `met_sknt`: 0.8
- `met_vsby`: 0.6
- `met_precip`: 0.4
- `met_age_min`: 0.2
- `met_gust`: 0.1
- `met_low_vis`: 0.0
- `met_fog`: 0.0
- `met_p01i`: 0.0
- `met_strong_wind`: 0.0
- `met_missing`: 0.0

XGBoost D:

- `met_precip`: 20975784.0
- `met_tmpf`: 4199160.0
- `met_low_vis`: 3882496.8
- `met_relh`: 3604448.0
- `met_strong_wind`: 3220740.2
- `met_age_min`: 2953717.5
- `met_sknt`: 2433931.0
- `met_vsby`: 2129548.8
- `met_fog`: 1361058.4
- `met_gust`: 852040.9
- `met_p01i`: 0.0
- `met_missing`: 0.0

Airport E — `met_precip` / `met_tmpf` / `met_relh` gain by airport (top 3 shown in findings JSON).

## Reading

The stacking risk on Jan+Jul is real, even if overall only moved +0.04 s:

- CatBoost **got worse** with METAR (370.37→371.19, >30 783→793).
- Airport experts barely moved (−0.36 overall) and **worsened** >30 (782→786).
- Ensemble >30 min 774.8→781.0; SSE share 44.7%→45.4%. Same Jan+Jul tail failure as E22.

December is the winter effect E22 saw, and it **does survive inside the ensemble**:

- All three experts improve (C −6.9, D −9.3, E −5.9).
- Blend 228.69→221.71 (−7.0), matched 214.78→207.93, >30 672.6→618.5.
- Airport E gains land where winter weather is a thing: EDDM −22, LFPG −15, EHAM −13, LSZH −12. LEBL / LEMD / LTFM / LIRF ≈ 0.

NNLS, fit on Jan+Jul, still shifts weight toward E (0.491→0.557) and away from C (0.456→0.368). Frozen E20 weights on METAR experts are almost the same as the refit (Jan+Jul 368.12 vs 368.07). The blend is not the problem; the ranking-analogue split is.

E20 December OOF with rounded weights 0.456/0.053/0.491 is 228.69 (journal 228.45). Jan+Jul matches **368.03** exactly. Comparison is to this OOF reconstruction.

CatBoost ran on GPU; D and E on CPU. The seasonal pattern is the same on all three experts, so GPU vs E20-CPU is not the story.

## Decision

**Verdict:** INCONCLUSIVE

E20→E23-NNLS Jan+Jul 368.03→368.07 (+0.04), matched 244.76→244.96 (+0.20); >30 774.8→781.0; SSE>30 44.7%→45.4%. Dec 228.69→221.71 (−6.98), matched 214.78→207.93 (−6.85); >30 672.6→618.5 (survives in ensemble). Frozen-w on wx: Jan+Jul +0.09, Dec −6.92. NNLS C/D/E=0.368/0.075/0.557.

Accept only if the METAR-augmented blend beats E20 on **both** splits
without regressing matched/overall, and without hurting Jan+Jul (the
overfit risk). Readme current-model is updated only on ACCEPT.

Do not ship METAR in the production ensemble. Ranking is Jan+Jul-like
(leaderboard 317 vs holdout 368). A December-only win is the wrong split
to chase. METAR remains a documented optional winter feature, not a
contest-RMSE upgrade.

Artifacts: `analysis/E23/`, `experiments/results/E23.json`.

