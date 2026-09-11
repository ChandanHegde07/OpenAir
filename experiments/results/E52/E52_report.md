# E52 — ranking-safe surface congestion into v13: report

## Goal
Feed strongest surface-congestion / recent-history features into v13's Δ-hat
and leftover residual: exact active-surface counts at AOBT (interval overlap,
ranking-safe), rolling mean-P (= MVT−AOBT of recent flights, a ranking-safe
taxi-speed proxy) over 10/30 min per airport/runway, dep/arr pressure, heavy
active. NO rolling TAXITIME (not ranking-safe).

## Results (cross-month OOF, same harness as v13)
| split | E20 matched | v13 (E50) | E52 (v13 + congestion) |
|---|---:|---:|---:|
| Jan+Jul | 244.76 | 238.52 | **238.50** (Δ −6.26 vs E20; top1 −15.1%, top5 −8.9%) |
| Dec | 214.78 | 212.08 | **212.11** (Δ −2.66) |

## Decision: no gain, no v15
Congestion features are redundant with v13's E36-clocks + e20 feature set:
matched Jan+Jul 238.50 vs 238.52, Dec 212.11 vs 212.08 — essentially identical.
Consistent with E19/E32/E41: surface/queue/congestion information adds nothing
beyond what v13 already captures. **No `likable-eagle_v15.parquet`.**
