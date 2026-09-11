# E56 — multi-clock off-block decomposition — REJECT

## Hypothesis
For each off-block reference clock C: `T = (MVT−C) − G_C`, `G_C = BLOCK−C` learned. Does a clock other than SCHED/AOBT isolate a more learnable nuisance process (an E33-style breakthrough)?

## Availability audit (ranking DEP)
SCHED 100%; AOBT/LOBT/IOBT/EOBT 98.47% (identical null-set = unmatched rows). All clocks usable at prediction time.

## Clock tournament (reconstructed taxi, matched, cross-month OOF)
| Clock | Jan+Jul matched | top-1% SSE | Dec matched |
|---|---:|---:|---:|
| SCHED | 797.1 | 1.92e11 | 750.3 |
| EOBT | 286.31 | 1.037e10 | 231.2 |
| LOBT | 286.86 | 9.86e9 | 231.2 |
| IOBT | 287.07 | 9.88e9 | 231.2 |
| AOBT | 286.66 | 1.058e10 | 230.3 |
| **v13** | **238.5** | — | **212.1** |

EOBT/LOBT/IOBT/AOBT decompositions agree within ~0.5 s ⇒ the NM clocks are
mutually identical for matched rows (confirms E38). None beats v13; the AOBT
decomposition is already v13's `rec1` (grec) component. SCHED's decomposition is
not learnable standalone (gate delay dominates G_SCHED), which is why E33 needed
it only for the LIRF-unmatched slice with a learned G.

## Error orthogonality
Decompositions are near-identical to each other and v13 already blends the AOBT
form; no complementary clock signal exists.

## Decision: REJECT — close the clock-decomposition hypothesis family
No operational clock exposes a new learnable decomposition beyond SCHED/AOBT.
This is the expected negative result (E38's clock identity). **No v15.** Artifacts:
`experiments/run_e56_clock_tournament.py`, `experiments/results/E56/`.
