# E56 — multi-clock off-block decomposition

| Clock | ranking coverage | Jan+Jul matched | top-1% SSE |
|---|---:|---:|---:|
| SCHED | 1.0 | 797.13 | 1.917e+11 |
| EOBT | 0.9846595967416868 | 286.31 | 1.037e+10 |
| LOBT | 0.9846595967416868 | 286.86 | 9.858e+09 |
| IOBT | 0.9846595967416868 | 287.07 | 9.884e+09 |
| AOBT | 0.9846595967416868 | 286.66 | 1.058e+10 |
| v13_AOBT_alone | - | 286.66 | 1.058e+10 |

Note: E38 established AOBT/LOBT/IOBT/EOBT are mutually identical for matched rows (clk_range median 0) — so LOBT/IOBT/EOBT decompositions are expected to coincide with the AOBT decomposition (already in v13).
