# E52 — beat v13 on both splits

v13 published OOF 238.52 / 212.08. Ranking is Jan+Jul.

| spec | cand | Jan+Jul | Dec |
|---|---|---:|---:|
| base | v13 recon | 238.68 | 212.04 |
| base | **hpos** (hat≥0 only) | 238.55 | 211.95 |
| geo | v13 | 239.69 | 212.12 |
| op | v13 | 238.58 | 212.00 |
| **op** | **hpos** | **238.48** | **211.92** |

Geo_mean and LIRF extra residual do not help. Operator + **positive-only leftover** is the only both-split improvement (small).

**v15** via `python experiments/submit.py --version v15 --hat-mode pos --operator`

If LB does not beat **284.10**, keep v13.
