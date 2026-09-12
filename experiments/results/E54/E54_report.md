# E54 — richer inbound stand state

v16 baseline 236.86 / 210.54.

| spec | Jan+Jul | Dec |
|---|---:|---:|
| v16 | 236.86 | 210.54 |
| + delay/since_dep | 237.07 | 210.39 |
| + AOBT-asof taxi-in | 237.00 | 210.37 |
| **all extra inbound** | **236.80** | **210.29** |
| all + LIRF λ=0.6 | 236.39 | 210.35 |

LIRF λ=0.6 helps JJ, hurts Dec. Ship **all extra inbound**, λ=0.5.

**v17:** `submit.py --version v17 --hat-mode full --no-operator --arr-taxiin --arr-rich`

Jan+Jul OOF edge is tiny (−0.06). Keep v16 if LB ≥ 282.09.
