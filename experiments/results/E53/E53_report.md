# E53 — previous ARR taxi-in in Δ

Ranking-safe: last ARR at the same stand, taxi-in and time since in-block (ARR TAXITIME is on ranking).

| spec | Jan+Jul matched | Dec matched | overall JJ |
|---|---:|---:|---:|
| v13 recon | 238.68 | 212.08 | 364.08 |
| **arr taxi-in + v13 recipe** | **236.86** | **210.54** | **362.91** |
| + LIRF grec λ=0.7 | 236.28 | 210.66 | 362.54 |

LIRF λ=0.7 helps JJ, hurts Dec. Ship **arr taxi-in without LIRF boost**.

LFPG wrong-day wrap cannot hit ~245 here: ranking LFPG EJU unmatched D<2400 are **hour 4 and 10** (morning FPs), not the Jan-2025 afternoon bombs. Tight wrap gate fires **0** ranking rows.

**v16:** `python experiments/submit.py --version v16 --hat-mode full --no-operator --arr-taxiin`

Expected if transfer is like v12/v13: LB ~282–283, not 245.
