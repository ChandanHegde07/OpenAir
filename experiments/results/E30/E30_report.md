# E30 — 246 RMSE attack: distribution/schema audit report

E20 untouched. No candidate improved total SSE. This report answers the
fundamental question: **why can the leaderboard be ~246 while our system is
~317?** — with the evidence gathered.

## Schema / availability (C32)
- Training and ranking have **identical 30 columns**; no hidden/unused field.
- Ranking DEP 344,841: NM fields present for **98.47%** (AOBT/LOBT/IOBT/EOBT),
  SCHED 100%. Unmatched = 5,290 (1.53%) and they null **every** flt field
  including `FLIGHT_ID_mvt`; movement fields (FLIGHT/ADEP/ADES/SCHED/MVT/runway/
  stand/type) remain.
- `ARVT_3_flt` is present but occurs **after MVT for 100%** of rows → forbidden
  future information; not used.

## Population shift (C31)
MVT−SCHED (the dominant driver) distributions:

| slice | n | median | p90 | unmatched |
|---|---:|---:|---:|---:|
| train 2025 | 2,085,047 | 1402 | 3484 | 1.08% |
| Dec 2025 | 165,677 | 1385 | 3486 | 1.00% |
| Jan 2025 | 153,706 | 1277 | 3425 | 0.93% |
| Jul 2025 | 190,713 | 1715 | 4378 | 2.06% |
| **ranking 2026** | 344,841 | 1501 | 4035 | 1.53% |

PSI(ms, ranking vs train) = **0.0145** (negligible). Categorical unseen rates:
runway 0.0%, stand 0.1%, aircraft type 0.0%.

LIRF-unmatched MVT−SCHED is identical: ranking median **6955** vs 2025 **6781**
(p90 15902 vs 14939). **The ranking population is not materially shifted from
the 2025 holdout.**

## Leaderboard arithmetic
- Internal Jan+Jul 368.03 ↔ LB 316.9654 ⇒ offset ≈ 51.1 s.
- LB 246 target SSE (N=344,841): 2.087e10. Our LB-implied SSE (317): 3.464e10.
  The gap is essentially the unmatched SSE (LIRF gate-delay rows).
- Internal 2025: matched SSE 2.031e10 (244.76 RMSE) is already ~ the 246 target
  level. **The entire 246 gap is unmatched/LIRF.**

## Conclusion
Because (a) the ranking population matches the 2025 holdout (PSI 0.015), (b)
all 30 columns are shared and audited, and (c) LIRF-unmatched gate delay was
proven unobservable from movement-only fields in E29 (C28), **no legitimate,
observable feature in the provided data explains a ~246 leaderboard result.**
Reaching 246 would require either (i) information not present in the dataset,
(ii) the forbidden post-takeoff `ARVT_3_flt`/future fields, or (iii) a ranking
ground-truth definition that differs from training `MVT−BLOCK`.

No submission. E20 remains production (LB 316.9654). Branches closed with exact
reasons: service recovery (C24), matched-tail correction (C25), movement-only
unmatched incl. LIRF (C28–C30), MVT−SCHED clipping/blend (C30).
