# E19-B — Wrong-day BLOCK gate

**Project:** OpenAir  
**Experiment:** E19-B  
**Data:** 12 `training_*.parquet` files only. No ranking/submission.  
**Script:** `experiments/run_e19b_wrongday_block.py`

E19-A found that 97% of LFPG unmatched SSE is two easyJet rows whose BLOCK
is 16–23 h before schedule. This experiment asks whether a ranking-safe
gate can apply a 24 h wrap `y_hat = (MVT−SCHED) + 86400` only to those
bombs.

---

## Catalog (full-year training)

Rows with `y − (MVT−SCHED) > 12 h` (implied BLOCK more than 12 h early): **9**.

Unmatched (5):

| Airport | Flight | Prefix | y | MVT−SCHED | SCHED−BLOCK | Month |
|---|---|---|---:|---:|---:|---:|
| LSZH | NJE389D | NJE | 87,341 | 939 | 86,402 (~24 h) | May |
| LIRF | ITY332 | ITY | 87,168 | 11,694 | 75,474 | Jun |
| LFPG | EJU983W | EJU | 84,240 | 1,740 | 82,500 (~23 h) | **Jan** |
| LFPG | EJU42AY | EJU | 58,206 | 2,043 | 56,163 (~16 h) | **Jan** |
| LFPG | TAY1MN | TAY | −7 | −74,644 | 74,637 | Jul |

ITY332 is LIRF unmatched (already `MVT−SCHED`). TAY1MN is a negative taxi
with a nonsense schedule, not a long-taxi bomb. Real non-LIRF unmatched
wrong-day BLOCKs in 2025: **three rows** (two CDG easyJet in January, one
Zurich NetJets in May).

Matched “wrong-day” rows (4) have *normal* taxi and hugely negative
`MVT−SCHED` (schedule after takeoff). Different bug; ignore.

---

## Gate search (train = not Jan+Jul)

Precision = gated rows that are true bombs (`y − mvt_sched > 12 h` and `y > 8 h`).
A false positive gets a ~24 h prediction and becomes a new RMSE bomb.

The two CDG bombs sit in **January = primary val**. Jan+Jul *train* therefore
contains **zero** CDG bombs. Every `LFPG ∩ EJU ∩ mvt_sched < T` gate has
train precision **0** (the train hits are ordinary easyJet unmatched).

On val, `LFPG ∩ EJU ∩ mvt_sched < 2400` is 2/2 bombs, precision 1.0 — but
that is peeking at the holdout. December (no bombs): the same gate fires
on **2 false positives**.

`mvt_sched < 30 min` unmatched is thousands of normal rows (train n=4,043,
precision 0).

**No gate is selectable on train and safe on December.**

---

## Holdout (E18-H reproduced, then oracle wrap on true non-LIRF unmatched bombs)

| Split | E18-H overall | Oracle 24 h wrap | n bombs wrapped |
|---|---:|---:|---:|
| Jan+Jul | **372.36** | **334.39** (−38.0) | 2 |
| December | **238.01** | 238.01 (0) | 0 |

E18-H matched RMSE unchanged (250.98 / 223.95). The −38 s is exactly the
two January CDG rows. 24 h wrap is imperfect on EJU42AY (16 h early, not
24): wrap predicts 88,443 vs 58,206.

December has none of these events. A gate that fires there only adds error.

---

## Decision

**Verdict: REJECT as a deployable rule. KEEP as an oracle fact.**

- OLD: LFPG unmatched 21.8% of SSE might be a second unmatched DGP we can
  gate like LIRF.
- EVIDENCE: it is three wrong-day BLOCK rows in a year. Ranking-safe
  handles (easyJet, small MVT−SCHED, CDG) also match ordinary unmatched
  flights. Train of the ranking analogue does not contain the CDG bombs.
  December false-positives kill the apparent val-perfect gate. Oracle wrap
  372→334 on Jan+Jul, 0 on December.
- NEW: Do not ship a 24 h wrap. Do not treat LFPG unmatched as a
  population. If ranking 2026 happens to contain the same two-row failure,
  it is luck, not a model. Remaining structural lever is **LIRF unmatched
  regimes** (~30% of SSE), which need neighbour information E13 did not
  have.

Artifacts: `analysis/E19B/`.
