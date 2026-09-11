# E46 — OpenSky / OPDI unmatched AOBT audit

Trino username/password are empty in pyopensky settings, so historical state vectors were **not** queried.
Used the PRC+OpenSky **OPDI v0.0.2** open parquet (same ADS-B network, no login).

## Flight-list join (full 2025 unmatched)

| Airport | unmatched | joined | coverage | med(MVT−first_seen) | RMSE(T, MVT−first_seen) | med\|BLOCK−first_seen\| |
|---|---:|---:|---:|---:|---:|---:|
| LIRF | 1488 | 1181 | 79.4% | -26 s | 8443 | 1760 s |
| LFPG | 3773 | 3515 | 93.2% | -42 s | 2204 | 1065 s |

Jan+Jul / Dec coverage (LIRF first_seen): 85.9% / 67.0%.

`first_seen` ≈ takeoff (`MVT`), not off-block. RMSE is worse than predicting a constant ~1100 s.

## Events sample (2025-01-05 to 2025-01-15)

| Airport | first_seen flights | exit-parking near origin | take-off near origin | entry-runway near origin |
|---|---:|---:|---:|---:|
| LIRF | 3723 | 0 (0.0% of flights) | 1 | 797 |
| LFPG | 5761 | 14 (0.2% of flights) | 54 | 2586 |

exit-parking_position on LIRF departures geolocates at DESTINATION (arrival stand), not FCO. entry-runway at FCO is simultaneous with first_seen (lift-off).

## GO / NO-GO

- GO flag: **False**
- AOBT* is airborne/MVT: True
- No FCO origin parking events: True

OPDI/OSN first_seen is lift-off (~MVT), not off-block. FCO origin exit-parking events are absent in the events sample. Unmatched flights *are* in ADS-B, but the missing clock is not.

**Verdict: NO-GO.** Do not splice OpenSky/OPDI into v9. Surface ADS-B at LIRF does not observe off-block.

Artifacts: `experiments/run_e46_opdi_aobt.py`, `experiments/download_opdi.py`, `data/external/opdi/`, `experiments/results/E46/`.
