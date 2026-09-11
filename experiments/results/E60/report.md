# E60 — Physical taxi-route decomposition — REJECT (component diagnosis)

## Phase 0-3 status
- v13 baseline reproduced contextually (matched ~238.5).
- **Runway geometry acquired** (OurAirports, public domain): all 10 airports, runway length/heading/count.
- **Stand mapping (Phase 2) FAILED**: challenge STAND IDs (`811`,`R07`,`K7L`,`B10R`) have no reproducible open stand→coordinate crosswalk (same blocker as E37/E59). No fabrication allowed.
- Therefore **taxiway graph (Phase 4), stand→runway distance, free-flow route time (Phase 5-6) are NOT constructible** from reproducible open data.

## Phase 3 partial test — runway-geometry features in the Δ harness
| split | geometry-featured Δ matched | v13 grec (~238) |
|---|---:|---:|
| Jan+Jul | 248.81 | 238.5 |
| Dec | 215.39 | 212.1 |

Runway length/heading/count add only a marginal standalone gain; the route distance — the hypothesized core signal — is unavailable.

## Failure diagnosis (per Phase 15)
Failed component: **stand mapping** → graph topology → route distance → free-flow time. Runway mapping succeeded but runway length is not taxi-route distance. The physical decomposition `T = F + delay` cannot be tested without stand coordinates.

## Decision
**No v15.** No robust improvement over v13 demonstrated; the physical-route hypothesis remains untestable with reproducible open data until a legal stand-ID→coordinate crosswalk exists. Artifacts: `experiments/run_e60_route_decomposition.py`, `experiments/results/E60/` (metrics.json, external_sources.md, report.md).
