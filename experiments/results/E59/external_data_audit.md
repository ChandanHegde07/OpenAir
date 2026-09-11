# E59 — External trajectory (OpenSky) data audit

## Outcome: E59 cannot be implemented as specified — documented blockers.

### 1. Dataset / provider / license
- Dataset: OpenSky Network historical state-vector / trajectory data (adsb-exchange-derived, ICAO ANSP feeds).
- Provider: OpenSky Network (opensky-network.org). License: **ODbL** for raw states (attribution + share-alike); derived/aggregated products **CC-BY-SA 4.0** (per OpenSky usage/licensing terms). GPLv3 code + documented attribution satisfies the competition's external-data rules.
- Reachability: anonymous states API returns HTTP 200 (live regional states). Historical **Tracks/Impala** access requires authenticated API + per-flight time-range queries with strict rate limits.

### 2. Coverage / resolution
- Historical ground-phase (on-ground, alt≈0, speed<50 kt) trajectories exist for the 10 airports' 2025 + Jan/Jul 2026 airspace.
- Resolution: ~5–10 s state-vector intervals for most transponders; gaps in ground coverage (antenna shadowing at some ramps).

### 3. CRITICAL — linking identifiers
- The challenge's 30 columns contain **no ICAO24 / registration / hex / tail number** (audited: zero such columns).
- The only candidate key is the **callsign (`FLIGHT_mvt`) + airport + timestamp + aircraft type**.
- Known problem: `FLIGHT_mvt` is a **city-pair flight identity that repeats daily on different physical aircraft** (E7), so callsign-only linkage has material false-match risk and cannot unambiguously attach an ADS-B track to a specific challenge movement. No registration field exists to disambiguate.

### 4. Feasibility of acquisition
- Full 18-month × 10-airport coverage = ~400k departures; historical per-flight track queries under anonymous/authenticated rate limits are **not feasible in-session**, and a small subset (few airports × few days) cannot produce a transferable free-flow reference for the ranking set.

### 5. Anti-leakage constraint (E59 §20)
- ADS-B ground tracks make actual BLOCK/TAXITIME directly observable; using complete tracks for ranking predictions is forbidden by the project policy and the mission's own §20. Even legitimate prediction-cutoff feature extraction would require resolving the challenge's intended prediction moment.

### 6. Conclusion
The hypothesis (recover physical gate-to-runway geometry) is valid, but:
1. No aircraft-identity link key exists in the challenge data (no ICAO24/registration) → trajectory-to-flight association is unreliable.
2. Historical trajectory volume needed for a transferable free-flow reference is not obtainable within the session.
3. Licensing is compliant (ODbL/CC-BY-SA + attribution), so the blocker is **data linkage and acquisition**, not licensing.

**Decision: E59 not implemented beyond this audit.** No trajectory matching, route graph, or free-flow reference produced; no model trained; no submission. Documented per §23 ("if no, document why").
