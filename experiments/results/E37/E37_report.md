# E37 — ground-state representation: report

## Baseline
v9 = `likable-eagle_v9.parquet`, **LB 288.9003**.

## Geometry feasibility (blocker)
The dataset carries only heterogeneous stand IDs (`811`, `R07`, `K7L`, `B10R`,
`A52`), with no openly licensed, reproducible stand-ID → coordinate mapping.
Building the requested airport graph (stand→taxiway→runway shortest paths,
turns, crossings, edge pressures) would require fabricating coordinates, which
E37 explicitly forbids. The full geometry engine was therefore not built.

## Coordinate-free subset tested
Two-stage model: `T_phys` = shrunk median y by (airport × runway × stand-zone)
→ (airport × runway) → airport, plus `T_excess` = LightGBM on runway
configuration state (dominant-runway share, entropy, target-is-dominant,
switch age), route (runway×zone) pressure, and surface counts. Matched rows,
stand zones derived from the stand-ID prefix.

## Results (matched RMSE / top-5% SSE)
| split | E20/v9 | ground-state | best blend |
|---|---:|---:|---:|
| Jan+Jul | 244.8 / 1.264e10 | 353.5 / 3.018e10 | g=0.0 → E20 |
| Dec | 214.8 / 4.240e9 | 282.6 / 7.682e9 | g=0.0 → E20 |

The optimal gate weight is **0.0** on both splits: the representation never
beats E20. **Top-5% matched SSE is unchanged** (the regime that holds ~32% of
matched SSE).

## Decision
**No `likable-eagle_v10.parquet`.** Per E37 §15, the new representation does not
reduce top-5% SSE, so polishing it is stopped. The real geometry engine is not
reproducible from the provided stand IDs; alternatively E20's existing
train-split stand×runway `geo_mean` already captures the route prior, leaving no
incremental signal in the coordinate-free configuration/route-pressure features.

Artifacts: `experiments/run_e37_ground_state.py`,
`experiments/results/E37/E37_ground_state.json`.
