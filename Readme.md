# OpenAir

OpenAir is an entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), organised by the EUROCONTROL Performance Review Commission. The goal is to predict the **taxi-out time** of flights at 10 major European airports and be ranked by Root Mean Square Error (RMSE).

## Project Idea

Build a machine learning model that predicts taxi-out time for every departure using the 2025 movements dataset. Features drawn from the data:

- **Airport & runway**: taxi distance depends on the stand, runway, and airport layout (EDDF, LFPG, EGLL, ...).
- **Aircraft & wake category**: heavies taxi and hold differently (A380, heavy/medium/light).
- **Market segment**: mainline, low-cost, regional, charter, all-cargo have different turnaround behaviour.
- **Time factors**: hour-of-day, day-of-week, and month to capture congestion peaks and seasonality.
- **Congestion**: rolling count of departures and arrivals in the minutes before pushback as a proxy for airport congestion.

## Status

Research-phase results are tracked in [`status.md`](status.md), the permanent research journal. Highlights (January + July 2025 holdout):

| Model | Overall RMSE | Matched RMSE | MAE |
|---|---:|---:|---:|
| Airport mean | 660 | 440 | — |
| E3 linear (clocks + geometry) | 577 | 304 | 191 |
| E11 LIRF unmatched override added | 410 | 304 | 190 |
| E9 residual LightGBM + override (current best) | **378** | **256** | 165 |

Key findings: `MVT − AOBT` and `AOBT − EOBT` are the two useful clocks; historical stand×runway `geo_mean` adds ~14 s; linear traffic/queue features are redundant; LIRF flights with no NM match are a separate generating process scored with `MVT − SCHED`.

## Repository layout

```
analysis/            # Discovery scripts (01..05) + DISCOVERY_REPORT.md
  output/            # Analysis run outputs
experiments/         # E0-E13 experiment runners + shared common.py
  results/           # Saved results (JSON/txt) per experiment
air-data/            # Raw parquet files (gitignored)
status.md            # Research journal: conclusions, decisions, experiment log
```

Data policy is binding: **train, fit, calibrate, encode, and select using `training_*.parquet` only**. `ranking.parquet` / `submitting.parquet` must never enter fits or feature statistics, and `experiments/common.py` refuses to load them.

## Validation protocol

- Primary split: train all 2025 months except January and July; validate January + July 2025.
- Sanity split: train Jan–Nov, validate December.
- Rolling features use strictly previous flights (shift 1); geometry/calibration tables are fit on the training split only.
- Always report both overall RMSE (dominated by ~1% unmatched rows) and matched RMSE.

## License

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for more details.

Copyright (C) 2026 Chandan Hegde
