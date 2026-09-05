# OpenAir

OpenAir is an entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), organised by the EUROCONTROL Performance Review Commission. The goal is to predict the **taxi-out time** of flights at 10 major European airports and be ranked by Root Mean Square Error (RMSE).

## Project Idea

Build a machine learning model that predicts taxi-out time for every departure using the 2025 movements dataset. Features drawn from the data:

- **Airport & runway**: taxi distance depends on the stand, runway, and airport layout (EDDF, LFPG, EGLL, ...).
- **Aircraft & wake category**: heavies taxi and hold differently (A380, heavy/medium/light).
- **Market segment**: mainline, low-cost, regional, charter, all-cargo have different turnaround behaviour.
- **Time factors**: hour-of-day, day-of-week, and month to capture congestion peaks and seasonality.
- **Congestion**: rolling count of departures and arrivals in the minutes before pushback as a proxy for airport congestion.

## License

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for more details.

Copyright (C) 2026 Chandan Hegde
