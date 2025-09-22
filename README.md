# NFL Prediction Model (WIP)

**Status:** In progress. This repo contains an experimental pipeline that builds weekly team features from `nfl_data_py` play-by-play, rolls them forward as pre-game moving averages, and trains simple models (Linear Regression + LASSO) to predict **game point differential**. Results are evaluated out-of-sample by season.

## What it does (current)
- Pulls schedules and play-by-play for seasons **2020–2023** via `nfl_data_py`.
- Aggregates per-team/per-week **offense** and **defense** features (EPA splits, yards, rates, conversions, etc.).
- Creates **pre-game** features using expanding means shifted by 1 week (avoids leakage).
- Joins home/away rows to form a modeling table with Vegas `spread_line` and final `result`.
- Trains:
  - **Linear Regression** (train: < 2023, test: 2023).
  - **LASSO (LassoCV + TimeSeriesSplit)** to select a sparse feature set and inspect coefficients.
- Prints metrics (R² / RMSE), **directional accuracy vs spread**, VIF for selected features, and a per-year rolling RMSE table.
- Plots a bar chart of LASSO coefficients by magnitude.
