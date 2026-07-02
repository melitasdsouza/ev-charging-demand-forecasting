# EV Charging Demand Forecasting

A time-series ML pipeline that forecasts daily EV charging demand from
real public charging session data (City of Palo Alto, 2018–2020),
benchmarked against a seasonal-naive baseline.

## The data

[Dataset link here] — 102,781 real charging sessions across 45 stations
in Palo Alto, CA. Note: despite the dataset's title mentioning weather
and traffic, the actual file only contains session-level charging data
(timestamps, energy, duration, station info) — no weather or traffic
columns. This project works with what's actually in the data.

## Pipeline

1. Aggregated 100k+ individual sessions into a daily citywide demand
   time series
2. Engineered calendar features (day of week, holidays, cyclical
   encodings) and lag/rolling features (yesterday's demand, last week,
   rolling averages)
3. Built a seasonal-naive baseline (predict today = same weekday last
   week)
4. Trained a Gradient Boosting model, evaluated with a time-based
   train/test split

## Results (pre-pandemic test period)

| Model | MAE (kWh) | RMSE (kWh) |
|---|---|---|
| Seasonal-naive baseline | 203.4 | 296.5 |
| Gradient Boosting | 155.5 | 192.6 |

The ML model reduces error by about 24% over the baseline by picking up
on finer patterns like holiday dips that a pure "last week" rule misses.

[Insert your actual_vs_predicted.png here]

## The pandemic stress test

[Write 2-3 sentences here: what did you do, and what happened to the
numbers? Use your 155.5 -> 461.2 MAE comparison.]

[Write your one-sentence explanation of WHY it fails here]

[Insert your pandemic_stress_test.png here]

## What I'd do differently in production

[Write a sentence or two about drift detection/alerting here]

## Project structure

[Describe your folders: data/, src/, outputs/]

## Running it

[Write the pip install + how to run your scripts, in order]
