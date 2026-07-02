import pandas as pd
import numpy as np

daily = pd.read_csv("../data/daily_demand.csv", parse_dates=["date"])
print(daily.head())

daily["day_of_week"] = daily["date"].dt.dayofweek  # Monday=0 ... Sunday=6
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
daily["month"] = daily["date"].dt.month

print(daily[["date", "day_of_week", "is_weekend", "month"]].head(10))

us_holidays = pd.to_datetime([
    "2018-01-01", "2018-05-28", "2018-07-04", "2018-09-03", "2018-11-22", "2018-12-25",
    "2019-01-01", "2019-05-27", "2019-07-04", "2019-09-02", "2019-11-28", "2019-12-25",
    "2020-01-01", "2020-05-25", "2020-07-04", "2020-09-07", "2020-11-26", "2020-12-25",
])

daily["is_holiday"] = daily["date"].isin(us_holidays).astype(int)

print(daily[daily["is_holiday"] == 1][["date", "is_holiday"]])

daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

print(daily[["date", "day_of_week", "dow_sin", "dow_cos"]].head(8))

daily["lag_1d"] = daily["total_energy_kwh"].shift(1)
daily["lag_7d"] = daily["total_energy_kwh"].shift(7)

print(daily[["date", "total_energy_kwh", "lag_1d", "lag_7d"]].head(10))

daily["rolling_7d_avg"] = daily["total_energy_kwh"].shift(1).rolling(7).mean()

print(daily[["date", "total_energy_kwh", "lag_1d", "rolling_7d_avg"]].head(12))

daily["rolling_14d_avg"] = daily["total_energy_kwh"].shift(1).rolling(14).mean()

print("Before dropping NaNs:", daily.shape)
daily = daily.dropna().reset_index(drop=True)
print("After dropping NaNs:", daily.shape)
daily.to_csv("../data/features.csv", index=False)
print("Saved features.csv")