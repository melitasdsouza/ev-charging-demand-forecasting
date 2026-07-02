import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/features.csv", parse_dates=["date"])

# Restrict to pre-pandemic data only for the main evaluation —
# demand collapsed in March 2020 and never recovered in this dataset,
# which would badly distort a normal train/test comparison
pre_pandemic = df[df["date"] < "2020-03-01"].reset_index(drop=True)
print("Pre-pandemic data:", pre_pandemic["date"].min(), "to", pre_pandemic["date"].max())
print("Rows:", len(pre_pandemic))

split_index = int(len(pre_pandemic) * 0.8)
train = pre_pandemic.iloc[:split_index]
test = pre_pandemic.iloc[split_index:]

print("Train:", train["date"].min(), "to", train["date"].max())
print("Test:", test["date"].min(), "to", test["date"].max())

baseline_predictions = test["lag_7d"]
actual = test["total_energy_kwh"]

mae = mean_absolute_error(actual, baseline_predictions)
rmse = mean_squared_error(actual, baseline_predictions) ** 0.5

print(f"Baseline MAE: {mae:.1f} kWh")
print(f"Baseline RMSE: {rmse:.1f} kWh")