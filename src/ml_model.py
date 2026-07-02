import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/features.csv", parse_dates=["date"])

pre_pandemic = df[df["date"] < "2020-03-01"].reset_index(drop=True)

split_index = int(len(pre_pandemic) * 0.8)
train = pre_pandemic.iloc[:split_index]
test = pre_pandemic.iloc[split_index:]

print("Train:", train["date"].min(), "to", train["date"].max())
print("Test:", test["date"].min(), "to", test["date"].max())

feature_columns = [
    "day_of_week", "is_weekend", "month", "is_holiday",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1d", "lag_7d", "rolling_7d_avg", "rolling_14d_avg",
]

X_train = train[feature_columns]
y_train = train["total_energy_kwh"]

X_test = test[feature_columns]
y_test = test["total_energy_kwh"]

print(X_train.shape, X_test.shape)

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
)

model.fit(X_train, y_train)
print("Model trained")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print(f"ML Model MAE: {mae:.1f} kWh")
print(f"ML Model RMSE: {rmse:.1f} kWh")

import matplotlib.pyplot as plt

plt.figure(figsize=(13, 5))
plt.plot(test["date"], y_test.values, label="Actual", color="black")
plt.plot(test["date"], predictions, label="Predicted", color="red")
plt.legend()
plt.title("Actual vs Predicted EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.savefig("../outputs/actual_vs_predicted.png")
plt.show()

importances = pd.Series(model.feature_importances_, index=feature_columns)
importances = importances.sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../outputs/feature_importance.png")
plt.show()

pandemic_period = df[df["date"] >= "2020-03-01"].reset_index(drop=True)
print("Pandemic period:", pandemic_period["date"].min(), "to", pandemic_period["date"].max())
print("Rows:", len(pandemic_period))

X_pandemic = pandemic_period[feature_columns]
y_pandemic = pandemic_period["total_energy_kwh"]

pandemic_predictions = model.predict(X_pandemic)

pandemic_mae = mean_absolute_error(y_pandemic, pandemic_predictions)
pandemic_rmse = mean_squared_error(y_pandemic, pandemic_predictions) ** 0.5

print(f"Pandemic period MAE: {pandemic_mae:.1f} kWh")
print(f"Pandemic period RMSE: {pandemic_rmse:.1f} kWh")

plt.figure(figsize=(13, 5))
plt.plot(pandemic_period["date"], y_pandemic.values, label="Actual", color="black")
plt.plot(pandemic_period["date"], pandemic_predictions, label="Predicted (pre-pandemic model)", color="red")
plt.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", label="Lockdowns begin")
plt.legend()
plt.title("Stress Test: Pre-Pandemic Model on Pandemic Data")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.savefig("../outputs/pandemic_stress_test.png")