import pandas as pd

df = pd.read_csv("../data/EVcharging.csv")
df["Start Date"] = pd.to_datetime(df["Start Date"], format = "%d-%m-%Y %H:%M")

#strip time from the Start Date column
df["date"] = df["Start Date"].dt.date
print(df[["Start Date","date"]].head())

#group by date and aggregate
daily = df.groupby("date").agg(
    total_energy_kwh=("Energy (kWh)", "sum"),
    session_count=("Energy (kWh)", "count"),
    unique_stations_used=("Station Name", "nunique"),
).reset_index()

print(daily.head())
print(daily.shape)

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = daily.set_index("date").reindex(full_range).fillna(0.0)
daily = daily.rename_axis("date").reset_index()

print(daily.shape)
print(daily.isnull().sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(daily["date"], daily["total_energy_kwh"])
plt.title("Daily Total EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.savefig("../outputs/daily_demand_raw.png")
plt.show()

daily.to_csv("../data/daily_demand.csv", index=False)
print("Saved daily_demand.csv")