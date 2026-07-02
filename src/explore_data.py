import pandas as pd

df = pd.read_csv("../data/EVcharging.csv")

df["Start Date"] = pd.to_datetime(df["Start Date"], format = "%d-%m-%Y %H:%M")

print(df["Start Date"].head())
print(df["Start Date"].dtype)

print(df.isnull().sum())

print("Earliest date:", df["Start Date"].min())
print("Latest date:", df["Start Date"].max())

print("Number of unique stations:", df["Station Name"].nunique())
print(df["Station Name"].value_counts().head(10))

print(df["Energy (kWh)"].describe())