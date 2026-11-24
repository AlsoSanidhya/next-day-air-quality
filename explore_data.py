import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV into a table called a DataFrame.
df = pd.read_csv("data/raw/city_day.csv")
print("Full dataset shape:", df.shape)

# Keep the rows that belong to Delhi.
city_data = df[df["City"] == "Delhi"].copy()

print("\nFirst five Delhi records:")
print(city_data.head())

print("\nDelhi dataset shape:", city_data.shape)

print("\nMissing values in each Delhi column:")
print(city_data.isnull().sum())

print("\nMissing percentage in each Delhi column:")
print(city_data.isnull().sum() / len(city_data) * 100)

# Convert dates and arrange the records from oldest to newest.
city_data["Date"] = pd.to_datetime(city_data["Date"], format="%Y-%m-%d")
city_data = city_data.sort_values("Date")

print("\nFirst date:", city_data["Date"].min())
print("Last date:", city_data["Date"].max())
print("Duplicate dates:", city_data["Date"].duplicated().sum())

# Compare the recorded dates with a complete daily calendar.
all_dates = pd.date_range(city_data["Date"].min(), city_data["Date"].max(), freq="D")
missing_dates = all_dates.difference(city_data["Date"])
print("Missing days:", len(missing_dates))
print("Missing dates:", missing_dates.tolist())

plt.figure(figsize=(10, 4))
plt.plot(city_data["Date"], city_data["PM2.5"])

plt.title("Delhi PM2.5 Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")

plt.tight_layout()
plt.show()