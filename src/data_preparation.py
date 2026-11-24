"""Load Delhi data and build the inputs shared by training and the app."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITY = "Delhi"
TEST_START = "2020-01-01"
RANDOM_SEED = 42
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "O3"]
LAGS = [1, 2, 3, 7]
AQI_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def load_city_data(csv_path=None):
    """Return one row per calendar day, with dates in ascending order."""
    if csv_path is None:
        csv_path = ROOT / "data" / "raw" / "city_day.csv"
    data = pd.read_csv(csv_path)
    data = data[data["City"] == CITY].copy()
    if data.empty:
        raise ValueError(f"No records found for {CITY}.")
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data = data.sort_values("Date").set_index("Date")
    if data.index.has_duplicates:
        raise ValueError("Duplicate city dates need checking before training.")
    dates = pd.date_range(data.index.min(), data.index.max(), freq="D")
    data = data.reindex(dates)
    data.index.name = "Date"
    return data


def build_features(daily):
    """At origin day t, predict day t+1 using readings available by t."""
    # Carry readings forward for at most two days. Future readings are never used.
    history = daily[POLLUTANTS].ffill(limit=2)
    features = pd.DataFrame(index=daily.index)
    features["PM2.5_today"] = history["PM2.5"]

    for pollutant in POLLUTANTS:
        for lag in LAGS:
            features[f"{pollutant}_lag_{lag}"] = history[pollutant].shift(lag)

    # Shift first: these windows end at t-1, as required by the project brief.
    past_pm25 = history["PM2.5"].shift(1)
    for window in [3, 7]:
        features[f"PM2.5_mean_{window}"] = past_pm25.rolling(window).mean()
        features[f"PM2.5_std_{window}"] = past_pm25.rolling(window).std()

    # The calendar of tomorrow is already known today.
    tomorrow = features.index + pd.Timedelta(days=1)
    features["month"] = tomorrow.month
    features["day_of_week"] = tomorrow.dayofweek
    year_length = np.where(tomorrow.is_leap_year, 366, 365)
    angle = 2 * np.pi * (tomorrow.dayofyear - 1) / year_length
    features["day_of_year_sin"] = np.sin(angle)
    features["day_of_year_cos"] = np.cos(angle)
    # Exclude the first seven origins, which lack seven days of history.
    return features.iloc[max(LAGS):]


def build_targets(daily):
    """Keep observed next-day answers. Missing targets are never filled."""
    targets = pd.DataFrame(index=daily.index)
    targets["target_date"] = daily.index + pd.Timedelta(days=1)
    targets["PM2.5_tomorrow"] = daily["PM2.5"].shift(-1)
    targets["AQI_tomorrow"] = daily["AQI_Bucket"].shift(-1)
    targets["PM2.5_today_observed"] = daily["PM2.5"]
    return targets
