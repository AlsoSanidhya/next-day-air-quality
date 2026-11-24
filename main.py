"""Predict Delhi's next-day numerical AQI with two regression models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
CITY = "Delhi"
TEST_START = "2020-01-01"
FEATURES = ["PM2.5", "PM10", "NO2", "CO", "O3", "AQI"]


def main():
    # One row per calendar day ensures shift(-1) means tomorrow.
    data = pd.read_csv(ROOT / "data" / "raw" / "city_day.csv")
    data = data[data["City"] == CITY].copy()
    if data.empty:
        raise ValueError(f"No data found for {CITY}.")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")
    if data.index.has_duplicates:
        raise ValueError("Duplicate dates must be resolved first.")
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="D"))

    # X contains today's readings; y contains the observed AQI tomorrow.
    data["Tomorrow_AQI"] = data["AQI"].shift(-1)
    data["Target_date"] = data.index + pd.Timedelta(days=1)
    samples = data.dropna(subset=["Tomorrow_AQI"]).copy()
    train = samples[samples["Target_date"] < TEST_START]
    test = samples[samples["Target_date"] >= TEST_START]
    if train.empty or test.empty:
        raise ValueError("The date cutoff must leave both training and test data.")
    X_train, y_train = train[FEATURES], train["Tomorrow_AQI"]
    X_test, y_test = test[FEATURES], test["Tomorrow_AQI"]

    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        ),
    }
    results = []
    predictions = pd.DataFrame({
        "Target_date": test["Target_date"].to_numpy(),
        "Actual_AQI": y_test.to_numpy(),
    })
    best_model, best_name, best_cv_mae = None, "", float("inf")
    # Select a model on earlier data, keeping the final test period separate.
    cv = TimeSeriesSplit(n_splits=3, gap=1)
    for name, estimator in models.items():
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        cv_mae = -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
        ).mean()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        results.append({
            "Model": name, "CV_MAE": cv_mae,
            "Test_MAE": mean_absolute_error(y_test, predicted),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
            "Test_R2": r2_score(y_test, predicted),
        })
        if cv_mae < best_cv_mae:
            best_model, best_name, best_cv_mae = model, name, cv_mae

    results = pd.DataFrame(results)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    results.to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    predictions.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    print(f"\nNext-Day AQI Prediction: {CITY}")
    print("Target: numerical AQI (regression)")
    print(f"Inputs: {', '.join(FEATURES)}")
    print(f"Training samples: {len(train)} | Test samples: {len(test)}")
    print(f"Training target dates: before {TEST_START}")
    print(f"Test target dates: {test['Target_date'].min().date()} to "
          f"{test['Target_date'].max().date()}")
    print("\nModel comparison (lower MAE/RMSE is better; higher R2 is better):")
    print(results.round(3).to_string(index=False))
    print(f"\nSelected using training-period cross-validation: {best_name}")

    last_day = data.index.max()
    forecast = best_model.predict(data.loc[[last_day], FEATURES])[0]
    print(f"Readings from: {last_day.date()}")
    print(f"Predicted AQI for {(last_day + pd.Timedelta(days=1)).date()}: {forecast:.2f}")
    print("Historical demonstration; this is not a live forecast for today's date.")

    plt.figure(figsize=(10, 4))
    plt.plot(predictions["Target_date"], predictions["Actual_AQI"], label="Actual AQI")
    plt.plot(predictions["Target_date"], predictions[best_name], label=f"Predicted ({best_name})")
    plt.xlabel("Target date")
    plt.ylabel("AQI")
    plt.title(f"{CITY}: next-day AQI on the test period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "aqi_predictions.png", dpi=150)
    plt.close()
    print("\nSaved metrics.csv, predictions.csv and aqi_predictions.png in results/.")


if __name__ == "__main__":
    main()
