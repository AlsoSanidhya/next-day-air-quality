"""A small Tkinter window for demonstrating historical next-day predictions."""
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

from src.data_preparation import ROOT, TEST_START, build_features, load_city_data


def predict_day(bundle, features, origin_date, today_pm25=None):
    """Used by the window and by the prediction smoke check."""
    origin = pd.Timestamp(origin_date)
    row = features.loc[[origin]].copy()
    if today_pm25 is not None:
        if not 0 <= today_pm25 <= 2000:
            raise ValueError("Enter a PM2.5 value between 0 and 2000 ug/m3.")
        row["PM2.5_today"] = today_pm25
    concentration = float(bundle["regression"].predict(row)[0])
    category = str(bundle["classification"].predict(row)[0])
    return origin + pd.Timedelta(days=1), concentration, category


class AirQualityApp:
    def __init__(self, window):
        self.window = window
        window.title("Delhi Next-Day Air Quality")
        window.geometry("840x760")
        window.minsize(780, 730)
        self.daily = load_city_data()
        self.features = build_features(self.daily)
        self.bundle = joblib.load(ROOT / "models" / "forecast_models.joblib")
        summary = self.bundle["summary"]
        dates = self.features.index[
            self.features.index + pd.Timedelta(days=1) >= pd.Timestamp(TEST_START)
        ]
        self.dates = [str(date.date()) for date in dates]

        frame = ttk.Frame(window, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Delhi Next-Day Air Quality", font=("Segoe UI", 21, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Historical demonstration using 2015-2020 data", font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 14))
        ttk.Label(frame, text="Choose a day. The model predicts the following calendar day.",
                  wraplength=740).pack(anchor="w")
        ttk.Label(frame, text="Models trained on target dates before 1 January 2020.",
                  wraplength=740).pack(anchor="w", pady=(3, 12))

        inputs = ttk.LabelFrame(frame, text="Prediction inputs", padding=14)
        inputs.pack(fill="x")
        ttk.Label(inputs, text="Date of available readings").grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.date = tk.StringVar(value=self.dates[0])
        picker = ttk.Combobox(inputs, textvariable=self.date, values=self.dates,
                              state="readonly", width=22)
        picker.grid(row=0, column=1, sticky="w")
        picker.bind("<<ComboboxSelected>>", self.update_reading)
        ttk.Label(inputs, text="PM2.5 on that day (ug/m3)").grid(row=1, column=0, sticky="w", pady=12)
        self.today = tk.StringVar()
        ttk.Entry(inputs, textvariable=self.today, width=25).grid(row=1, column=1, sticky="w")
        ttk.Label(inputs, text="You can edit PM2.5 for a what-if exercise. Other inputs come from the selected day's history.",
                  wraplength=690).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(frame, text="Predict next day", command=self.show_prediction).pack(anchor="w", pady=16)

        output = ttk.LabelFrame(frame, text="Forecast", padding=16)
        output.pack(fill="x")
        self.result = tk.StringVar(value="Choose inputs and click Predict next day.")
        ttk.Label(output, textvariable=self.result, font=("Segoe UI", 15),
                  wraplength=710, justify="left").pack(anchor="w")
        self.observed = tk.StringVar()
        ttk.Label(output, textvariable=self.observed, wraplength=710).pack(anchor="w", pady=(12, 0))

        ttk.Label(frame, text=f"Regression model: {summary['regression_model']}\n"
                  f"AQI classifier: {summary['classification_model']}\n"
                  "PM2.5 and AQI category are predicted by separate models.",
                  wraplength=740).pack(anchor="w", pady=14)
        ttk.Button(frame, text="View model comparison", command=self.show_metrics).pack(anchor="w")
        ttk.Label(frame, text="The final available date can demonstrate a forecast with no next-day observation to compare.",
                  wraplength=740).pack(anchor="w", pady=12)
        self.update_reading()

    def update_reading(self, event=None):
        value = self.daily.loc[pd.Timestamp(self.date.get()), "PM2.5"]
        self.today.set("" if pd.isna(value) else f"{value:.2f}")
        self.result.set("Choose inputs and click Predict next day.")
        self.observed.set("")

    def show_prediction(self):
        try:
            value = float(self.today.get())
            target_date, concentration, category = predict_day(
                self.bundle, self.features, self.date.get(), value
            )
            self.result.set(f"Prediction for {target_date.date()}\n"
                            f"PM2.5: {concentration:.2f} ug/m3\nAQI category: {category}")
            if target_date in self.daily.index:
                actual = self.daily.loc[target_date, "PM2.5"]
                actual_category = self.daily.loc[target_date, "AQI_Bucket"]
                pm_text = "missing" if pd.isna(actual) else f"{actual:.2f} ug/m3"
                category_text = "missing" if pd.isna(actual_category) else str(actual_category)
                self.observed.set(f"Recorded next day: PM2.5 {pm_text}; AQI category {category_text}.")
            else:
                self.observed.set("The dataset has no observed reading for this target date.")
        except (ValueError, KeyError) as error:
            messagebox.showerror("Check prediction inputs", str(error), parent=self.window)

    def show_metrics(self):
        metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
        text = "2020 held-out results\n\n"
        for _, row in metrics.iterrows():
            if row["task"] == "regression":
                text += f"{row['model']}: MAE {row['mae']:.2f}, RMSE {row['rmse']:.2f}\n"
            else:
                text += f"{row['model']}: macro-F1 {row['macro_f1']:.3f}\n"
        messagebox.showinfo("Model comparison", text, parent=self.window)


if __name__ == "__main__":
    root = tk.Tk()
    if not (ROOT / "models" / "forecast_models.joblib").exists():
        root.withdraw()
        messagebox.showerror("Train the models first", "Run python train_models.py, then reopen app.py.")
        root.destroy()
    else:
        AirQualityApp(root)
        root.mainloop()
