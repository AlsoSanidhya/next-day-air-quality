# Next-Day AQI Prediction for Delhi

This AML project uses daily historical readings to predict Delhi's numerical AQI one day ahead. It compares a Decision Tree Regressor and a Random Forest Regressor. PM2.5 is one of the inputs, and numerical AQI is the sole prediction target.

## Run in the VS Code terminal

```powershell
.\.venv\Scripts\python.exe main.py
```

For a fresh setup, create a virtual environment with `python -m venv .venv` and install dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## How it works

1. Read `data/raw/city_day.csv`, select Delhi and arrange its readings by calendar date.
2. Use today's AQI and five pollutant readings to predict tomorrow's observed AQI.
3. Train on target dates before 2020 and test on the 2020 dates in the dataset. Fill missing inputs with medians learned during fitting.
4. Select the model with the lowest mean absolute error in chronological cross-validation on the training period. Print both models' test scores and a prediction for the day after the last reading in the dataset.

`main.py` is the only project script. Each run trains the models and writes a scores table, test predictions and a graph to `results/`. The demonstration uses historical readings through July 2020.

Dataset: [Air Quality Data in India by Rohan Rao](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
