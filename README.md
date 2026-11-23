# Next-day Air Quality

An individual Applied Machine Learning project by Sanidhya Negi.

Goal: use historical pollution readings to predict next-day PM2.5 concentration
and AQI category for one Indian city. The city will be selected after inspecting
data availability. This is an educational project using historical data, not a
live forecasting service.

## Current progress

- Project repository created.
- Python environment setup in progress.
- Dataset exploration and model training have not started.

## Step 1: Check the Python setup

Open this folder in VS Code. Select `.venv/Scripts/python.exe` using
**Python: Select Interpreter**, then run `check_setup.py`.

From the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe check_setup.py
```

The program should print the Python and library versions, followed by a success
message. This step checks the tools; it does not train a model.

## Data

Source: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Download the dataset in your browser, extract the ZIP, and put `city_day.csv`
inside `data/raw/`. Preserve the downloaded CSV without modifying it.

## Planned stages

1. Inspect data and missing values.
2. Prepare dates and historical features.
3. Build a persistence baseline and simple regression.
4. Compare the remaining required models.
5. Evaluate and document results.

## References and assistance

Project brief: https://github.com/tali7c/Applied-Machine-Learning/tree/main/projects/P02-air-quality

Teacher-provided lab examples guide the coding style. ChatGPT assists with
planning, setup, explanations and code review. Code is developed incrementally
so the student can explain and modify it.
