# Next-Day Air Quality for Delhi

Individual Applied Machine Learning project by **Sanidhya Negi**
(GitHub: **AlsoSanidhya**), B.Tech CSE specialising in AI and ML.

Predict next-day PM2.5 concentration and next-day AQI category using historical
pollution readings. A simple Tkinter desktop window demonstrates predictions
and lets you change the selected day's PM2.5 for a viva exercise.

## Run in VS Code

Open this project folder and select `.venv/Scripts/python.exe` as the interpreter.
The prepared environment and trained model are already present on the original PC.

```powershell
.\.venv\Scripts\python.exe app.py
```

Choose a historical date and click **Predict next day**. Compare the prediction
with the recorded next-day reading. The date selector covers test-period origins
and the final available date. That final date predicts 2 July 2020, for which the
dataset has no observed answer. Editing PM2.5 changes only that input; the other
inputs continue to come from the selected historical dates.

## Recreate from a fresh checkout

Use Python 3.13 with Tkinter. Tkinter is included in standard Windows Python and
Anaconda installations. Linux may require its distribution's `python3-tk` package.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe make_report.py
.\.venv\Scripts\python.exe app.py
```

`train_models.py` creates the saved pipelines and every result table/figure.
`make_report.py` recreates the four-page PDF from those results. The generated
model is excluded from Git and is recreated by training.

## Main results

Training uses targets before **1 January 2020**. The final held-out evaluation
covers **1 January-1 July 2020**, with **183 test days** for each task.
Models are chosen using four-fold chronological cross-validation inside the
training period, with a one-row gap.

| Regressor | Test MAE (ug/m3) | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: |
| Persistence | 19.745 | 29.453 | 0.721 |
| Linear | 19.135 | 27.333 | 0.760 |
| Ridge | 19.168 | 27.331 | 0.760 |
| Lasso | 19.373 | 27.681 | 0.753 |
| Random forest | 18.983 | 27.229 | 0.761 |

Cross-validation selects **Lasso, alpha=1** for the app. It improves held-out MAE
over persistence by **1.89%**. Random forest has the lowest test MAE, while
training-period CV chooses Lasso.

| Classifier | Accuracy | Macro-F1 (six classes) | Mean defined OvR AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.694 | 0.489 | 0.878 |
| Decision tree | 0.743 | 0.645 | 0.908 |
| Random forest | 0.738 | 0.588 | 0.919 |

Cross-validation selects **logistic regression**. The test set contains no Good
examples, so Good's ROC/AUC is undefined and the AUC average covers five classes.
Macro-F1 uses all six predefined class labels, including zero for Good. Only
three Severe examples occur in this period. The selected classifier has zero
Satisfactory recall. Read the per-class reports alongside overall accuracy.

## Read the code in this order

1. `explore_data.py` loads the CSV, filters Delhi and shows missingness and a graph.
2. `src/data_preparation.py` prepares the daily calendar, features and targets.
3. `train_models.py` trains, selects and evaluates the models.
4. `app.py` loads the trained pipelines and displays a prediction.
5. `VIVA_GUIDE.md` explains the concepts and gives small code-change exercises.

The four executed notebooks in `notebooks/` show EDA, preprocessing, training and
evaluation in order. They reuse the same Python functions as the app and scripts.

## Data and missing values

Source: [Air Quality Data in India, Rohan Rao](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
The raw `data/raw/city_day.csv` is included for reproducibility. To download it
again, open the source in a browser, choose Download, extract the ZIP and place
`city_day.csv` in `data/raw/`. The file has 29,531 rows and 16 columns. Delhi has
2,009 days with no missing calendar days or duplicate dates.

Five input pollutants keep the project manageable: PM2.5, PM10, NO2, CO and O3.
Xylene is excluded because 38.88% is missing. Other unused pollutants are also
excluded as a scope choice. Inputs are carried forward for at most two days.
Remaining gaps use a training-fold median. Missing targets are excluded per task.
Regression also requires an observed current-day PM2.5, ensuring fair comparison
with persistence. Pollution spikes are retained. See `results/missingness.csv`
and `results/missingness_by_year.csv`.

The 29 features are today's PM2.5, 1/2/3/7-day lags of five pollutants,
past-only 3/7-day PM2.5 rolling means/stds, and four calendar features for the
target day. Targets use `.shift(-1)`. Lag inputs use positive shifts. The rolling
windows end at t-1. See the hand-checked `results/alignment_example.csv`.

## Syllabus and project scope

`SYLLABUS_MAP.md` maps the implementation to the uploaded seven-unit syllabus.
The project uses supervised learning, pandas/NumPy/scikit-learn, preprocessing,
linear/logistic regression, Ridge/Lasso, tree ensembles and classification metrics.
Feature engineering and time-series splitting specialise the syllabus topics
to the project's dates. The nonlinear regressor applies the forest idea to a
numerical target, as requested by the project brief.

This is a historical Delhi experiment using data ending in July 2020. Recent
forecasts would require recent readings and validation on recent years. PM2.5
and AQI category are separately predicted outputs and may disagree. The observed
AQI bucket is the category target; it is not obtained by thresholding predicted
PM2.5. Models do not use same-day AQI as a circular answer.

## Submission files

- `notebooks/01-eda.ipynb` through `04-evaluation.ipynb`
- `results/metrics.csv`, per-class reports, predictions, ROC/AUC and figures
- `report/report.pdf` (four pages)
- `REPRODUCIBILITY.md`, `SYLLABUS_MAP.md` and `VIVA_GUIDE.md`

The external brief's group-formation and semester-history rules belong to its
course setup. This repository records an individual project, as specified by the
student. No historical checkpoint tags or backdated contributions are claimed.

## References and assistance

- [Project brief](https://github.com/tali7c/Applied-Machine-Learning/tree/main/projects/P02-air-quality)
- Uploaded `TheorySyllabusMl.pdf` for CSAI2017P, syllabus version 1.0, Units I-VII.
- Teacher-provided lab examples for regression, classification and preprocessing.
- ChatGPT assisted setup, implementation, validation, report generation and
  explanations. The viva guide is intended to help the student explain and modify
  the retained code.

Implementation and experiments were completed with ChatGPT assistance. The
student's remaining contribution is to study, practise changes and present the
project with an understanding of its results and limitations.
