# Understand and modify your air-quality project

## Your 45-second explanation

"My project predicts tomorrow's PM2.5 concentration and AQI category for Delhi.
I use daily historical pollution readings. I inspect missing values, sort and
check dates, then create past-day inputs and next-day targets. I train on earlier
years and evaluate on 2020. I compare regression models with persistence and
compare three classifiers. Cross-validation selects the model used by the desktop
app. I report errors and per-class performance, including the weak results."

## Understand the files in order

1. `explore_data.py` is your first exercise. Read it line by line.
2. `src/data_preparation.py` contains the city, test date and input configuration.
   `load_city_data()` loads the CSV and makes the daily calendar.
   `build_features()` creates X. `build_targets()` creates tomorrow's observed y.
3. `train_models.py` contains the fit/predict/evaluate work. Start with the model
   dictionaries, then read `make_pipeline()` and the regression training loop.
   Read classification after that.
4. `app.py` reads saved pipelines. `predict_day()` prepares one row and calls
   predict(). The remaining class mainly builds labels, boxes and buttons.
5. `make_report.py` formats already measured results into a PDF.

The four notebooks let you inspect the same stages one cell at a time.

## Essential code concepts

| Code | Meaning |
| --- | --- |
| pd.read_csv(...) | Load the raw table |
| df[df['City'] == CITY] | Select the chosen city |
| .copy() | Create a table we can modify independently |
| pd.to_datetime(...) | Convert date strings to dates |
| .reindex(dates) | Insert a row for every calendar day |
| .ffill(limit=2) | Carry an earlier reading forward at most two missing days |
| .shift(1) | Place yesterday's value beside today's row |
| .shift(-1) | Place tomorrow's observed answer beside today's row |
| .rolling(3).mean() | Calculate the mean in a three-row window |
| .dropna(subset=[target]) | Exclude rows whose answer is unknown |
| SimpleImputer(strategy='median') | Learn training medians and fill remaining input gaps |
| StandardScaler() | Learn training means/stds and standardize inputs |
| Pipeline(...) | Keep imputation, scaling and model together |
| model.fit(X_train, y_train) | Learn from inputs and known answers |
| model.predict(X_test) | Produce answers for held-out inputs |
| joblib.dump/load | Save/load fitted pipelines |

## One actual row to explain

For origin **2015-01-08**, today's PM2.5 is **229.90**. Its lag-one input comes
from **2015-01-07** and equals **217.87**. Its next-day target comes from
**2015-01-09** and equals **201.66**. See results/alignment_example.csv.

The rolling-three mean for this origin uses 5, 6 and 7 January. Calling shift(1)
before rolling keeps the window ending yesterday. Today's PM2.5 is separately
included as PM2.5_today. The model receives 29 inputs and learns one output per
task. Tomorrow's target columns never enter X.

## Likely viva questions

**What is PM2.5?** Fine particles with diameter at most 2.5 micrometres.
The concentration unit is micrograms per cubic metre (ug/m3).

**Regression versus classification?** Regression predicts a number such as
48.04 ug/m3. Classification chooses a category such as Moderate.

**What is the target?** The observed next day's PM2.5 or AQI bucket.
The AQI bucket comes from the dataset and accounts for more than PM2.5 alone.

**Why Delhi?** It has a long continuous daily record, only two missing PM2.5
readings and ten missing AQI categories. Xylene has much higher missingness.

**Why avoid a random split?** Randomly mixing days can let later information
influence evaluation of earlier conditions. Forecasting is tested chronologically.

**Why split on target date?** The forecast is scored for tomorrow. An origin on
31 December 2019 predicts 1 January 2020, so it belongs in the test period.

**What is leakage?** Information unavailable at forecast time entering training
or input preparation. Examples include future readings in rolling windows or
learning imputation medians from the final test period.

**Why median imputation?** It gives a simple replacement fitted on training data
and is less sensitive to extremes than a mean. Missing answers are excluded.

**Why keep spikes?** High pollution can be a real event. Automatically capping
it could remove the very events we want to forecast.

**What is persistence?** Predict tomorrow's PM2.5 as today's observed PM2.5.
It is a meaningful baseline because neighbouring days often resemble one another.

**What do Linear, Ridge and Lasso change?** Linear regression fits a weighted
sum of inputs. Ridge adds an L2 penalty, shrinking weights. Lasso adds an L1
penalty and can set some weights to zero. Larger alpha means a stronger penalty.

**Why scaling?** Inputs have different units and ranges. Scaling makes their
sizes comparable, especially for penalised and logistic models. Tree models do
not need scaling, but using one shared pipeline keeps the teaching code uniform.

**What is the forest?** Many decision trees whose predictions are combined.
Regression averages numerical tree outputs. Classification combines class predictions.

**Why logistic regression for categories?** It learns class probabilities and
chooses the predicted label. Scikit-learn handles the multiclass case.

**What is cross-validation?** Repeated training/validation within the training
period. TimeSeriesSplit expands the training window into the future. gap=1
separates fold boundaries. Final test data remains held out during selection.

**Why class_weight='balanced'?** Rare labels receive more weight while fitting.
This addresses imbalance without generating extra examples.

**What are MAE, RMSE and R2?** MAE is average absolute error. RMSE squares errors
before averaging and square-rooting, giving large errors more influence. R2
compares squared prediction error with a mean-value predictor. R2 can be negative.

**Why macro-F1?** It averages class F1 values with equal class importance. Accuracy
can look good when a model mostly predicts common classes. F1 combines precision
and recall. Recall measures how many actual members of a class were found.

**What is ROC/AUC here?** Treat one category as positive and the rest as negative,
vary a probability threshold and plot true-positive rate against false-positive
rate. AUC summarises that curve. Good has no test positives, so its AUC is undefined.

**What are your actual results?** Lasso was selected by CV with alpha=1. Its MAE
is 19.37 versus persistence 19.75, a 1.89% gain. The selected logistic classifier
has accuracy 69.40% and six-class macro-F1 0.489. It misses Satisfactory on this
test period. Only three Severe examples make that class's result unstable.

**Why does the app use a model with worse test results than another model?** We
choose using training CV, then report the final test honestly. Choosing after
viewing test scores would use test information for model selection.

**Can it predict tomorrow in the current year?** It needs recent input readings
and validation on recent years. The provided app demonstrates historical dates.

**Why can the two outputs disagree?** Separate models predict PM2.5 and the AQI
category, and the AQI category reflects multiple pollutants.

## Small changes to practise yourself

Make one change, save, run and explain the effect. Record the original value first.
After training changes, rerun train_models.py, recreate the report and reopen app.py.

| Teacher's request | Where to change | What to explain |
| --- | --- | --- |
| Show three rows | explore_data.py: head() to head(3) | Changes the preview length |
| Change the graph title | explore_data.py: plt.title(...) | Changes the displayed title |
| Display missing percent rounded | Add .round(2) to the printed percentages | Presentation precision |
| Show a different city | CITY in src/data_preparation.py | Coverage, missingness and scores may change; update the Delhi app labels too |
| Change the test period | TEST_START in src/data_preparation.py | A different chronological evaluation question |
| Use a five-day lag too | LAGS = [1, 2, 3, 5, 7] | Adds past-day inputs; earliest usable history remains seven days |
| Change rolling windows | for window in [3, 7] in build_features | Window size controls smoothing; keep windows no larger than earliest excluded history or update that rule |
| Use mean imputation | make_pipeline: strategy='median' to 'mean' | The replacement becomes more sensitive to extremes |
| Try a stronger Ridge penalty | RIDGE_ALPHAS in train_models.py | CV tests the added alpha using training folds |
| Make the tree shallower | TREE_DEPTH = 3 | May reduce overfitting and expressive power |
| Use fewer forest trees | FOREST_TREES = 50 | Training cost and stability may change |
| Change CV folds | CV_SPLITS = 3 | Changes the expanding validation partitions |
| Remove class weights | class_weight=None in model constructors | Rare classes may receive less attention |
| Inspect learned coefficients | In run_project(), after the regression loop, add print(best_regression.named_steps['model'].coef_) | Coefficients correspond to the feature columns of the selected Lasso |
| Change one current-day reading | Edit the PM2.5 field in app.py's window | Changes one input while retaining historical lag inputs |

## A short study order

First explain the load/filter/date lines. Next trace the hand-checked row. Then
explain fit/predict and the chronological split. Learn the baseline and the
metrics before comparing model families. Finally practise the small edits above
in VS Code. You do not need to memorise report formatting or Tkinter layout code
to explain the forecasting pipeline, but you should know what each file does.
