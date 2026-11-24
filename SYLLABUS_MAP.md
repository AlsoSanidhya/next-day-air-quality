# Mapping to the uploaded AML syllabus

Source reviewed visually: TheorySyllabusMl.pdf, CSAI2017P, version 1.0.

| Unit | Uploaded syllabus topics | Where this project uses them |
| --- | --- | --- |
| I - Introduction | Supervised learning; Python, NumPy, pandas, scikit-learn | Loading tables and learning numerical/category targets |
| II - Loss functions | MAE and MSE, classification losses | MAE/RMSE comparison; squared-error regression; logistic model training |
| III - Optimizer function | Gradient descent and its variants | Background for explaining iterative learning; library estimators perform fitting |
| IV - Data preprocessing | Missing data, outliers, scaling, encoding, feature engineering/selection, cross-validation, imbalance | Daily calendar, short forward fills, median imputation, scaling, lag/rolling/calendar features, chronological CV and class weights |
| V - Regression | Linear/multiple regression, logistic regression, R2, overfitting, cross-validation, Ridge and Lasso | Four regressors, Ridge/Lasso tuning, logistic classification, regression evaluation |
| VI - Classification | Decision trees, ensemble learning/random forests, evaluation, ROC and AUC | Three classifiers, confusion matrices, per-class scores, macro-F1 and OvR ROC/AUC |
| VII - Clustering techniques | Hierarchical clustering, K-means, K-medoids, DBSCAN | No clustering is needed for this supervised forecasting question |

## Project-specific applications of the syllabus

- A lag column is feature engineering applied to dated observations.
- A rolling statistic summarises earlier observations as an input.
- Sine/cosine features encode a repeating calendar cycle. They add no learned model.
- TimeSeriesSplit is chronological cross-validation. Training always precedes validation.
- Class weights are a practical way to handle imbalance. The syllabus explicitly
  names over/undersampling as other methods, but this project uses weights to
  avoid a separate resampling dependency.
- RandomForestRegressor applies the tree/forest concept in Unit VI to a numerical
  output. The linked brief requests one nonlinear regression model in addition
  to Linear, Ridge and Lasso.
- Pipeline packages the preprocessing steps and estimator so each CV fold learns
  its own median and scale. It is a scikit-learn programming tool.
- Tkinter is the presentation interface. Its widgets do not change the ML method.

The project need not use every syllabus topic. The chosen methods address the
two specified prediction tasks while keeping the implementation manageable.
