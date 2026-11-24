"""Run the complete historical next-day forecasting experiment."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
    make_scorer, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_preparation import (
    AQI_CLASSES, CITY, RANDOM_SEED, ROOT, TEST_START,
    build_features, build_targets, load_city_data,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
CV_SPLITS = 4
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.1, 1.0, 10.0]
TREE_DEPTH = 5
FOREST_TREES = 100


def make_pipeline(model):
    # fit() learns the median and scale on each training fold only.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def regression_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
    }


def run_project():
    for folder in [RESULTS, FIGURES, MODELS]:
        folder.mkdir(parents=True, exist_ok=True)
    daily = load_city_data()
    features = build_features(daily)
    targets = build_targets(daily)

    missing = daily.isna().mean().mul(100).rename("missing_percent")
    missing.to_csv(RESULTS / "missingness.csv")
    yearly_missing = daily.isna().groupby(daily.index.year).mean().mul(100)
    yearly_missing.index.name = "Year"
    yearly_missing.to_csv(RESULTS / "missingness_by_year.csv")
    daily["AQI_Bucket"].value_counts().reindex(AQI_CLASSES, fill_value=0).to_csv(
        RESULTS / "class_counts.csv", header=["count"]
    )
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index, daily["PM2.5"], linewidth=0.7)
    plt.title(f"{CITY} observed daily PM2.5")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.tight_layout()
    plt.savefig(FIGURES / "pm25_history.png", dpi=160)
    plt.close()

    # Prove the shift direction on one row using the original daily readings.
    example_date = features.index[0]
    example = pd.DataFrame({
        "origin_date": [str(example_date.date())],
        "lag1_date": [str((example_date - pd.Timedelta(days=1)).date())],
        "lag1_value": [features.loc[example_date, "PM2.5_lag_1"]],
        "today_value": [features.loc[example_date, "PM2.5_today"]],
        "target_date": [str(targets.loc[example_date, "target_date"].date())],
        "tomorrow_value": [targets.loc[example_date, "PM2.5_tomorrow"]],
    })
    assert example.iloc[0]["lag1_value"] == daily.loc[
        example_date - pd.Timedelta(days=1), "PM2.5"
    ]
    assert example.iloc[0]["tomorrow_value"] == daily.loc[
        example_date + pd.Timedelta(days=1), "PM2.5"
    ]
    example.to_csv(RESULTS / "alignment_example.csv", index=False)

    # All regressors and persistence use exactly the same observed test answers.
    regression = features.join(targets[[
        "target_date", "PM2.5_tomorrow", "PM2.5_today_observed"
    ]]).dropna(subset=["PM2.5_tomorrow", "PM2.5_today_observed"])
    train = regression[regression["target_date"] < TEST_START]
    test = regression[regression["target_date"] >= TEST_START]
    if train.empty or test.empty:
        raise ValueError("The chosen chronological split has an empty period.")
    X_train = train[features.columns]
    X_test = test[features.columns]
    y_train = train["PM2.5_tomorrow"]
    y_test = test["PM2.5_tomorrow"]
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=1)
    rows = []
    regression_predictions = pd.DataFrame({
        "origin_date": test.index,
        "target_date": test["target_date"].values,
        "actual": y_test.values,
        "Persistence": test["PM2.5_today_observed"].values,
    })
    baseline = regression_metrics(y_test, test["PM2.5_today_observed"])
    rows.append({"task": "regression", "model": "Persistence", **baseline})
    regression_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=30000),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=FOREST_TREES, max_depth=6, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=1,
        ),
    }
    best_regression = None
    best_cv_mae = float("inf")
    best_regression_name = ""
    for name, model in regression_models.items():
        print(f"Training regression: {name}", flush=True)
        pipeline = make_pipeline(model)
        if name in ["Ridge", "Lasso"]:
            alphas = RIDGE_ALPHAS if name == "Ridge" else LASSO_ALPHAS
            search = GridSearchCV(
                pipeline, {"model__alpha": alphas}, cv=cv,
                scoring="neg_mean_absolute_error", n_jobs=1, error_score="raise",
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            cv_mae = -search.best_score_
            pd.DataFrame(search.cv_results_).to_csv(RESULTS / f"{name}_cv.csv", index=False)
        else:
            cv_mae = -cross_val_score(
                pipeline, X_train, y_train, cv=cv,
                scoring="neg_mean_absolute_error", error_score="raise",
            ).mean()
            pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        regression_predictions[name] = predictions
        alpha = pipeline.named_steps["model"].get_params().get("alpha")
        rows.append({
            "task": "regression", "model": name, "cv_mae": cv_mae,
            "alpha": alpha, **regression_metrics(y_test, predictions),
        })
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_regression = pipeline
            best_regression_name = name
    regression_predictions.to_csv(RESULTS / "regression_predictions.csv", index=False)
    plt.figure(figsize=(11, 4))
    plt.plot(test["target_date"], y_test, label="Actual", linewidth=1)
    plt.plot(test["target_date"], regression_predictions[best_regression_name],
             label=f"Predicted ({best_regression_name})", linewidth=1)
    plt.xlabel("Target date")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.title("Held-out 2020 next-day predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "predicted_vs_actual.png", dpi=160)
    plt.close()

    classification = features.join(targets[["target_date", "AQI_tomorrow"]])
    classification = classification.dropna(subset=["AQI_tomorrow"])
    class_train = classification[classification["target_date"] < TEST_START]
    class_test = classification[classification["target_date"] >= TEST_START]
    Xc_train, yc_train = class_train[features.columns], class_train["AQI_tomorrow"]
    Xc_test, yc_test = class_test[features.columns], class_test["AQI_tomorrow"]
    class_models = {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=TREE_DEPTH, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=FOREST_TREES, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1,
        ),
    }
    macro_f1 = make_scorer(f1_score, average="macro", labels=AQI_CLASSES, zero_division=0)
    best_classifier, best_class_name, best_f1 = None, "", -1
    class_predictions = pd.DataFrame({
        "origin_date": class_test.index,
        "target_date": class_test["target_date"].values,
        "actual": yc_test.values,
    })
    roc_rows = []
    for name, model in class_models.items():
        print(f"Training classification: {name}", flush=True)
        pipeline = make_pipeline(model)
        cv_f1 = cross_val_score(
            pipeline, Xc_train, yc_train, cv=cv, scoring=macro_f1, error_score="raise"
        ).mean()
        pipeline.fit(Xc_train, yc_train)
        predicted = pipeline.predict(Xc_test)
        class_predictions[name] = predicted
        report = classification_report(
            yc_test, predicted, labels=AQI_CLASSES, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(RESULTS / f"{name}_classification_report.csv")
        ConfusionMatrixDisplay.from_predictions(
            yc_test, predicted, labels=AQI_CLASSES, xticks_rotation=35, cmap="Blues"
        )
        plt.title(name)
        plt.tight_layout()
        plt.savefig(FIGURES / f"{name}_confusion_matrix.png", dpi=140)
        plt.close()

        probabilities = pipeline.predict_proba(Xc_test)
        learned_classes = pipeline.named_steps["model"].classes_
        auc_values = []
        plt.figure(figsize=(7, 5))
        for label in AQI_CLASSES:
            binary_actual = (yc_test == label).astype(int)
            if label not in learned_classes or binary_actual.nunique() < 2:
                roc_rows.append({"model": name, "class": label, "auc": np.nan,
                                 "status": "undefined: class absent from train or test"})
                continue
            column = list(learned_classes).index(label)
            scores = probabilities[:, column]
            auc = roc_auc_score(binary_actual, scores)
            auc_values.append(auc)
            roc_rows.append({"model": name, "class": label, "auc": auc, "status": "defined"})
            fpr, tpr, _ = roc_curve(binary_actual, scores)
            plt.plot(fpr, tpr, label=f"{label} ({auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{name}: one-vs-rest ROC")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES / f"{name}_roc.png", dpi=140)
        plt.close()
        rows.append({
            "task": "classification", "model": name, "cv_macro_f1": cv_f1,
            "accuracy": accuracy_score(yc_test, predicted),
            "macro_f1": f1_score(yc_test, predicted, labels=AQI_CLASSES,
                                 average="macro", zero_division=0),
            "roc_auc_ovr_macro": np.mean(auc_values) if auc_values else np.nan,
        })
        if cv_f1 > best_f1:
            best_classifier, best_class_name, best_f1 = pipeline, name, cv_f1

    class_predictions.to_csv(RESULTS / "classification_predictions.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(RESULTS / "roc_auc_by_class.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    summary = {
        "city": CITY, "test_start": TEST_START, "seed": RANDOM_SEED,
        "features": list(features.columns), "regression_model": best_regression_name,
        "classification_model": best_class_name,
        "regression_train_rows": len(train), "regression_test_rows": len(test),
        "classification_train_rows": len(class_train), "classification_test_rows": len(class_test),
        "first_date": str(daily.index.min().date()), "last_date": str(daily.index.max().date()),
        "forecast_origin_first": str(features.index.min().date()),
        "forecast_origin_last": str(features.index.max().date()),
        "model_selection": "training-period chronological cross-validation only",
        "regression_test_first_target": str(test["target_date"].min().date()),
        "regression_test_last_target": str(test["target_date"].max().date()),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump({"regression": best_regression, "classification": best_classifier,
                 "summary": summary}, MODELS / "forecast_models.joblib")
    print("\nCompleted. Models were selected using training-period CV.")
    print(metrics.round(4).to_string(index=False))
    return metrics


if __name__ == "__main__":
    run_project()
