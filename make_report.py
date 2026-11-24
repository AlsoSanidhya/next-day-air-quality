"""Create the four-page project report from the saved experiment results."""
import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

from src.data_preparation import ROOT


def make_report():
    results = ROOT / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(results / "metrics.csv")
    folder = ROOT / "report"
    folder.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 8))

    def table(data, widths=None):
        item = Table(data, colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17364d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f6")]),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#17364d")),
        ]))
        story.append(item)
        story.append(Spacer(1, 12))

    paragraph("Next-Day Air Quality for Delhi", "Title")
    paragraph("Applied Machine Learning | Sanidhya Negi | B.Tech CSE (AI and ML)")
    paragraph("1. Question and dataset", "Heading1")
    paragraph("Can pollution readings available by day t predict PM2.5 concentration and the AQI category on day t+1? The experiment compares regression models with persistence, which predicts tomorrow's PM2.5 as today's observed value.")
    paragraph("The source is Rohan Rao's Air Quality Data in India (2015-2020), compiled from CPCB data. city_day.csv contains 29,531 city-day rows and 16 columns. Delhi contributes 2,009 days from 1 January 2015 to 1 July 2020. It was selected for its long daily coverage and low missingness in the prediction targets. Dates were parsed, sorted and reindexed to a complete daily calendar. Delhi has zero duplicate dates and zero missing calendar days.")
    paragraph("Missing values and data policy", "Heading2")
    table([["Column", "Missing", "Percent"], ["PM2.5", "2", "0.10%"],
           ["PM10", "77", "3.83%"], ["NO2", "2", "0.10%"],
           ["CO", "0", "0.00%"], ["O3", "84", "4.18%"],
           ["Xylene", "781", "38.88%"], ["AQI bucket", "10", "0.50%"]])
    paragraph("The inputs use PM2.5, PM10, NO2, CO and O3. Xylene and other unused columns are excluded to keep the feature set manageable. Input readings are forward-filled for at most two days, then any remaining feature gaps are filled with training-fold medians. Targets always retain their observed values, and rows without an observed target are excluded for that task. High pollution spikes are retained because they may represent real events. Missing percentages by year are saved in results/missingness_by_year.csv.")
    paragraph('Source: <link href="https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india">Kaggle dataset</link>. Project brief: <link href="https://github.com/tali7c/Applied-Machine-Learning/tree/main/projects/P02-air-quality">P02 air-quality brief</link>.')
    story.append(PageBreak())

    paragraph("2. Inputs, targets and evaluation design", "Heading1")
    paragraph("There are 29 inputs: today's PM2.5; lags 1, 2, 3 and 7 for each of five pollutants; PM2.5 rolling means and standard deviations over 3 and 7 days; and four calendar features for the target day (month, weekday, sine and cosine of day-of-year). The cyclic encoding places December and January close together. The seven-day lag requires excluding the first seven forecast origins.")
    paragraph("A positive shift creates a past input. A negative shift creates the future answer. Rolling statistics are computed after shift(1), so their windows end at t-1. Today's PM2.5 is an explicitly named additional input available at the end of day t. Future target columns are excluded from X.")
    alignment = pd.read_csv(results / "alignment_example.csv").iloc[0]
    paragraph(f"Hand-checked example: origin {alignment['origin_date']}; yesterday {alignment['lag1_date']} has PM2.5 {alignment['lag1_value']}; today's value is {alignment['today_value']}; target {alignment['target_date']} has observed PM2.5 {alignment['tomorrow_value']}. The script asserts the lag and target against the original daily data.")
    paragraph(f"Targets before 1 January 2020 form training data. Targets from 1 January to 1 July 2020 form the untouched final test period. Regression has {summary['regression_train_rows']} training rows and {summary['regression_test_rows']} test rows; classification has {summary['classification_train_rows']} training rows and {summary['classification_test_rows']} test rows. Regression also requires observed origin-day PM2.5 so every regressor and persistence is compared on the same dates.")
    paragraph("Within training data, four expanding TimeSeriesSplit folds use a one-row gap. Median imputation and standardization are inside each pipeline and are fitted again in each fold. Ridge and Lasso penalties are selected by cross-validation MAE. The deployed regression and classification models are selected using cross-validation MAE and macro-F1 respectively. The final test scores do not choose the deployed model.")
    story.append(Image(str(results / "figures" / "pm25_history.png"), width=6.6*inch, height=2.4*inch))
    paragraph("The plot shows recurring annual peaks and sharp daily spikes. The early-2020 section is lower than the preceding winter peak. This graph alone cannot establish the causes of a change.")
    story.append(PageBreak())

    paragraph("3. Regression results", "Heading1")
    reg = metrics[metrics.task == "regression"]
    table([["Model", "CV MAE", "Test MAE", "Test RMSE", "Test R2"]] + [
        [r.model, "-" if pd.isna(r.cv_mae) else f"{r.cv_mae:.2f}",
         f"{r.mae:.2f}", f"{r.rmse:.2f}", f"{r.r2:.3f}"] for r in reg.itertuples()
    ])
    selected = reg[reg.model == summary["regression_model"]].iloc[0]
    baseline = reg[reg.model == "Persistence"].iloc[0]
    improvement = 100 * (baseline.mae - selected.mae) / baseline.mae
    paragraph(f"Cross-validation selected {escape(summary['regression_model'])}. Its held-out MAE is {selected.mae:.2f} ug/m3, compared with {baseline.mae:.2f} for persistence, an improvement of {improvement:.2f}%. This is a modest gain. Random forest has the lowest test MAE in this run, but the app retains the model chosen using training-period CV. MAE measures average absolute error; RMSE puts more weight on large errors; R2 compares squared errors with a test-mean predictor.")
    story.append(Image(str(results / "figures" / "predicted_vs_actual.png"), width=6.6*inch, height=2.4*inch))
    predictions = pd.read_csv(results / "regression_predictions.csv")
    predictions["absolute_error"] = (predictions.actual - predictions[summary["regression_model"]]).abs()
    worst = predictions.nlargest(3, "absolute_error")
    paragraph("Largest held-out errors", "Heading2")
    table([["Target date", "Actual", "Predicted", "Absolute error"]] + [
        [r["target_date"][:10], f"{r['actual']:.2f}",
         f"{r[summary['regression_model']]:.2f}", f"{r['absolute_error']:.2f}"]
        for _, r in worst.iterrows()
    ])
    paragraph("Abrupt changes are difficult because recent readings may provide limited warning. Results cover a partial year with few severe events. Forecasts for recent years would require recent input data and a new validation exercise. Training on one city also limits transfer to other cities.")
    story.append(PageBreak())

    paragraph("4. Classification, demonstration and syllabus", "Heading1")
    classes = metrics[metrics.task == "classification"]
    table([["Model", "CV macro-F1", "Accuracy", "Macro-F1", "OvR AUC*"]] + [
        [r.model, f"{r.cv_macro_f1:.3f}", f"{r.accuracy:.3f}",
         f"{r.macro_f1:.3f}", f"{r.roc_auc_ovr_macro:.3f}"] for r in classes.itertuples()
    ])
    paragraph("*OvR AUC is the mean across the five classes with positives and negatives in the test set. Good has no test examples, so its AUC is undefined. Macro-F1 uses all six predefined classes, assigning zero to undefined class F1. Confusion matrices, individual class reports and ROC curves are saved for all three classifiers.")
    paragraph(f"Cross-validation selected {summary['classification_model']}. Class weights give greater training importance to rarer classes. Accuracy can hide poor minority-class recognition, so per-class recall and macro-F1 are reported. The selected logistic model has zero recall for Satisfactory on this test period. Severe has only three test examples, making its recall unstable. These limitations are part of the result.")
    story.append(Image(str(results / "figures" / "LogisticRegression_confusion_matrix.png"), width=3.6*inch, height=2.7*inch))
    paragraph("Run python train_models.py, python make_report.py, then python app.py. The Tkinter app selects a historical origin date and predicts the following day using saved model pipelines. An editable current PM2.5 field supports a simple viva what-if exercise. The PM2.5 and AQI category come from separate models, so they may not always agree. There is no live-data connection.")
    paragraph("Syllabus mapping: Unit I covers Python libraries and supervised learning; Unit II covers MAE and MSE; Unit IV covers cleaning, scaling, feature engineering, splitting and imbalance; Unit V covers linear/logistic regression, Ridge, Lasso and R2; Unit VI covers trees, random forests, classification evaluation and ROC/AUC. Random-forest regression applies the same tree-ensemble idea to a numerical target, as the project brief requests a nonlinear regressor. Unit III optimizer theory supports learning explanations, and Unit VII clustering is not needed for this forecasting question.")
    paragraph("Assistance and reproducibility: teacher lab examples informed the coding style; ChatGPT assisted implementation and explanations. Requirements, random seed 42, notebook order, data source and run instructions are recorded in the repository. The student should be able to explain every retained line.")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 26, "Sanidhya Negi | Applied Machine Learning | Delhi")
        canvas.drawRightString(553, 26, f"{doc.page}")

    document = SimpleDocTemplate(str(folder / "report.pdf"), rightMargin=42, leftMargin=42,
                                 topMargin=35, bottomMargin=40)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created report/report.pdf")


if __name__ == "__main__":
    make_report()
