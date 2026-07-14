from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_generator import ensure_dataset
from src.predict import CATEGORICAL_FIELDS, FEATURE_FIELDS, NUMERIC_FIELDS


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "loan_applicants_sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "smart_lender_model.pkl"
JOBLIB_MODEL_PATH = MODEL_DIR / "smart_lender_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
PLOTS_DIR = BASE_DIR / "static" / "plots"


def _get_xgboost_classifier():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=220,
            max_depth=3,
            learning_rate=0.11,
            subsample=0.92,
            colsample_bytree=0.90,
            reg_lambda=1.2,
            random_state=42,
        )
    except ImportError:
        return GradientBoostingClassifier(random_state=42)


def create_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FIELDS),
            ("categorical", categorical_pipeline, CATEGORICAL_FIELDS),
        ],
        remainder="drop",
    )


def create_models() -> dict[str, object]:
    xgb_model = _get_xgboost_classifier()
    xgb_name = "XGBoost" if xgb_model.__class__.__name__ == "XGBClassifier" else "Gradient Boosting"

    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_leaf=4,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=17),
        xgb_name: xgb_model,
    }


def load_training_data() -> pd.DataFrame:
    df = ensure_dataset(DATA_PATH)
    expected_columns = set(FEATURE_FIELDS + ["Loan_Status"])
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    return df


def train_and_select_model(df: pd.DataFrame) -> tuple[str, Pipeline, dict[str, dict[str, object]]]:
    X = df[FEATURE_FIELDS]
    y = df["Loan_Status"].map({"N": 0, "Y": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -1.0

    for model_name, estimator in create_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_validation_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )

        results[model_name] = {
            "train_accuracy": round(float(train_accuracy), 4),
            "test_accuracy": round(float(test_accuracy), 4),
            "cross_validation_accuracy": round(float(cross_validation_scores.mean()), 4),
            "cross_validation_std": round(float(cross_validation_scores.std()), 4),
            "classification_report": classification_report(
                y_test,
                test_predictions,
                target_names=["Rejected", "Approved"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        }

        if test_accuracy > best_score:
            best_name = model_name
            best_pipeline = pipeline
            best_score = float(test_accuracy)

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    return best_name, best_pipeline, results


def save_eda_plots(df: pd.DataFrame, metrics: dict[str, dict[str, object]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="Loan_Status", hue="Loan_Status", palette=["#bd4f5c", "#1c7c54"], legend=False)
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loan_status_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="Credit_History", hue="Loan_Status", palette=["#bd4f5c", "#1c7c54"])
    plt.title("Credit History vs Loan Status")
    plt.xlabel("Credit History")
    plt.ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "credit_history_vs_status.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df, x="ApplicantIncome", hue="Loan_Status", bins=28, kde=True, palette=["#bd4f5c", "#1c7c54"])
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "income_distribution.png", dpi=160)
    plt.close()

    model_rows = [
        {
            "Model": name,
            "Training Accuracy": values["train_accuracy"],
            "Testing Accuracy": values["test_accuracy"],
        }
        for name, values in metrics.items()
    ]
    metrics_frame = pd.DataFrame(model_rows).melt(id_vars="Model", var_name="Metric", value_name="Accuracy")
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=metrics_frame, x="Model", y="Accuracy", hue="Metric", palette=["#476f95", "#d58a3a"])
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_accuracy_comparison.png", dpi=160)
    plt.close()


def save_artifacts(best_name: str, best_pipeline: Pipeline, metrics: dict[str, dict[str, object]]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_metrics = metrics[best_name]
    payload = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "feature_fields": FEATURE_FIELDS,
        "metrics": best_metrics,
        "all_metrics": metrics,
    }
    joblib.dump(payload, MODEL_PATH)
    joblib.dump(payload, JOBLIB_MODEL_PATH)

    report_payload = {
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "all_models": metrics,
    }
    METRICS_PATH.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


def main() -> None:
    df = load_training_data()
    best_name, best_pipeline, metrics = train_and_select_model(df)
    save_artifacts(best_name, best_pipeline, metrics)
    save_eda_plots(df, metrics)

    best_metrics = metrics[best_name]
    print(f"Dataset: {DATA_PATH} ({len(df)} rows)")
    print(f"Best model: {best_name}")
    print(f"Training accuracy: {best_metrics['train_accuracy']:.3f}")
    print(f"Testing accuracy: {best_metrics['test_accuracy']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved compatibility model: {JOBLIB_MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
