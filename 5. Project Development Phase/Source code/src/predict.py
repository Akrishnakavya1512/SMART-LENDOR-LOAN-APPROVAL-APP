from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "smart_lender_model.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "smart_lender_model.joblib"

CATEGORICAL_FIELDS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_FIELDS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

FEATURE_FIELDS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


class ApplicantInputError(ValueError):
    """Raised when form input cannot be converted to model features."""


@dataclass(frozen=True)
class PredictionResult:
    decision: str
    risk_level: str
    confidence: float
    model_name: str
    applicant: dict[str, object]
    guidance: str


def get_model_status() -> dict[str, object]:
    active_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not active_path.exists():
        return {
            "available": False,
            "path": str(MODEL_PATH),
            "model_name": None,
            "test_accuracy": None,
        }

    payload = joblib.load(active_path)
    metrics = payload.get("metrics", {})
    return {
        "available": True,
        "path": str(active_path),
        "model_name": payload.get("model_name", "Unknown model"),
        "test_accuracy": metrics.get("test_accuracy"),
    }


def _coerce_numeric(field: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ApplicantInputError(f"{field} must be a valid number.") from exc

    if number < 0:
        raise ApplicantInputError(f"{field} cannot be negative.")

    return number


def build_applicant_frame(form_data: Mapping[str, object]) -> pd.DataFrame:
    missing = [field for field in FEATURE_FIELDS if form_data.get(field) in (None, "")]
    if missing:
        raise ApplicantInputError(f"Missing required field: {missing[0]}")

    row: dict[str, object] = {}
    for field in CATEGORICAL_FIELDS:
        row[field] = str(form_data[field])

    for field in NUMERIC_FIELDS:
        row[field] = _coerce_numeric(field, form_data[field])

    row["ApplicantIncome"] = int(row["ApplicantIncome"])
    row["CoapplicantIncome"] = int(row["CoapplicantIncome"])
    row["LoanAmount"] = float(row["LoanAmount"])
    row["Loan_Amount_Term"] = int(row["Loan_Amount_Term"])
    row["Credit_History"] = float(row["Credit_History"])

    return pd.DataFrame([row], columns=FEATURE_FIELDS)


def _extract_probability(model, applicant_frame: pd.DataFrame, prediction: int) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(applicant_frame)[0]
        class_index = list(model.classes_).index(prediction)
        return float(probabilities[class_index])

    return 0.5


def predict_loan_status(form_data: Mapping[str, object]) -> PredictionResult:
    active_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not active_path.exists():
        raise FileNotFoundError(MODEL_PATH)

    payload = joblib.load(active_path)
    model = payload["pipeline"]
    applicant_frame = build_applicant_frame(form_data)

    prediction = int(model.predict(applicant_frame)[0])
    confidence = _extract_probability(model, applicant_frame, prediction)

    if prediction == 1:
        decision = "Loan Approved"
        risk_level = "Low Risk"
        guidance = "Applicant profile matches repayment patterns from the training data."
    else:
        decision = "Loan Rejected"
        risk_level = "High Risk"
        guidance = "Applicant profile requires manual review before approval."

    return PredictionResult(
        decision=decision,
        risk_level=risk_level,
        confidence=round(confidence * 100, 2),
        model_name=payload.get("model_name", "Smart Lender model"),
        applicant=applicant_frame.iloc[0].to_dict(),
        guidance=guidance,
    )
