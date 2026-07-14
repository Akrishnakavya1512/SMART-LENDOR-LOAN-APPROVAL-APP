from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_COLUMNS = [
    "Loan_ID",
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
    "Loan_Status",
]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def generate_loan_dataset(row_count: int = 720, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic education dataset with the common loan schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for index in range(row_count):
        gender = rng.choice(["Male", "Female"], p=[0.76, 0.24])
        married = rng.choice(["Yes", "No"], p=[0.64, 0.36])
        dependents = rng.choice(["0", "1", "2", "3+"], p=[0.55, 0.18, 0.17, 0.10])
        education = rng.choice(["Graduate", "Not Graduate"], p=[0.79, 0.21])
        self_employed = rng.choice(["Yes", "No"], p=[0.14, 0.86])
        property_area = rng.choice(["Urban", "Semiurban", "Rural"], p=[0.33, 0.38, 0.29])
        credit_history = rng.choice([1.0, 0.0], p=[0.84, 0.16])
        loan_term = int(rng.choice([120, 180, 240, 300, 360, 480], p=[0.03, 0.06, 0.08, 0.08, 0.70, 0.05]))

        income_base = 4400 if education == "Graduate" else 3100
        income_multiplier = 1.18 if self_employed == "Yes" else 1.0
        applicant_income = int(max(1200, rng.lognormal(np.log(income_base * income_multiplier), 0.42)))

        if married == "Yes":
            coapplicant_income = int(max(0, rng.lognormal(np.log(1900), 0.55) - 500))
        else:
            coapplicant_income = int(max(0, rng.lognormal(np.log(900), 0.75) - 650))

        total_income = applicant_income + coapplicant_income
        requested_multiplier = rng.uniform(0.018, 0.034)
        loan_amount = int(max(35, min(650, total_income * requested_multiplier + rng.normal(0, 18))))
        monthly_payment_proxy = loan_amount * 1000 / max(loan_term, 1)
        monthly_income_proxy = total_income / 12
        burden_ratio = monthly_payment_proxy / max(monthly_income_proxy, 1)

        score = -3.05
        score += 3.30 * credit_history
        score += 0.62 if education == "Graduate" else -0.25
        score += 0.36 if married == "Yes" else -0.10
        score += 0.48 if property_area == "Semiurban" else 0.18 if property_area == "Urban" else -0.38
        score += 0.64 if total_income >= 7600 else 0.28 if total_income >= 5200 else -0.58
        score += -0.55 if self_employed == "Yes" and total_income < 6500 else 0.08
        score += -1.05 if burden_ratio > 1.05 else -0.38 if burden_ratio > 0.76 else 0.45
        score += rng.normal(0, 0.34)

        approved_probability = _sigmoid(score)
        loan_status = "Y" if approved_probability >= 0.50 else "N"
        if rng.random() < 0.075:
            loan_status = "N" if loan_status == "Y" else "Y"

        rows.append(
            {
                "Loan_ID": f"LP{index + 1:06d}",
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
                "Loan_Status": loan_status,
            }
        )

    df = pd.DataFrame(rows, columns=DATA_COLUMNS)
    missing_plan = {
        "Gender": 0.025,
        "Married": 0.018,
        "Dependents": 0.020,
        "Self_Employed": 0.026,
        "LoanAmount": 0.018,
        "Loan_Amount_Term": 0.012,
        "Credit_History": 0.016,
    }

    for column, rate in missing_plan.items():
        mask = rng.random(row_count) < rate
        df.loc[mask, column] = np.nan

    return df


def ensure_dataset(path: Path, row_count: int = 720, force: bool = False) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return pd.read_csv(path)

    df = generate_loan_dataset(row_count=row_count)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Smart Lender sample dataset.")
    parser.add_argument("--force", action="store_true", help="Regenerate the CSV even when it already exists.")
    args = parser.parse_args()

    output_path = Path("data/loan_applicants_sample.csv")
    generated = ensure_dataset(output_path, force=args.force)
    print(f"Wrote {len(generated)} rows to {output_path}")
