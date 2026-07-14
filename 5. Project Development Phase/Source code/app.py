import os
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from src.predict import ApplicantInputError, get_model_status, predict_loan_status


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "smart-lender-dev-key")


@app.route("/")
def index():
    model_status = get_model_status()
    return render_template("home.html", model_status=model_status)


@app.route("/architecture")
def architecture():
    model_status = get_model_status()
    return render_template("architecture.html", model_status=model_status)


@app.route("/health")
def health():
    model_status = get_model_status()
    status_code = 200 if model_status["available"] else 503
    return (
        jsonify(
            {
                "status": "ok" if model_status["available"] else "model_missing",
                "model_available": model_status["available"],
                "model_name": model_status["model_name"],
                "test_accuracy": model_status["test_accuracy"],
            }
        ),
        status_code,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_status = get_model_status()

    if request.method == "POST":
        try:
            result = predict_loan_status(request.form)
        except FileNotFoundError:
            flash("Model artifact not found. Run python -m src.train_model first.", "error")
            return redirect(url_for("predict"))
        except ApplicantInputError as exc:
            flash(str(exc), "error")
            return redirect(url_for("predict"))

        return render_template("submit.html", result=result, model_status=model_status)

    return render_template("predict.html", model_status=model_status)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
