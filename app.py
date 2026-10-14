"""Flask web demonstration application for Federated IVF Live-Birth Prediction.

Allows interactive clinical exploration by taking patient and cycle parameters,
applying the fitted preprocessing pipeline, and generating real-time live-birth predictions
using the privacy-preserving federated model weights.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import create_model, set_parameters
from preprocessing.preprocess import transform_data

app = Flask(__name__)

# Paths to models and preprocessors
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "federated_model.pt")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessor.joblib")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "data", "feature_schema.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "fl_metrics.json")

# Global state variables
preprocessor = None
model = None
schema = None
test_auc_str = "0.647"


def load_artifacts():
    """Load model, preprocessor, and schema at app startup."""
    global preprocessor, model, schema, test_auc_str

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    else:
        raise FileNotFoundError(f"Missing preprocessor at {PREPROCESSOR_PATH}. Run python main.py first.")

    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                test_auc_str = f"{d['final_test_metrics']['auc']:.3f}"
        except Exception:
            pass

    input_dim = len(preprocessor["all_feature_names"])
    model = create_model(input_dim=input_dim)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()


def predict_probability(input_dict: dict) -> float:
    """Transform patient input dict and predict live-birth probability using federated model."""
    full_dict = schema.get("default_values", {}).copy() if schema and "default_values" in schema else {}
    full_dict.update(input_dict)
    df_single = pd.DataFrame([full_dict])

    # Convert numeric fields
    for col in preprocessor["numeric_cols"]:
        if col in df_single.columns:
            df_single[col] = pd.to_numeric(df_single[col], errors="coerce")

    # Transform features
    df_transformed = transform_data(df_single, preprocessor, target_col="none")
    x_tensor = torch.tensor(df_transformed.values.astype(np.float32))

    with torch.no_grad():
        prob = float(model(x_tensor).item())

    return float(np.clip(prob, 0.0, 1.0))


@app.route("/", methods=["GET"])
def index():
    """Render home page with input form and default values."""
    defaults = schema.get("default_values", {}) if schema else {}
    return render_template(
        "index.html",
        values=defaults,
        prob_pct=55.2,
        test_auc=test_auc_str
    )


@app.route("/predict", methods=["POST"])
def predict_ui():
    """Handle form submission and render predicted probability."""
    form_data = request.form.to_dict()

    # Cast numeric values
    clean_inputs = {}
    for k, v in form_data.items():
        try:
            if "." in v:
                clean_inputs[k] = float(v)
            else:
                clean_inputs[k] = int(v)
        except ValueError:
            clean_inputs[k] = v

    prob = predict_probability(clean_inputs)
    prob_pct = round(prob * 100, 1)

    return render_template(
        "index.html",
        values=clean_inputs,
        prob_pct=prob_pct,
        test_auc=test_auc_str
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST API endpoint for inference."""
    try:
        data = request.get_json(force=True)
        prob = predict_probability(data)
        return jsonify({
            "status": "success",
            "predicted_probability": prob,
            "predicted_percentage": f"{round(prob * 100, 2)}%",
            "model_type": "Federated Learning (Flower FedAvg)",
            "disclaimer": "This is an educational research demonstration and is not medical advice or a clinical prediction tool."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    load_artifacts()
    print("\n" + "=" * 60)
    print("FEDERATED IVF WEB DEMO RUNNING")
    print("URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
