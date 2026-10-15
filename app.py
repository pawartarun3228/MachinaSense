
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "machinasense.joblib"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "reports" / "test_predictions.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load the trained model.
try:
    model_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    app.logger.exception("Failed to load the trained model.")
    model_bundle = None


def load_test_data():
    """Load the chronological 15% held-out test partition."""
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        return df

    # The training pipeline uses a chronological 70/15/15 split.
    test_start = int(len(df) * 0.85)
    df = df.iloc[test_start:].copy()

    # The downloaded CSV may not contain a UDI column.
    if "UDI" not in df.columns:
        df.insert(0, "UDI", df.index + 1)

    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model_ready": model_bundle is not None,
        "dataset_ready": DATA_PATH.exists()
    })


@app.route("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Metrics file not found"}), 404

    with open(METRICS_PATH, encoding="utf-8") as file:
        return jsonify(json.load(file))


@app.route("/api/sensor-data")
def sensor_data():
    """Return sensor readings from the entire test partition."""
    try:
        df = load_test_data()

        if df is None:
            return jsonify({"error": "Dataset file not found"}), 503

        columns = [
            "UDI",
            "Air_temperature",
            "Process_temperature",
            "Rotational_speed",
            "Torque",
            "Tool_wear"
        ]

        missing = [col for col in columns if col not in df.columns]

        if missing:
            return jsonify({
                "error": "Required sensor columns are missing",
                "missing_columns": missing
            }), 500

        try:
            limit = int(request.args.get("limit", 1500))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400

        if limit < 1:
            return jsonify({"error": "limit must be positive"}), 400

        limit = min(limit, len(df), 1500)
        records = df[columns].head(limit)

        return jsonify({
            "count": len(records),
            "source": "AI4I 2020 held-out test partition",
            "data": records.astype(object).where(
                pd.notna(records), None
            ).to_dict(orient="records")
        })

    except Exception:
        app.logger.exception("Sensor endpoint failed")
        return jsonify({"error": "Could not load sensor data"}), 500



def classify_severity(probability):
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High"
    return "Medium"


def recommend_action(probability):
    if probability >= 0.80:
        return "Prioritize prompt maintenance assessment"
    if probability >= 0.60:
        return "Schedule a priority maintenance inspection"
    return "Schedule a preventive inspection"


@app.route("/api/alerts")
def alerts():
    if not PREDICTIONS_PATH.exists():
        return jsonify({"error": "Prediction file not found"}), 404

    try:
        predictions = pd.read_csv(PREDICTIONS_PATH)

        if "predicted_failure" not in predictions.columns:
            return jsonify({
                "error": "Prediction file has no predicted_failure column"
            }), 500

        test_data = load_test_data()

        if "UDI" not in predictions.columns and test_data is not None:
            if len(test_data) == len(predictions):
                predictions.insert(
                    0, "UDI", test_data["UDI"].to_numpy()
                )

        if "failure_probability" not in predictions.columns:
            return jsonify({
                "error": "Prediction file has no failure_probability column"
            }), 500

        flagged = predictions[
            predictions["predicted_failure"] == 1
        ].copy()

        flagged["severity"] = flagged["failure_probability"].apply(
            classify_severity
        )
        flagged["recommended_action"] = flagged["failure_probability"].apply(
            recommend_action
        )

        return jsonify(
            flagged.tail(100).astype(object).where(
                pd.notna(flagged.tail(100)), None
            ).to_dict(orient="records")
        )

    except Exception:
        app.logger.exception("Alert endpoint failed")
        return jsonify({"error": "Could not load alerts"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    if model_bundle is None:
        return jsonify({"error": "Trained model unavailable"}), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    feature_cols = model_bundle["feature_cols"]
    numeric_cols = model_bundle["numeric_cols"]

    missing = [col for col in feature_cols if col not in payload]

    if missing:
        return jsonify({
            "error": "Required features are missing",
            "missing_features": missing
        }), 400

    try:
        row = pd.DataFrame(
            [{col: payload[col] for col in feature_cols}],
            columns=feature_cols
        )

        for col in numeric_cols:
            row[col] = pd.to_numeric(row[col], errors="raise")

        if row[numeric_cols].isna().any().any():
            return jsonify({
                "error": "Sensor values cannot be empty"
            }), 400

        probabilities = model_bundle["classifier"].predict_proba(row)[0]
        classes = list(model_bundle["classifier"].classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        scaled = model_bundle["anomaly_scaler"].transform(
            row[numeric_cols]
        )
        anomaly_score = float(
            -model_bundle["anomaly_detector"].decision_function(
                scaled
            )[0]
        )

        threshold = float(model_bundle["threshold"])

        if probability >= max(threshold, 0.75):
            severity = "High"
        elif probability >= threshold:
            severity = "Medium"
        else:
            severity = "Low"

        return jsonify({
            "predicted_failure": int(probability >= threshold),
            "failure_probability": probability,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "severity": severity,
            "note": "Model estimate only; inspect the machine before acting."
        })

    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed"}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    """Run predictions on an uploaded sensor CSV."""
    if model_bundle is None:
        return jsonify({"error": "Trained model unavailable"}), 503

    file = request.files.get("file")

    if file is None or not file.filename:
        return jsonify({"error": "Choose a CSV file"}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    try:
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "CSV contains no records"}), 400

        feature_cols = model_bundle["feature_cols"]
        numeric_cols = model_bundle["numeric_cols"]

        missing = [col for col in feature_cols if col not in df.columns]

        if missing:
            return jsonify({
                "error": "CSV is missing required model features",
                "missing_features": missing
            }), 400

        X = df[feature_cols].copy()

        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X[numeric_cols].isna().any().any():
            return jsonify({
                "error": "CSV contains missing or invalid numeric values"
            }), 400

        classifier = model_bundle["classifier"]
        classes = list(classifier.classes_)
        positive_index = classes.index(1)

        probabilities = classifier.predict_proba(X)[:, positive_index]
        threshold = float(model_bundle["threshold"])

        result = df.copy()
        result["failure_probability"] = probabilities
        result["predicted_failure"] = (
            probabilities >= threshold
        ).astype(int)

        preview = result.head(100)

        return jsonify({
            "rows_analyzed": len(result),
            "alerts": int(result["predicted_failure"].sum()),
            "results": preview.astype(object).where(
                pd.notna(preview), None
            ).to_dict(orient="records"),
            "note": "Model estimates only, not confirmed mechanical faults."
        })

    except (ValueError, pd.errors.ParserError) as exc:
        return jsonify({"error": f"Invalid CSV: {exc}"}), 400
    except Exception:
        app.logger.exception("CSV analysis failed")
        return jsonify({"error": "CSV analysis could not be completed"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
