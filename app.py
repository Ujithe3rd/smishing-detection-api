"""
Nigerian SMS Smishing Detection API
------------------------------------
Wraps the trained TF-IDF + Random Forest pipeline in a Flask API.

Expects three files in the same directory:
    smishing_model.joblib
    tfidf_vectorizer.joblib
    label_encoder.joblib

Run locally:
    python app.py
Then POST to http://127.0.0.1:5000/predict with JSON: {"message": "..."}
"""

from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model artifacts once at startup, not per-request
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, "smishing_model.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


@app.route("/health", methods=["GET"])
def health():
    """Simple check that the service and model are up."""
    return jsonify({"status": "ok", "classes": list(label_encoder.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "message" not in data:
        return jsonify({"error": "Request body must be JSON with a 'message' field."}), 400

    message = data["message"]

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "'message' must be a non-empty string."}), 400

    # Same transform used at training time: fit_transform on train, transform only from here on
    vec = tfidf.transform([message])
    pred_idx = model.predict(vec)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    class_probabilities = {
        cls: float(p) for cls, p in zip(label_encoder.classes_, proba)
    }

    return jsonify({
        "message": message,
        "prediction": label,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities
    })


if __name__ == "__main__":
    # debug=True is fine for local testing only; set to False before any real deployment
    app.run(debug=True, host="0.0.0.0", port=5000)
