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


@app.route("/", methods=["GET"])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nigerian-Centric Smishing Detection System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 60px auto;
                padding: 20px;
                background-color: #f4f4f4;
            }
            h1 {
                color: #2c3e50;
                font-size: 22px;
            }
            textarea {
                width: 100%;
                height: 100px;
                padding: 10px;
                font-size: 15px;
                border-radius: 6px;
                border: 1px solid #ccc;
                box-sizing: border-box;
            }
            button {
                margin-top: 12px;
                padding: 10px 20px;
                background-color: #2c3e50;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                cursor: pointer;
            }
            button:hover {
                background-color: #1a252f;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                display: none;
            }
            .smishing {
                background-color: #f8d7da;
                color: #721c24;
                display: block !important;
            }
            .legitimate {
                background-color: #d4edda;
                color: #155724;
                display: block !important;
            }
            .neutral {
                background-color: #e2e3e5;
                color: #383d41;
                display: block !important;
            }
            #confidence {
                font-weight: normal;
                font-size: 13px;
                margin-top: 6px;
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>Nigerian-Centric Smishing Detection System</h1>
        <p>Enter an SMS message below to check whether it is legitimate or a smishing attempt.</p>
        <textarea id="messageInput" placeholder="Paste or type the SMS message here..."></textarea><br>
        <button onclick="checkMessage()">Check Message</button>
        <div id="result"></div>

        <script>
            async function checkMessage() {
                const message = document.getElementById('messageInput').value;
                const resultDiv = document.getElementById('result');

                if (!message.trim()) {
                    resultDiv.className = 'neutral';
                    resultDiv.innerHTML = 'Please enter a message first.';
                    return;
                }

                resultDiv.className = 'neutral';
                resultDiv.innerHTML = 'Checking...';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });
                    const data = await response.json();

                    if (data.error) {
                        resultDiv.className = 'neutral';
                        resultDiv.innerHTML = 'Error: ' + data.error;
                        return;
                    }

                    const prediction = data.prediction || '';
                    const confidence = data.confidence !== undefined
                        ? (data.confidence * 100).toFixed(1) + '%'
                        : 'N/A';

                    if (prediction.toLowerCase().includes('smish')) {
                        resultDiv.className = 'smishing';
                        resultDiv.innerHTML = 'Result: ' + prediction.toUpperCase() +
                            '<span id="confidence">Confidence: ' + confidence + '</span>';
                    } else {
                        resultDiv.className = 'legitimate';
                        resultDiv.innerHTML = 'Result: ' + prediction.toUpperCase() +
                            '<span id="confidence">Confidence: ' + confidence + '</span>';
                    }
                } catch (error) {
                    resultDiv.className = 'neutral';
                    resultDiv.innerHTML = 'Something went wrong. Please try again.';
                }
            }
        </script>
    </body>
    </html>
    '''


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