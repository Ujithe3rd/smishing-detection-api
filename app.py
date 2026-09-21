"""
Nigerian SMS Smishing Detection API


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
            .phishing {
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

                    if (prediction === 'Phishing') {
                        resultDiv.className = 'phishing';
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


MAX_BATCH_SIZE = 500  # safeguard against oversized payloads on limited hosting


@app.route("/telecom/batch-predict", methods=["POST"])
def telecom_batch_predict():
    """
    Batch classification endpoint for telecom-operator-level use.
    Screens multiple messages in a single request, simulating how a
    telecom SMS gateway would filter a stream of messages automatically,
    as distinct from the single-message /predict endpoint used by
    individual end users.

    Expects JSON: {"messages": ["msg1", "msg2", ...]}
    """
    data = request.get_json(silent=True)

    if not data or "messages" not in data:
        return jsonify({"error": "Request body must be JSON with a 'messages' field (a list of strings)."}), 400

    messages = data["messages"]

    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "'messages' must be a non-empty list of strings."}), 400

    if len(messages) > MAX_BATCH_SIZE:
        return jsonify({
            "error": f"Batch too large. Maximum {MAX_BATCH_SIZE} messages per request, received {len(messages)}."
        }), 400

    results = []
    phishing_count = 0
    legitimate_count = 0
    skipped_count = 0

    for i, message in enumerate(messages):
        if not isinstance(message, str) or not message.strip():
            results.append({
                "index": i,
                "message": message,
                "error": "Skipped: message must be a non-empty string."
            })
            skipped_count += 1
            continue

        vec = tfidf.transform([message])
        pred_idx = model.predict(vec)[0]
        label = label_encoder.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(vec)[0]
        confidence = float(max(proba))

        # Exact match against the real trained class label, confirmed via
        # /health: ["Phishing", "Promotional", "Safe"]. Promotional is
        # currently treated as non-phishing (allow) alongside Safe.
        is_phishing = label == "Phishing"
        action = "block" if is_phishing else "allow"

        if is_phishing:
            phishing_count += 1
        else:
            legitimate_count += 1

        results.append({
            "index": i,
            "message": message,
            "prediction": label,
            "confidence": round(confidence, 4),
            "recommended_action": action
        })

    return jsonify({
        "results": results,
        "summary": {
            "total_received": len(messages),
            "total_processed": len(messages) - skipped_count,
            "total_skipped": skipped_count,
            "phishing_detected": phishing_count,
            "legitimate": legitimate_count
        }
    })


if __name__ == "__main__":
    # debug=True is fine for local testing only; set to False before any real deployment
    app.run(debug=True, host="0.0.0.0", port=5000)