"""
Nigerian SMS Smishing Detection API


Run locally:
    python app.py
Then POST to http://127.0.0.1:5000/predict with JSON: {"message": "..."}
"""

from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from models import db, Prediction, Feedback, ModelVersion

app = Flask(__name__)

# Render gives you postgres:// but SQLAlchemy wants postgresql://
db_url = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
db_url = db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

CURRENT_MODEL_VERSION = 'v1'

# Load model artifacts once at startup, not per-request
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, "smishing_model.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

# Used only by /retrain. Point this at your original training CSV, and
# adjust TEXT_COLUMN / LABEL_COLUMN below if your header names differ.
ORIGINAL_CORPUS_PATH = os.path.join(MODEL_DIR, "original_training_corpus.csv")
TEXT_COLUMN = "message_text"
LABEL_COLUMN = "label"
RETRAIN_THRESHOLD = 30  # minimum new corrected rows before a retrain is allowed to run


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

    # Log this prediction so it can later be corrected via /feedback
    # and folded into a /retrain run.
    new_prediction = Prediction(
        message_text=message,
        predicted_label=label,
        confidence=confidence,
        source_type="individual",
        model_version=CURRENT_MODEL_VERSION
    )
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({
        "prediction_id": new_prediction.id,
        "message": message,
        "prediction": label,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "model_version": CURRENT_MODEL_VERSION
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

        # Log to the database. flush() assigns new_prediction.id right
        # away without a full commit, so we can return it in this
        # response; the actual commit happens once, after the loop.
        new_prediction = Prediction(
            message_text=message,
            predicted_label=label,
            confidence=confidence,
            source_type="telecom_batch",
            model_version=CURRENT_MODEL_VERSION
        )
        db.session.add(new_prediction)
        db.session.flush()

        results.append({
            "index": i,
            "prediction_id": new_prediction.id,
            "message": message,
            "prediction": label,
            "confidence": round(confidence, 4),
            "recommended_action": action
        })

    db.session.commit()

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


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Records a human correction for a previous prediction. This is the
    data that /retrain later uses to improve the model.

    Expects JSON: {"prediction_id": 123, "corrected_label": "Safe"}
    """
    data = request.get_json(silent=True)

    if not data or "prediction_id" not in data or "corrected_label" not in data:
        return jsonify({"error": "Request body must include 'prediction_id' and 'corrected_label'."}), 400

    prediction_id = data["prediction_id"]
    corrected_label = data["corrected_label"]

    valid_labels = list(label_encoder.classes_)
    if corrected_label not in valid_labels:
        return jsonify({"error": f"'corrected_label' must be one of {valid_labels}."}), 400

    prediction = Prediction.query.get(prediction_id)
    if not prediction:
        return jsonify({"error": f"No prediction found with id {prediction_id}."}), 404

    new_feedback = Feedback(
        prediction_id=prediction_id,
        corrected_label=corrected_label
    )
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({
        "status": "feedback recorded",
        "feedback_id": new_feedback.id,
        "prediction_id": prediction_id,
        "corrected_label": corrected_label
    })


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Retrains the model on the original corpus plus any accumulated,
    human-corrected feedback. Saves new model artifacts under a new
    version tag but does NOT automatically replace the live model in
    this app, that is a separate, deliberate step (see the notes that
    follow the code).
    """
    unused_feedback = Feedback.query.filter_by(used_in_retrain=False).all()

    if len(unused_feedback) < RETRAIN_THRESHOLD:
        return jsonify({
            "status": "skipped",
            "reason": f"only {len(unused_feedback)} new corrected rows, threshold is {RETRAIN_THRESHOLD}"
        }), 200

    if not os.path.exists(ORIGINAL_CORPUS_PATH):
        return jsonify({
            "error": f"Original training corpus not found at {ORIGINAL_CORPUS_PATH}."
        }), 500

    original_df = pd.read_csv(ORIGINAL_CORPUS_PATH)

    feedback_rows = []
    for fb in unused_feedback:
        pred = Prediction.query.get(fb.prediction_id)
        feedback_rows.append({
            TEXT_COLUMN: pred.message_text,
            LABEL_COLUMN: fb.corrected_label
        })
    feedback_df = pd.DataFrame(feedback_rows)

    combined_df = pd.concat(
        [original_df[[TEXT_COLUMN, LABEL_COLUMN]], feedback_df],
        ignore_index=True
    )

    X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
        combined_df[TEXT_COLUMN],
        combined_df[LABEL_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=combined_df[LABEL_COLUMN]
    )

    new_encoder = LabelEncoder()
    y_train = new_encoder.fit_transform(y_train_raw)
    y_test = new_encoder.transform(y_test_raw)

    new_vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X_train_vec = new_vectorizer.fit_transform(X_train_text)
    X_test_vec = new_vectorizer.transform(X_test_text)

    # SMOTE applied to the training partition only, per your project's
    # documented methodology.
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    new_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    )
    new_model.fit(X_train_resampled, y_train_resampled)

    y_pred = new_model.predict(X_test_vec)
    new_accuracy = accuracy_score(y_test, y_pred)
    new_f1 = f1_score(y_test, y_pred, average="weighted")

    version_tag = f"v{ModelVersion.query.count() + 2}"  # +2 because v1 is the original, unlogged model
    joblib.dump(new_model, os.path.join(MODEL_DIR, f"smishing_model_{version_tag}.joblib"))
    joblib.dump(new_vectorizer, os.path.join(MODEL_DIR, f"tfidf_vectorizer_{version_tag}.joblib"))
    joblib.dump(new_encoder, os.path.join(MODEL_DIR, f"label_encoder_{version_tag}.joblib"))

    version_record = ModelVersion(
        version_tag=version_tag,
        training_rows=len(combined_df),
        accuracy=new_accuracy,
        f1_score=new_f1,
        notes=f"Retrained with {len(unused_feedback)} feedback-corrected rows"
    )
    db.session.add(version_record)

    for fb in unused_feedback:
        fb.used_in_retrain = True

    db.session.commit()

    return jsonify({
        "status": "retrained",
        "version": version_tag,
        "accuracy": round(new_accuracy, 4),
        "f1_score": round(new_f1, 4),
        "training_rows": len(combined_df)
    })


if __name__ == "__main__":
    # debug=True is fine for local testing only; set to False before any real deployment
    app.run(debug=True, host="0.0.0.0", port=5000)