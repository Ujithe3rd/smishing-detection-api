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
            .feedback-box { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,.12); font-weight: normal; font-size: 13px; }
            .feedback-box .fb { margin: 8px 5px 0 0; padding: 7px 10px; font-size: 12px; border-radius: 5px; }
            .feedback-box .safe { background:#198754; } .feedback-box .phish { background:#dc3545; } .feedback-box .promo { background:#6f42c1; }
            #feedbackStatus { margin-top: 8px; font-size: 12px; font-weight: normal; }
        </style>
    </head>
    <body>
        <h1>Nigerian-Centric Smishing Detection System</h1><p><a href="/telecom/dashboard">Telecom Dashboard</a> | <a href="/model-management">Model Management</a></p>
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
                    resultDiv.textContent = 'Please enter a message first.';
                    return;
                }

                resultDiv.className = 'neutral';
                resultDiv.textContent = 'Checking...';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });
                    const data = await response.json();

                    if (data.error) {
                        resultDiv.className = 'neutral';
                        resultDiv.textContent = 'Error: ' + data.error;
                        return;
                    }

                    const prediction = data.prediction || '';
                    const confidence = data.confidence !== undefined
                        ? (data.confidence * 100).toFixed(1) + '%'
                        : 'N/A';

                    resultDiv.className = prediction === 'Phishing' ? 'phishing' : 'legitimate';
                    resultDiv.innerHTML = '';

                    const resultText = document.createElement('div');
                    resultText.textContent = 'Result: ' + prediction.toUpperCase();
                    resultDiv.appendChild(resultText);

                    const confidenceSpan = document.createElement('span');
                    confidenceSpan.id = 'confidence';
                    confidenceSpan.textContent = 'Confidence: ' + confidence;
                    resultDiv.appendChild(confidenceSpan);

                    const feedbackBox = document.createElement('div');
                    feedbackBox.className = 'feedback-box';

                    const feedbackQuestion = document.createElement('span');
                    feedbackQuestion.textContent = 'Is this prediction correct?';
                    feedbackBox.appendChild(feedbackQuestion);

                    const feedbackOptions = [
                        { label: 'Legitimate / Safe', value: 'Safe', className: 'safe' },
                        { label: 'Phishing', value: 'Phishing', className: 'phish' },
                        { label: 'Promotional', value: 'Promotional', className: 'promo' }
                    ];

                    feedbackOptions.forEach(function(option) {
                        const button = document.createElement('button');
                        button.type = 'button';
                        button.className = 'fb ' + option.className;
                        button.textContent = option.label;
                        button.addEventListener('click', function() {
                            sendFeedback(data.prediction_id, option.value);
                        });
                        feedbackBox.appendChild(button);
                    });

                    resultDiv.appendChild(feedbackBox);

                    const feedbackStatus = document.createElement('div');
                    feedbackStatus.id = 'feedbackStatus';
                    resultDiv.appendChild(feedbackStatus);
                } catch (error) {
                    console.error('Prediction error:', error);
                    resultDiv.className = 'neutral';
                    resultDiv.textContent = 'Something went wrong. Please try again.';
                }
            }
        async function sendFeedback(predictionId, correctedLabel) {
            const status = document.getElementById('feedbackStatus');
            status.textContent = 'Saving feedback...';
            try {
                const response = await fetch('/feedback', {
                    method: 'POST', headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({prediction_id: predictionId, corrected_label: correctedLabel})
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Could not save feedback.');
                status.textContent = 'Feedback recorded. Thank you.';
            } catch (e) { status.textContent = 'Feedback error: ' + e.message; }
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

@app.route("/telecom/dashboard", methods=["GET"])
def telecom_dashboard():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telecom SMS Screening Dashboard</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#f4f7fb;color:#172033}
.topbar{background:#172033;color:white;padding:18px 32px;display:flex;justify-content:space-between;align-items:center}
.brand h1{margin:0;font-size:21px}.brand p{margin:5px 0 0;color:#cbd5e1;font-size:13px}
.status{display:flex;align-items:center;gap:8px;font-size:13px}.dot{width:9px;height:9px;border-radius:50%;background:#22c55e}
.container{max-width:1250px;margin:30px auto;padding:0 20px 40px}
.upload-card,.results-card{background:white;border:1px solid #e3e8ef;border-radius:12px;box-shadow:0 4px 16px rgba(23,32,51,.06)}
.upload-card{padding:26px}.section-title{margin:0 0 7px;font-size:19px}.section-subtitle{margin:0 0 22px;color:#667085;font-size:14px}
.dropzone{display:block;border:2px dashed #b8c3d1;border-radius:10px;padding:34px 20px;text-align:center;background:#fafcff;cursor:pointer}
.dropzone:hover,.dropzone.dragover{border-color:#2563eb;background:#f3f7ff}.upload-icon{font-size:34px;margin-bottom:8px}
.dropzone strong{display:block;font-size:16px}.dropzone span{display:block;margin-top:7px;color:#667085;font-size:13px}#fileInput{display:none}
.file-name{margin-top:12px;color:#2563eb;font-size:13px;min-height:18px}.controls{margin-top:18px;display:flex;gap:10px;flex-wrap:wrap}
button{border:0;border-radius:7px;padding:11px 18px;font-weight:600;cursor:pointer;font-size:14px}
.primary{background:#2563eb;color:white}.primary:hover{background:#1d4ed8}.secondary{background:#e9eef5;color:#334155}
button:disabled{opacity:.55;cursor:not-allowed}.message{margin-top:15px;padding:12px 14px;border-radius:7px;display:none;font-size:13px}
.error{background:#fee2e2;color:#991b1b;display:block}.info{background:#e0f2fe;color:#075985;display:block}
.results-card{margin-top:24px;overflow:hidden;display:none}.results-header{padding:20px 22px;border-bottom:1px solid #e3e8ef}
.results-header h2{margin:0;font-size:18px}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:18px 22px;background:#fafbfc;border-bottom:1px solid #e3e8ef}
.stat{background:white;border:1px solid #e4e9f0;border-radius:9px;padding:15px}.stat-label{color:#667085;font-size:12px;margin-bottom:7px}
.stat-value{font-size:23px;font-weight:700}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:850px}
th,td{padding:13px 15px;text-align:left;border-bottom:1px solid #edf0f4;font-size:13px;vertical-align:top}
th{background:#f8fafc;color:#475467;font-size:12px;text-transform:uppercase;letter-spacing:.03em}td.message-cell{max-width:430px;line-height:1.45}
.badge{display:inline-block;padding:5px 9px;border-radius:20px;font-size:11px;font-weight:700}.phishing{background:#fee2e2;color:#b91c1c}
.safe{background:#dcfce7;color:#166534}.promotional{background:#fef3c7;color:#92400e}.action{font-weight:700;text-transform:uppercase;font-size:11px}
.confidence-bar{width:100px;height:7px;background:#e5e7eb;border-radius:10px;overflow:hidden;display:inline-block;margin-right:7px;vertical-align:middle}
.confidence-fill{height:100%;background:#2563eb}.operator-actions{white-space:nowrap}
.action-btn{padding:6px 9px;margin-right:4px;font-size:11px;border:1px solid #d0d5dd;background:white;color:#344054}
.action-btn.selected{background:#172033;color:white;border-color:#172033}.feedback-controls{margin-top:8px}.mini{padding:5px 7px;margin:2px;border:0;border-radius:4px;font-size:10px;color:white;cursor:pointer}.mini.safe{background:#198754}.mini.phish{background:#dc3545}.mini.promo{background:#6f42c1}.feedback-saved{font-size:10px;color:#198754;font-weight:700}
.footer-note{padding:15px 22px;color:#667085;font-size:12px;background:#fafbfc}
@media(max-width:760px){.topbar{padding:16px 18px}.container{margin-top:18px}.summary{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header class="topbar">
<div class="brand"><h1>Telecom SMS Screening</h1><p>Batch smishing detection and operator review</p></div>
<div class="status"><span class="dot"></span> API Online</div>
</header>
<main class="container">
<section class="upload-card">
<h2 class="section-title">Screen an SMS Batch</h2>
<p class="section-subtitle">Upload a CSV or TXT file containing SMS messages. The system supports up to 500 messages per screening batch.</p>
<label class="dropzone" id="dropzone" for="fileInput">
<div class="upload-icon">↑</div><strong>Choose a CSV or TXT file</strong>
<span>CSV: one SMS message per row (first column), or TXT: one message per line</span>
<input id="fileInput" type="file" accept=".csv,.txt,text/csv,text/plain">
</label>
<div class="file-name" id="fileName"></div>
<div class="controls">
<button class="primary" id="runBtn" onclick="runScreening()" disabled>Run Screening</button>
<button class="secondary" id="clearBtn" onclick="clearDashboard()">Clear</button>
</div>
<div id="messageBox" class="message"></div>
</section>
<section class="results-card" id="resultsCard">
<div class="results-header"><h2>Screening Results</h2></div>
<div class="summary">
<div class="stat"><div class="stat-label">TOTAL RECEIVED</div><div class="stat-value" id="totalReceived">0</div></div>
<div class="stat"><div class="stat-label">PROCESSED</div><div class="stat-value" id="totalProcessed">0</div></div>
<div class="stat"><div class="stat-label">FLAGGED / SMISHING</div><div class="stat-value" id="phishingCount">0</div></div>
<div class="stat"><div class="stat-label">CLEARED</div><div class="stat-value" id="legitimateCount">0</div></div>
</div>
<div class="table-wrap"><table>
<thead><tr><th>#</th><th>SMS Message</th><th>Prediction</th><th>Confidence</th><th>Recommended Action</th><th>Operator Action</th></tr></thead>
<tbody id="resultsBody"></tbody>
</table></div>
<div class="footer-note">The recommended action is generated by the existing batch prediction API. Operator action buttons are review controls. Feedback buttons below each result save the human correction to the database for retraining.</div>
</section>
</main>
<script>
let selectedFile=null;
const fileInput=document.getElementById('fileInput'),dropzone=document.getElementById('dropzone'),fileName=document.getElementById('fileName'),runBtn=document.getElementById('runBtn'),messageBox=document.getElementById('messageBox');
fileInput.addEventListener('change',function(){if(this.files.length)setFile(this.files[0]);});
dropzone.addEventListener('dragover',function(e){e.preventDefault();dropzone.classList.add('dragover');});
dropzone.addEventListener('dragleave',function(){dropzone.classList.remove('dragover');});
dropzone.addEventListener('drop',function(e){e.preventDefault();dropzone.classList.remove('dragover');if(e.dataTransfer.files.length)setFile(e.dataTransfer.files[0]);});
function setFile(file){const name=file.name.toLowerCase();if(!name.endsWith('.csv')&&!name.endsWith('.txt')){showMessage('Please select a .csv or .txt file.',true);return;}selectedFile=file;fileName.textContent='Selected: '+file.name;runBtn.disabled=false;hideMessage();}
function showMessage(text,isError=false){messageBox.textContent=text;messageBox.className='message '+(isError?'error':'info');}
function hideMessage(){messageBox.textContent='';messageBox.className='message';}
function parseCSVLine(line){const values=[];let current='',insideQuotes=false;for(let i=0;i<line.length;i++){const char=line[i];if(char==='"'){if(insideQuotes&&line[i+1]==='"'){current+='"';i++;}else{insideQuotes=!insideQuotes;}}else if(char===','&&!insideQuotes){values.push(current.trim());current='';}else{current+=char;}}values.push(current.trim());return values;}
function extractMessages(text,name){const lines=text.split(/\r?\n/).filter(line=>line.trim()!=='');if(!lines.length)return[];if(name.toLowerCase().endsWith('.txt'))return lines.map(line=>line.trim()).filter(Boolean);const rows=lines.map(parseCSVLine),firstRow=rows[0].map(v=>v.toLowerCase().trim()),headers=['message','messages','message_text','sms','sms_text','text'];let col=firstRow.findIndex(v=>headers.includes(v));if(col!==-1)rows.shift();else col=0;return rows.map(row=>(row[col]||'').trim()).filter(Boolean);}
async function runScreening(){if(!selectedFile)return;runBtn.disabled=true;runBtn.textContent='Screening...';showMessage('Reading file and sending messages to the screening API...');try{const text=await selectedFile.text(),messages=extractMessages(text,selectedFile.name);if(!messages.length)throw new Error('No SMS messages were found in the selected file.');if(messages.length>500)throw new Error('The file contains '+messages.length+' messages. The API maximum is 500 messages per batch.');const response=await fetch('/telecom/batch-predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:messages})}),data=await response.json();if(!response.ok||data.error)throw new Error(data.error||'The screening request failed.');renderResults(data);showMessage('Screening completed successfully for '+data.summary.total_processed+' message(s).');}catch(error){showMessage(error.message||'Something went wrong. Please try again.',true);}finally{runBtn.disabled=false;runBtn.textContent='Run Screening';}}
function getPredictionClass(prediction){const value=String(prediction||'').toLowerCase();if(value==='phishing'||value==='smishing')return'phishing';if(value==='promotional')return'promotional';return'safe';}
function renderResults(data){const summary=data.summary||{},results=data.results||[];document.getElementById('totalReceived').textContent=summary.total_received??0;document.getElementById('totalProcessed').textContent=summary.total_processed??0;document.getElementById('phishingCount').textContent=summary.phishing_detected??0;document.getElementById('legitimateCount').textContent=summary.legitimate??0;const body=document.getElementById('resultsBody');body.innerHTML='';results.forEach((item,position)=>{const row=document.createElement('tr');if(item.error){row.innerHTML='<td>'+(position+1)+'</td><td class="message-cell">'+escapeHtml(String(item.message??''))+'</td><td colspan="4">'+escapeHtml(item.error)+'</td>';body.appendChild(row);return;}const confidence=Number(item.confidence||0),percent=(confidence*100).toFixed(1),predictionClass=getPredictionClass(item.prediction),action=String(item.recommended_action||'allow').toUpperCase();row.innerHTML='<td>'+(position+1)+'</td><td class="message-cell">'+escapeHtml(item.message)+'</td><td><span class="badge '+predictionClass+'">'+escapeHtml(item.prediction)+'</span></td><td><span class="confidence-bar"><span class="confidence-fill" style="width:'+Math.min(confidence*100,100)+'%"></span></span>'+percent+'%</td><td class="action">'+escapeHtml(action)+'</td><td class="operator-actions"><button class="action-btn" onclick="selectAction(this)">Block</button><button class="action-btn" onclick="selectAction(this)">Flag</button><button class="action-btn" onclick="selectAction(this)">Pass</button><div class="feedback-controls"><button class="mini safe" onclick="sendFeedback("+item.prediction_id+", 'Safe', this)">Safe</button><button class="mini phish" onclick="sendFeedback("+item.prediction_id+", 'Phishing', this)">Phishing</button><button class="mini promo" onclick="sendFeedback("+item.prediction_id+", 'Promotional', this)">Promo</button></div></td>';body.appendChild(row);});document.getElementById('resultsCard').style.display='block';document.getElementById('resultsCard').scrollIntoView({behavior:'smooth',block:'start'});}
function selectAction(button){const parent=button.parentElement;parent.querySelectorAll('.action-btn').forEach(btn=>btn.classList.remove('selected'));button.classList.add('selected');}
function clearDashboard(){selectedFile=null;fileInput.value='';fileName.textContent='';document.getElementById('resultsBody').innerHTML='';document.getElementById('resultsCard').style.display='none';runBtn.disabled=true;hideMessage();}
function escapeHtml(value){return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');}
async function sendFeedback(predictionId, correctedLabel, btn){ try { const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction_id:predictionId,corrected_label:correctedLabel})}); const d=await r.json(); if(!r.ok) throw new Error(d.error||'Feedback failed'); const cell=btn.parentElement; cell.innerHTML='<span class=\"feedback-saved\">Saved: '+correctedLabel+'</span>'; } catch(e){ alert(e.message); } }
</script>
</body>
</html>
    '''




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





@app.route('/database', methods=['GET'])
def database_dashboard():
    total_predictions = Prediction.query.count()
    phishing = Prediction.query.filter_by(predicted_label='Phishing').count()
    safe = Prediction.query.filter_by(predicted_label='Safe').count()
    promotional = Prediction.query.filter_by(predicted_label='Promotional').count()
    individual = Prediction.query.filter_by(source_type='individual').count()
    telecom = Prediction.query.filter_by(source_type='telecom_batch').count()
    total_feedback = Feedback.query.count()
    unused_feedback = Feedback.query.filter_by(used_in_retrain=False).count()
    recent = Prediction.query.order_by(Prediction.id.desc()).limit(50).all()
    feedback_rows = Feedback.query.order_by(Feedback.id.desc()).limit(30).all()
    versions = ModelVersion.query.order_by(ModelVersion.id.desc()).limit(20).all()
    def esc(v):
        return str(v or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
    pred_rows=''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1f}%</td><td>{}</td><td>{}</td></tr>'.format(p.id,esc(p.message_text),esc(p.predicted_label),p.confidence*100,esc(p.source_type),esc(p.model_version)) for p in recent)
    if not pred_rows: pred_rows='<tr><td colspan="6">No predictions recorded yet.</td></tr>'
    fb_rows=''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(f.id, f.prediction_id, esc(f.corrected_label), 'Used in retraining' if f.used_in_retrain else 'Pending') for f in feedback_rows)
    if not fb_rows: fb_rows='<tr><td colspan="4">No feedback recorded yet.</td></tr>'
    ver_rows=''.join('<tr><td>{}</td><td>{}</td><td>{:.2%}</td><td>{:.2%}</td><td>{}</td></tr>'.format(esc(v.version_tag),v.training_rows,v.accuracy,v.f1_score,esc(v.notes)) for v in versions)
    if not ver_rows: ver_rows='<tr><td colspan="5">No model versions recorded yet.</td></tr>'
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Database & Analytics</title><style>body{font-family:Arial;margin:0;background:#f4f7fb;color:#172033}.bar{background:#172033;color:white;padding:20px 30px;display:flex;justify-content:space-between}.bar a{color:#cbd5e1;margin-left:18px}.wrap{max-width:1200px;margin:30px auto;padding:0 20px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:white;border:1px solid #e3e8ef;border-radius:12px;padding:22px;margin-bottom:20px;box-shadow:0 4px 16px rgba(23,32,51,.06)}.stat{padding:18px;border:1px solid #e3e8ef;border-radius:10px;background:white}.label{font-size:12px;color:#667085}.value{font-size:25px;font-weight:bold;margin-top:7px}table{width:100%;border-collapse:collapse;display:block;overflow-x:auto;white-space:nowrap}th,td{text-align:left;padding:11px;border-bottom:1px solid #edf0f4;font-size:12px}th{background:#f8fafc;font-size:11px}h2{margin-top:0}@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}.bar{padding:16px}.bar a{margin-left:8px}}</style></head><body><div class="bar"><div><b>SMS Smishing Detection System</b><br><small>Database & Analytics</small></div><div><a href="/">User</a><a href="/telecom/dashboard">Telecom</a><a href="/model-management">Models</a></div></div><div class="wrap"><div class="stats"><div class="stat"><div class="label">Total Predictions</div><div class="value">__TOTAL__</div></div><div class="stat"><div class="label">Phishing Detected</div><div class="value">__PHISH__</div></div><div class="stat"><div class="label">Safe</div><div class="value">__SAFE__</div></div><div class="stat"><div class="label">Feedback / Pending</div><div class="value">__FB__ / __UNUSED__</div></div></div><div class="card"><h2>Prediction History</h2><p>Individual: __INDIVIDUAL__ &nbsp;|&nbsp; Telecom batch: __TELECOM__ &nbsp;|&nbsp; Promotional: __PROMO__</p><table><tr><th>ID</th><th>Message</th><th>Prediction</th><th>Confidence</th><th>Source</th><th>Model</th></tr>__PRED_ROWS__</table></div><div class="card"><h2>Human Feedback</h2><p>Corrections are stored and become eligible for retraining when the configured threshold is reached.</p><table><tr><th>Feedback ID</th><th>Prediction ID</th><th>Corrected Label</th><th>Status</th></tr>__FB_ROWS__</table></div><div class="card"><h2>Model Version History</h2><table><tr><th>Version</th><th>Training Rows</th><th>Accuracy</th><th>F1 Score</th><th>Notes</th></tr>__VER_ROWS__</table></div></div></body></html>'''.replace('__TOTAL__',str(total_predictions)).replace('__PHISH__',str(phishing)).replace('__SAFE__',str(safe)).replace('__PROMO__',str(promotional)).replace('__INDIVIDUAL__',str(individual)).replace('__TELECOM__',str(telecom)).replace('__FB__',str(total_feedback)).replace('__UNUSED__',str(unused_feedback)).replace('__PRED_ROWS__',pred_rows).replace('__FB_ROWS__',fb_rows).replace('__VER_ROWS__',ver_rows)

@app.route('/model-management', methods=['GET'])
def model_management():
    total_predictions = Prediction.query.count()
    feedback_count = Feedback.query.count()
    unused_feedback = Feedback.query.filter_by(used_in_retrain=False).count()
    versions = ModelVersion.query.order_by(ModelVersion.id.desc()).limit(10).all()
    rows = ''.join('<tr><td>{}</td><td>{}</td><td>{:.2%}</td><td>{:.2%}</td><td>{}</td></tr>'.format(v.version_tag, v.training_rows, v.accuracy, v.f1_score, v.notes or '-') for v in versions)
    if not rows:
        rows = '<tr><td colspan="5">No retrained versions recorded yet.</td></tr>'
    html = '''
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model Management</title>
<style>body{font-family:Arial;margin:0;background:#f4f7fb;color:#172033}.bar{background:#172033;color:white;padding:20px 30px;display:flex;justify-content:space-between}.bar a{color:#cbd5e1;margin-left:18px}.wrap{max-width:1100px;margin:30px auto;padding:0 20px}.card{background:white;border:1px solid #e3e8ef;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 4px 16px rgba(23,32,51,.06)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:18px;border:1px solid #e3e8ef;border-radius:10px}.label{font-size:12px;color:#667085}.value{font-size:24px;font-weight:bold;margin-top:7px}button{background:#2563eb;color:white;border:0;border-radius:7px;padding:12px 20px;font-weight:bold;cursor:pointer}button:disabled{opacity:.6}.notice{margin-top:15px;padding:13px;background:#fff7ed;color:#9a3412;border-radius:8px;font-size:13px}.result{display:none;margin-top:15px;padding:13px;border-radius:8px;font-size:13px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #edf0f4;font-size:13px}th{background:#f8fafc;font-size:11px}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}.bar{padding:16px}}.mini{border:0;border-radius:5px;padding:6px 8px;margin:2px;font-size:11px;cursor:pointer}.mini.safe{background:#198754;color:white}.mini.phish{background:#dc3545;color:white}.mini.promo{background:#6f42c1;color:white}.feedback-saved{font-size:11px;color:#198754} </style></head>
<body><div class="bar"><div><b>SMS Smishing Detection System</b><br><small>Model Management & Retraining</small></div><div><a href="/">Individual User</a><a href="/telecom/dashboard">Telecom Dashboard</a></div></div>
<div class="wrap"><div class="stats"><div class="stat"><div class="label">Current Live Model</div><div class="value">__VERSION__</div></div><div class="stat"><div class="label">Total Predictions</div><div class="value">__PREDICTIONS__</div></div><div class="stat"><div class="label">Corrections Collected</div><div class="value">__FEEDBACK__</div></div><div class="stat"><div class="label">Unused Corrections</div><div class="value">__UNUSED__/__THRESHOLD__</div></div></div>
<div class="card"><h2>Model Retraining</h2><p>Human-corrected predictions are used together with the original training corpus when the feedback threshold is reached.</p><button id="btn" onclick="retrain()">Retrain Model</button><div class="notice"><b>Important:</b> retraining creates a new model version for evaluation. It does not automatically replace the current live model.</div><div id="ok" class="result"></div><div id="err" class="result"></div></div>
<div class="card"><h2>Model Version History</h2><table><tr><th>Version</th><th>Training Rows</th><th>Accuracy</th><th>F1 Score</th><th>Notes</th></tr>__ROWS__</table></div></div>
<script>async function retrain(){const b=document.getElementById('btn'),ok=document.getElementById('ok'),er=document.getElementById('err');ok.style.display='none';er.style.display='none';b.disabled=true;b.textContent='Retraining...';try{const r=await fetch('/retrain',{method:'POST',headers:{'Content-Type':'application/json'}}),d=await r.json();if(!r.ok)throw new Error(d.error||'Retraining failed.');if(d.status==='skipped'){ok.textContent='Retraining skipped: '+d.reason+'.';}else if(d.status==='retrained'){ok.textContent='Retraining completed. New version '+d.version+' created. Accuracy: '+(d.accuracy*100).toFixed(2)+'%, F1: '+(d.f1_score*100).toFixed(2)+'%. Live model was not automatically replaced.';}else{ok.textContent=JSON.stringify(d);}ok.style.display='block';}catch(e){er.textContent=e.message;er.style.display='block';}finally{b.disabled=false;b.textContent='Retrain Model';}}</script></body></html>'''
    html = html.replace('__VERSION__', CURRENT_MODEL_VERSION).replace('__PREDICTIONS__', str(total_predictions)).replace('__FEEDBACK__', str(feedback_count)).replace('__UNUSED__', str(unused_feedback)).replace('__THRESHOLD__', str(RETRAIN_THRESHOLD)).replace('__ROWS__', rows)
    return html

if __name__ == "__main__":
    # debug=True is fine for local testing only; set to False before any real deployment
    app.run(debug=True, host="0.0.0.0", port=5000)