from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    message_text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    source_type = db.Column(db.String(30), nullable=False)  # 'individual' or 'telecom_batch'
    model_version = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    corrected_label = db.Column(db.String(20), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    used_in_retrain = db.Column(db.Boolean, default=False)


class ModelVersion(db.Model):
    __tablename__ = 'model_versions'
    id = db.Column(db.Integer, primary_key=True)
    version_tag = db.Column(db.String(50), nullable=False, unique=True)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    training_rows = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    notes = db.Column(db.Text)