from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3, os, pickle, json, uuid, io, base64
import numpy as np
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ===== Matplotlib (Render-safe) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== ReportLab =====
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import lightgrey


# ==============================
# APP CONFIG
# ==============================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "healthrisksecret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ==============================
# DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "database.db"))
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        risk TEXT,
        timestamp TEXT,
        report_data TEXT
    )
    """)

    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "database.db"))
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/health")
def health():
    return "OK", 200


init_db()

# ==============================
# ML MODEL (LAZY LOAD 🔥)
# ==============================
model = None
scaler = None

def load_model():
    global model, scaler
    if model is None or scaler is None:
        model = pickle.load(open(os.path.join(BASE_DIR, "model", "health_risk_model.pkl"), "rb"))
        scaler = pickle.load(open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb"))
    return model, scaler

# ==============================
# AUTH ROUTES
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if user is None:
            error = "No account found with this email."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect password."
        else:
            session["user"] = dict(user)
            return redirect("/dashboard")

    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None

    if request.method == "POST":
        try:
            db = get_db()
            db.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (
                    request.form["name"],
                    request.form["email"],
                    generate_password_hash(request.form["password"])
                )
            )
            db.commit()
            return redirect("/")
        except sqlite3.IntegrityError:
            error = "Email already exists."

    return render_template("signup.html", error=error)

# ==============================
# DASHBOARD
# ==============================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    history = db.execute(
        "SELECT id, risk, timestamp FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (session["user"]["id"],)
    ).fetchall()

    return render_template("dashboard.html", user=session["user"], history=history)

# ==============================
# PREDICTION
# ==============================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        model, scaler = load_model()

        rr = float(request.form["respiratory_rate"])
        spo2 = float(request.form["oxygen_saturation"])
        o2_scale = float(request.form["o2_scale"])
        bp = float(request.form["systolic_bp"])
        hr = float(request.form["heart_rate"])
        temp = float(request.form["temperature"])
        on_o2 = int(request.form["on_oxygen"])
        consciousness = request.form["consciousness"]

        c = p = u = v = 0
        if consciousness == "C": c = 1
        elif consciousness == "P": p = 1
        elif consciousness == "U": u = 1
        elif consciousness == "V": v = 1

        features = [[
            rr, spo2, o2_scale, bp, hr, temp, on_o2,
            c, p, u, v
        ]]

        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]
        classes = model.classes_

        prob_map = dict(zip(classes, probs))
        risk_label = str(pred)

        normalization = {
            "Normal": "Low", "LOW": "Low",
            "Moderate": "Moderate", "MODERATE": "Moderate",
            "High": "High", "HIGH": "High"
        }
        risk_label = normalization.get(risk_label, risk_label)

        precaution_map = {
            "Low": "Maintain a healthy lifestyle.",
            "Moderate": "Monitor vitals and consult doctor if needed.",
            "High": "Seek immediate medical attention."
        }

        photo_path = None
        photo = request.files.get("photo")
        if photo and photo.filename:
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(UPLOAD_FOLDER, filename))
            photo_path = f"uploads/{filename}"

        session["result"] = {
            "report_id": str(uuid.uuid4())[:8].upper(),
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "patient": {
                "name": request.form.get("patient_name", "N/A"),
                "age": request.form.get("age", "N/A"),
                "photo": photo_path
            },
            "risk": risk_label,
            "precaution": precaution_map[risk_label],
            "probs": {
                "Low": round(prob_map.get("Low", 0) * 100, 2),
                "Moderate": round(prob_map.get("Moderate", 0) * 100, 2),
                "High": round(prob_map.get("High", 0) * 100, 2)
            }
        }

        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, risk, timestamp, report_data) VALUES (?, ?, ?, ?)",
            (
                session["user"]["id"],
                risk_label,
                session["result"]["timestamp"],
                json.dumps(session["result"])
            )
        )
        db.commit()

        return redirect("/result")

    return render_template("predict.html")

# ==============================
# RESULT
# ==============================
@app.route("/result")
def result():
    if "result" not in session:
        return redirect("/predict")
    return render_template("result.html", report=session["result"])

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


