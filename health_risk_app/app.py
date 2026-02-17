from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3, os, pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)
app.secret_key = "healthrisksecret"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model", "health_risk_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb"))


# DB Connection
def init_db():
    conn = sqlite3.connect("database.db")
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


@app.route("/", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if user is None:
            error = "No account found with this email. Please sign up first."
        elif not check_password_hash(user[3], password):
            error = "Incorrect password. Please try again."
        else:
            session["user"] = user
            return redirect("/dashboard")

    return render_template("login.html", error=error)


from sqlite3 import IntegrityError

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )
            db.commit()
            return redirect("/")

        except IntegrityError:
            error = "An account with this email already exists. Please login."

    return render_template("signup.html", error=error)



@app.route("/dashboard")

def dashboard():
    db = get_db()
    history = db.execute(
        "SELECT id, risk, timestamp FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (session["user"][0],)
    ).fetchall()
    return render_template("dashboard.html", user=session["user"], history=history)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        rr = float(request.form["respiratory_rate"])
        spo2 = float(request.form["oxygen_saturation"])
        o2_scale = float(request.form["o2_scale"])
        bp = float(request.form["systolic_bp"])
        hr = float(request.form["heart_rate"])
        temp = float(request.form["temperature"])
        on_o2 = int(request.form["on_oxygen"])
        consciousness = request.form["consciousness"]

        # One-hot encoding
        c = p = u = v = 0
        if consciousness == "C": c = 1
        elif consciousness == "P": p = 1
        elif consciousness == "U": u = 1
        elif consciousness == "V": v = 1

        features = [
            rr, spo2, o2_scale, bp, hr, temp, on_o2,
            c, p, u, v
        ]

        scaled = scaler.transform([features])

        # Prediction
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]
        classes = model.classes_

        # ✅ SAFE probability mapping
        probability_map = dict(zip(classes, probs))

        # Normalize missing labels
        low_prob = probability_map.get("Low", 0)
        moderate_prob = probability_map.get("Moderate", 0)
        high_prob = probability_map.get("High", 0)

        # Final risk label
        if isinstance(pred, str):
            risk_label = pred
        else:
            risk_map = {0: "Low", 1: "Moderate", 2: "High"}
            risk_label = risk_map.get(pred, "Unknown")
        label_normalization = {
        "Normal": "Low",
        "normal": "Low",
        "LOW": "Low",
        "Moderate": "Moderate",
        "MODERATE": "Moderate",
        "High": "High",
        "HIGH": "High"
    }
        risk_label = label_normalization.get(risk_label, risk_label)
        precaution_map = {
    "Low": (
        "Maintain a healthy lifestyle with regular physical activity and a balanced, nutritious diet. "
        "Ensure adequate hydration and follow a consistent sleep routine to support overall physiological well-being. "
        "Continue periodic health check-ups to monitor vital parameters and promptly address any emerging symptoms."
    ),

    "Moderate": (
        "Monitor vital signs regularly and adopt lifestyle modifications to reduce potential health risks. "
        "Limit exposure to physical or mental stress and ensure sufficient rest and recovery. "
        "Consult a healthcare professional for further evaluation if symptoms persist or worsen over time."
    ),

    "High": (
        "Seek immediate medical consultation for comprehensive evaluation and appropriate intervention. "
        "Avoid strenuous physical activity and closely monitor vital parameters until professional guidance is obtained. "
        "Follow all medical recommendations strictly and ensure timely follow-up to prevent potential complications."
    )
}


       
        photo_path = None

        photo = request.files.get("photo")
        if photo and photo.filename:
            upload_dir = os.path.join(app.root_path, "static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            filename = secure_filename(photo.filename)
            photo_path = f"uploads/{filename}"     # IMPORTANT
            photo.save(os.path.join(upload_dir, filename))



        session["result"] = {
        "report_id": str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),

        "patient": {
            "name": request.form.get("patient_name", "N/A"),
            "age": request.form.get("age", "N/A"),
            "photo": photo_path # optional
        },

        "risk": risk_label,
        "precaution": precaution_map.get(risk_label),
        "probs": {
            "Low": round(low_prob * 100, 2),
            "Moderate": round(moderate_prob * 100, 2),
            "High": round(high_prob * 100, 2)
        }
        }
        import json

        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, risk, timestamp, report_data) VALUES (?, ?, ?, ?)",
            (
                session["user"][0],
                session["result"]["risk"],
                session["result"]["timestamp"],
                json.dumps(session["result"])
            )
        )
        db.commit()

    

        return redirect("/result")

    return render_template("predict.html")


@app.route("/result")
def result():
    if "result" not in session:
        return redirect("/predict")
    return render_template("result.html", report=session["result"])




from reportlab.platypus import Image
import io
import base64

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import base64, io, os

from reportlab.lib.colors import lightgrey

def add_watermark(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica-Bold", 40)
    canvas.setFillColor(lightgrey)

    # Position + rotation
    canvas.translate(300, 400)
    canvas.rotate(45)

    canvas.drawCentredString(
        0, 0,
        "FOR PREDICTION ONLY"
    )

    canvas.setFont("Helvetica", 18)
    canvas.drawCentredString(
        0, -50,
        "NOT A MEDICAL DIAGNOSIS"
    )

    canvas.restoreState()


@app.route("/final-download")
def final_download():
    if "result" not in session:
        return redirect("/predict")

    import os
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = session["result"]

    os.makedirs("reports", exist_ok=True)
    file_path = "reports/clinical_health_report.pdf"

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title",
        fontSize=18,
        alignment=1,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "section",
        fontSize=14,
        spaceBefore=14,
        spaceAfter=8
    )

    content = []

    # ===== TITLE =====
    content.append(Paragraph("Clinical Health Risk Report", title_style))
    content.append(Paragraph(f"<b>Report ID:</b> {report['report_id']}", styles["Normal"]))
    content.append(Paragraph(f"<b>Date:</b> {report['timestamp']}", styles["Normal"]))
    content.append(Spacer(1, 16))

    # ===== PATIENT DETAILS (PHOTO + INFO SIDE BY SIDE) =====
    content.append(Paragraph("Patient Details", section_style))

    patient_info = [
        Paragraph(f"<b>Name:</b> {report['patient']['name']}", styles["Normal"]),
        Paragraph(f"<b>Age:</b> {report['patient']['age']}", styles["Normal"])
    ]

    if report["patient"]["photo"]:
        img = Image(
            os.path.join("static", report["patient"]["photo"]),
            1.2 * inch,
            1.2 * inch
        )
        table = Table([[img, patient_info]])
        table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ]))
        content.append(table)
    else:
        content.extend(patient_info)

    content.append(Spacer(1, 16))

    # ===== RISK SUMMARY =====
    content.append(Paragraph("Risk Summary", section_style))
    content.append(Paragraph(
        f"<b>Risk Level:</b> {report['risk']}",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    # ===== PRECAUTIONS =====
    content.append(Paragraph("Recommended Precautions", section_style))
    content.append(Paragraph(report["precaution"], styles["Normal"]))
    content.append(Spacer(1, 20))

    # ===== CHARTS (SERVER-GENERATED, MATCHING PREVIEW DATA) =====
    probs = report["probs"]

    # Bar chart
    plt.figure(figsize=(5, 3))
    plt.bar(["Low", "Moderate", "High"],
            [probs["Low"], probs["Moderate"], probs["High"]],
            color=["#3b82f6", "#f59e0b", "#ef4444"])
    plt.ylim(0, 100)
    plt.title("Risk Probability (%)")
    plt.tight_layout()
    plt.savefig("reports/bar.png")
    plt.close()

    content.append(Paragraph("Risk Analysis", section_style))
    content.append(Image("reports/bar.png", width=5 * inch, height=3 * inch))

    # ===== DISCLAIMER =====
    content.append(Spacer(1, 24))
    content.append(Paragraph(
        "<i>This report is generated using predictive analytics and is not a medical diagnosis.</i>",
        styles["Italic"]
    ))

    doc.build(
    content,
    onFirstPage=add_watermark,
    onLaterPages=add_watermark
)

    return send_file(file_path, as_attachment=True)




@app.route("/report_preview")
def report_preview():
    if "result" not in session:
        return redirect("/predict")
    print(session["result"])


    return render_template(
        "report_preview.html",
        report=session["result"]
    )

@app.route("/view-report/<int:report_id>")
def view_report(report_id):
    import json

    db = get_db()
    row = db.execute(
        "SELECT report_data FROM predictions WHERE id = ? AND user_id = ?",
        (report_id, session["user"][0])
    ).fetchone()

    if not row:
        return redirect("/dashboard")

    report = json.loads(row[0])

    return render_template(
        "report_preview.html",
        report=report
    )



if __name__ == "__main__":
        app.run(debug=True)





