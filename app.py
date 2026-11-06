
from flask import Flask, render_template, request, send_file, abort
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import uuid
import json
import io

app = Flask(__name__)

# Project folders
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_FOLDER = BASE_DIR / "models"

HISTORY_FILE = BASE_DIR / "history.json"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load AI Model
model = YOLO(str(MODEL_FOLDER / "yolo11n.pt"))


# Load saved detection history
def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


# Save detection history
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


# Main Dashboard
@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    objects = []
    error = None
    scan_id = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image."

        elif Path(file.filename).suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            error = "Invalid image format."

        else:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            input_path = UPLOAD_FOLDER / unique_name

            try:
                file.save(input_path)

                # Get confidence value from HTML slider
                confidence = request.form.get("confidence", "15")

                try:
                    confidence = float(confidence) / 100
                    confidence = max(0.05, min(confidence, 0.90))
                except ValueError:
                    confidence = 0.15

                # AI Object Detection
                predictions = model.predict(
                    source=str(input_path),
                    conf=confidence,
                    verbose=False
                )

                result = predictions[0]

                # Save detected image
                output_name = f"{uuid.uuid4().hex}.jpg"
                output_path = RESULT_FOLDER / output_name

                result.save(filename=str(output_path))

                result_image = f"results/{output_name}"

                # Extract detected objects
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    object_confidence = float(box.conf[0].item())

                    objects.append({
                        "name": result.names[class_id],
                        "confidence": round(object_confidence * 100, 2)
                    })

                # Save scan in history
                scan_id = uuid.uuid4().hex

                scan_record = {
                    "id": scan_id,
                    "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
                    "image": result_image,
                    "total_objects": len(objects),
                    "objects": objects
                }

                history = load_history()
                history.insert(0, scan_record)
                save_history(history)

            except Exception as e:
                print("Error:", e)
                error = "Image processing failed."

    history = load_history()

    return render_template(
        "index.html",
        result_image=result_image,
        objects=objects,
        error=error,
        scan_id=scan_id,
        history=history
    )


# Detection History page
@app.route("/history")
def detection_history():

    history = load_history()

    return render_template(
        "history.html",
        history=history
    )


# Download PDF Report
@app.route("/download-report/<scan_id>")
def download_report(scan_id):

    history = load_history()

    scan = next(
        (item for item in history if item["id"] == scan_id),
        None
    )

    if scan is None:
        abort(404)

    pdf_buffer = io.BytesIO()

    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

    width, height = A4

    pdf.setTitle("Vision AI Detection Report")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 60, "Vision AI Pro")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 90, "Object Detection Report")
    pdf.drawString(50, height - 120, f"Scan Date: {scan['date']}")
    pdf.drawString(
        50, height - 145,
        f"Total Detected Objects: {scan['total_objects']}"
    )

    y = height - 190

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Object Name")
    pdf.drawString(300, y, "Confidence")

    y -= 25

    pdf.setFont("Helvetica", 11)

    for obj in scan["objects"]:

        if y < 60:
            pdf.showPage()
            y = height - 60
            pdf.setFont("Helvetica", 11)

        pdf.drawString(50, y, str(obj["name"]))
        pdf.drawString(300, y, f"{obj['confidence']}%")

        y -= 22

    pdf.save()

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="VisionAI_Report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)
