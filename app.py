
from flask import Flask, render_template, request, send_file, abort
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import onnxruntime as ort
import numpy as np
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

# COCO class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Memory-conscious ONNX Runtime settings
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.enable_mem_pattern = False
session_options.enable_cpu_mem_arena = False

MODEL_PATH = MODEL_FOLDER / "yolo11n.onnx"

model = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = model.get_inputs()[0].name


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


# Prepare image for YOLO ONNX
def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = image.size

    target_size = 640

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas_image = Image.new(
        "RGB",
        (target_size, target_size),
        (114, 114, 114)
    )

    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2

    canvas_image.paste(image, (pad_x, pad_y))

    image_array = np.asarray(canvas_image, dtype=np.float32)
    image_array = image_array / 255.0

    input_tensor = np.transpose(image_array, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return (
        input_tensor.astype(np.float32),
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height
    )


# Non-Maximum Suppression
def nms(boxes, scores, threshold=0.45):

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]

    selected = []

    while order.size > 0:

        current = order[0]
        selected.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        width = np.maximum(0, xx2 - xx1)
        height = np.maximum(0, yy2 - yy1)

        intersection = width * height

        union = (
            areas[current] + areas[remaining] - intersection
        )

        iou = intersection / np.maximum(union, 1e-7)

        order = remaining[iou <= threshold]

    return selected


# Run ONNX object detection
def detect_objects(image, confidence):

    (
        input_tensor,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height
    ) = prepare_image(image)

    outputs = model.run(
        None,
        {input_name: input_tensor}
    )

    predictions = outputs[0][0].T

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    valid = scores >= confidence

    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(boxes) == 0:
        return []

    # Convert center boxes to corner boxes
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS separately for each class
    final_detections = []

    for class_id in np.unique(class_ids):

        class_mask = class_ids == class_id

        class_boxes = boxes[class_mask]
        class_scores_for_nms = scores[class_mask]

        keep = nms(
            class_boxes,
            class_scores_for_nms
        )

        class_indices = np.where(class_mask)[0]

        for index in keep:
            original_index = class_indices[index]

            final_detections.append((
                boxes[original_index],
                scores[original_index],
                class_ids[original_index]
            ))

    final_detections.sort(
        key=lambda item: item[1],
        reverse=True
    )

    detected_objects = []

    for box, score, class_id in final_detections[:100]:

        x1, y1, x2, y2 = box

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = int(np.clip(x1, 0, original_width))
        y1 = int(np.clip(y1, 0, original_height))
        x2 = int(np.clip(x2, 0, original_width))
        y2 = int(np.clip(y2, 0, original_height))

        if x2 <= x1 or y2 <= y1:
            continue

        detected_objects.append({
            "box": [x1, y1, x2, y2],
            "name": CLASS_NAMES[int(class_id)],
            "confidence": round(float(score) * 100, 2)
        })

    return detected_objects


# Draw detection boxes
def draw_results(image, objects, output_path):

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for obj in objects:

        x1, y1, x2, y2 = obj["box"]

        label = (
            f'{obj["name"]} '
            f'{obj["confidence"]}%'
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 120),
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 22)),
            label,
            fill=(0, 255, 120),
            font=font
        )

    image.save(output_path, format="JPEG", quality=90)


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

                confidence = request.form.get(
                    "confidence", "15"
                )

                try:
                    confidence = float(confidence) / 100
                    confidence = max(
                        0.05,
                        min(confidence, 0.90)
                    )
                except ValueError:
                    confidence = 0.15

                with Image.open(input_path) as source_image:

                    image = ImageOps.exif_transpose(
                        source_image
                    ).convert("RGB")

                detections = detect_objects(
                    image,
                    confidence
                )

                # Prepare object list for website
                for detection in detections:

                    objects.append({
                        "name": detection["name"],
                        "confidence": detection["confidence"]
                    })

                # Save annotated result image
                output_name = f"{uuid.uuid4().hex}.jpg"
                output_path = RESULT_FOLDER / output_name

                draw_results(
                    image,
                    detections,
                    output_path
                )

                result_image = f"results/{output_name}"

                # Save scan history
                scan_id = uuid.uuid4().hex

                scan_record = {
                    "id": scan_id,
                    "date": datetime.now().strftime(
                        "%d-%m-%Y %I:%M %p"
                    ),
                    "image": result_image,
                    "total_objects": len(objects),
                    "objects": objects
                }

                history = load_history()
                history.insert(0, scan_record)
                save_history(history)

            except Exception as e:

                print("Error:", repr(e))
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

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=A4
    )

    width, height = A4

    pdf.setTitle("Vision AI Detection Report")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(
        50, height - 60,
        "Vision AI Pro"
    )

    pdf.setFont("Helvetica", 12)

    pdf.drawString(
        50, height - 90,
        "Object Detection Report"
    )

    pdf.drawString(
        50, height - 120,
        f"Scan Date: {scan['date']}"
    )

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

        pdf.drawString(
            50, y,
            str(obj["name"])
        )

        pdf.drawString(
            300, y,
            f"{obj['confidence']}%"
        )

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
