from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import json  # for reading answer keys
import pillow_heif

# Third-party libraries (install if needed)
from flask_tailwind import TailwindCSS
from imutils.perspective import four_point_transform

# Helper functions (replace with your actual implementation)
def adjust_perspective(img_np):
    # Implement your perspective correction logic here
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            warped = four_point_transform(img_np, pts)
            return warped
    return img_np

def infer_answer_area_average_size(bubble_coords):
    # Implement your logic to infer answer area based on average bubble size
    if not bubble_coords:
        return None, None, None, None
    x_coords = [x for x, _, _, _ in bubble_coords]
    y_coords = [y for _, y, _, _ in bubble_coords]
    w_coords = [w for _, _, w, _ in bubble_coords]
    h_coords = [h for _, _, _, h in bubble_coords]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)+ max(w_coords)
    y_max = max(y_coords)+ max(h_coords)

    return x_min, y_min, x_max, y_max

def process_student_answers(columns, model_name, answer_key_path):
    # Simulate processing with a dummy function
    all_extracted_answers = ["A", "B", "C", "D"] * len(columns)
    scores = {"test_code_1": 85.50}
    return all_extracted_answers, scores

def handle_uploaded_image(uploaded_file):
    try:
        if uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(uploaded_file)
        elif uploaded_file.filename.lower().endswith('.heic'):
            heif_file = pillow_heif.read_heif(uploaded_file)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data
            )
        else:
            return None
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


# Initialize Flask app
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Auto-reload templates on changes
app.config["SECRET_KEY"] = "your_secret_key"  # Required for TailwindCSS
app.config['TAILWIND_CONFIG_FILE'] = './tailwind.config.js'

# Initialize TailwindCSS
tailwind = TailwindCSS(app)

# Answer key storage (replace with database or file-based storage)
answer_keys = {}

# Load answer keys from JSON file (replace with your logic)
def load_answer_keys():
    global answer_keys
    try:
        with open("answer_keys.json", "r") as f:
            answer_keys = json.load(f)
    except FileNotFoundError:
        print("answer_keys.json not found. Starting with empty answer keys.")
        answer_keys = {}

load_answer_keys()  # Load answer keys on startup


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files["image"]

        if not uploaded_file:
            return render_template("index.html", error="No file uploaded")

        try:
            # Handle image conversion and processing
            img_np = handle_uploaded_image(uploaded_file)

            if img_np is None:
                return render_template("index.html", error="Failed to process image.")

            # Preprocess image (call your functions)
            img_np_preprocessed = adjust_perspective(img_np.copy())

            # Detect bubbles and extract answer area (simulate)
            bubble_coords = [(100, 50, 40, 30), (150, 70, 35, 25)]
            x_min, y_min, x_max, y_max = infer_answer_area_average_size(bubble_coords)

            if x_min is None:
                return render_template("index.html", error="No bubbles detected.")

            # Extract answer area (simulate)
            answer_area = img_np_preprocessed[y_min:y_max, x_min:x_max]

            # Divide into columns (simulate)
            warped_height, warped_width = answer_area.shape[:2]
            column_width = warped_width // 4
            columns = [
                answer_area[0:warped_height, 0:column_width],
                answer_area[0:warped_height, column_width:2 * column_width],
                answer_area[0:warped_height, 2 * column_width:3 * column_width],
                answer_area[0:warped_height, 3 * column_width:warped_width],
            ]

            # Simulate processing with Gemini
            model_name = request.form["model_name"]
            all_extracted_answers, scores = process_student_answers(
                columns, model_name, "answer_keys.json"
            )

            return render_template(
                "result.html",
                image=img_np_preprocessed,
                bubble_coords=bubble_coords,
                answer_area=answer_area,
                columns=columns,
                extracted_answers=all_extracted_answers,
                scores=scores,
            )

        except Exception as e:
            print(f"Error processing: {e}")
            return render_template("index.html", error=f"An error occurred: {e}")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)