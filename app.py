from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LBPH model
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(MODEL_DIR, "lbph_model.xml"))

label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))

# Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"})

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image"})

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"})

    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (90, 90))

    label, confidence = model.predict(face)

    return jsonify({
        "prediction": label_map[label],
        "confidence": f"{100 - confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
