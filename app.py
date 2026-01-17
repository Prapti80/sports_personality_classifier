from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "lbph_model.xml")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = cv2.face.LBPHFaceRecognizer_create()
model.read(MODEL_PATH)

# Label map (FIXED ORDER)
label_map = {
    0: "lionel_messi",
    1: "maria_sharapova",
    2: "roger_federer",
    3: "serena_williams",
    4: "virat_kohli"
}


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))  # LBPH prefers 200x200

    return face


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        face = preprocess_image(path)
        if face is None:
            return jsonify({"error": "No face detected"})

        label, confidence = model.predict(face)

        return jsonify({
            "prediction": label_map[label],
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": "Server error"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
