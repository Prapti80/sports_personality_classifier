import cv2
import os
import numpy as np
import joblib

DATASET_DIR = "dataset"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

print("🔁 Training LBPH model...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (90, 90))
        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer.train(faces, labels)

# Save model
recognizer.save(os.path.join(MODEL_DIR, "lbph_model.xml"))
joblib.dump(label_map, os.path.join(MODEL_DIR, "label_map.pkl"))

print("✅ LBPH model trained successfully")
print("📌 Labels:", label_map)
