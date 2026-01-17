import cv2
import os
import numpy as np
import joblib

DATASET_DIR = "dataset"

faces = []
labels = []
label_map = {}

current_label = 0

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

        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

print("✅ Total images:", len(faces))
print("✅ Label map:", label_map)

model = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

model.train(faces, labels)

os.makedirs("model", exist_ok=True)
model.save("model/lbph_model.xml")
joblib.dump(label_map, "model/label_map.pkl")

print("🎉 LBPH model trained and saved successfully")
