import cv2
import os
import shutil
import numpy as np

DATASET_PATH = "dataset"
BAD_FOLDER = "bad_images"
DRY_RUN = False   # IMPORTANT: True = nothing is deleted
MIN_FACE_SIZE = 80
BLUR_THRESHOLD = 100

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

os.makedirs(BAD_FOLDER, exist_ok=True)

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def is_bad_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return True, "cannot_read"

    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return True, "low_resolution"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return True, "no_face"

    if len(faces) > 1:
        return True, "multiple_faces"

    x, y, fw, fh = faces[0]
    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return True, "face_too_small"

    face_roi = gray[y:y+fh, x:x+fw]
    eyes = eye_cascade.detectMultiScale(face_roi)

    if len(eyes) < 2:
        return True, "eyes_not_detected"

    if is_blurry(img):
        return True, "blurry"

    return False, "good"

print("\n🔍 Starting dataset check...\n")

stats = {}

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Checking folder: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        bad, reason = is_bad_image(img_path)

        stats[reason] = stats.get(reason, 0) + 1

        if bad:
            print(f"❌ {img_name} --> {reason}")

print("\n📊 Summary:")
for k, v in stats.items():
    print(f"{k}: {v}")
