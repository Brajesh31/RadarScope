import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from kymatio.numpy import Scattering2D

# Parameters
IMAGE_SIZE = 64
TEST_DIR = r"D:\PROJECTS\MSTAR Detection\sample"  # Change to your actual test image directory

# Load trained model and label encoder
model = load_model("radarmodel.h5")
label_encoder = joblib.load("label.pkl")

# Initialize scattering transform
scattering = Scattering2D(J=2, shape=(IMAGE_SIZE, IMAGE_SIZE))

# Load and process test images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    feat = scattering(img).flatten()
    return feat

# Predict classes for test images
for fname in os.listdir(TEST_DIR):
    fpath = os.path.join(TEST_DIR, fname)
    feat = preprocess_image(fpath)
    if feat is None:
        continue
    feat = feat.reshape(1, -1)
    prediction = model.predict(feat)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"{fname} ➜ Predicted: {predicted_class[0]}")
