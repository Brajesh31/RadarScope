import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from kymatio.numpy import Scattering2D
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Parameters
IMAGE_SIZE = 64
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model and label encoder
model = load_model("radarmodel.h5")
label_encoder = joblib.load("label.pkl")

# Initialize scattering transform
scattering = Scattering2D(J=2, shape=(IMAGE_SIZE, IMAGE_SIZE))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    feat = scattering(img).flatten()
    return feat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        feat = preprocess_image(file_path)
        if feat is None:
            os.remove(file_path)  # Clean up
            return jsonify({'error': 'Could not process image'})
        
        feat = feat.reshape(1, -1)
        prediction = model.predict(feat)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        os.remove(file_path)  # Clean up
        return jsonify({'filename': filename, 'prediction': predicted_class})
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)