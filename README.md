# 🛡️ MSTAR Tank Detection using Scattering Transform + CNN
  A web-based radar target classification system that detects and classifies military tanks from SAR (Synthetic Aperture Radar) images using deep learning. Built as our first project at LRDE, DRDO by Irfan Sulfi,   Adam Nahan, Ajmal Shan, and Mazin Muneer, this project combines the power of the Scattering Transform and a trained CNN model with a Flask-powered web interface.

## 📁 Project Structure
```
MSTAR-Tank-Detection/
├── app.py                               # Flask web application
├── radarmodel.h5                        # Trained CNN model for tank classification
├── label.pkl                            # Label encoder for class mapping
├── templates/
│   └── index.html                       # Frontend UI for uploading radar images
├── uploads/                             # Directory where user images are temporarily saved
```
## ⚙️ Features
- Upload a grayscale SAR image (e.g., .jpg, .png, .bmp)
- Automatically processes and classifies radar targets
- Uses Scattering2D Transform for feature extraction (via Kymatio)
- Predicts tank types using a pre-trained CNN model
- Clean Flask-based web interface for uploading and viewing results

## 🚀 How to Run
### 1. Clone the Repository
```
git clone https://github.com/irfansulfi11/MSTAR-Tank-Detection.git
cd MSTAR-Tank-Detection
```
### 2. Set Up the Environment
Ensure Python ≥3.8 is installed. Then install the required packages:
```
pip install -r requirements.txt
# If requirements.txt is not provided, install manually:
pip install flask tensorflow kymatio opencv-python joblib numpy
```
### 3. Run the App
```
python app.py
```
Open your browser and navigate to:
http://127.0.0.1:5000

## 🧠 Model Info
- Model Type: Convolutional Neural Network (CNN)
- Preprocessing: Scattering2D Transform on SAR images (64x64 grayscale)
- Model File: radarmodel.h5
- Label Mapping: label.pkl

## 🖼️ Sample Images
SAR image samples can be placed in the ``` uploads/``` folder for quick testing.

## 🛠️ Notes
- Make sure your input images are grayscale and in a supported format:``` .jpg, .jpeg, .png, .bmp```
- All temporary files are automatically removed after prediction
-Project developed as part of real-world defense research exposure at LRDE – DRDO

