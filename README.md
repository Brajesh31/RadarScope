Here is your final, **ready-to-use README** in proper `.md` (Markdown) format — **you can directly paste this into your `README.md` file**:


# 🛡️ MSTAR Tank Detection using Scattering Transform + CNN

A lightweight, web-based radar target classification system that detects and classifies military tanks from **SAR (Synthetic Aperture Radar)** images using deep learning techniques. This project was developed at **LRDE, DRDO** as part of real-world defense research exposure by **Irfan Sulfi**, **Adam Nahan**, **Ajmal Shan**, and **Mazin Muneer**.

It combines the power of **Scattering2D Transform** for feature extraction and a trained **CNN model**, all wrapped in a clean **Flask web interface** for user interaction.



## 🗂️ Project Structure


MSTAR-Tank-Detection/
├── app.py                # Main Flask application
├── radarmodel.h5         # Pre-trained CNN model
├── label.pkl             # Label encoder for class predictions
├── templates/
│   └── index.html        # Web UI for image upload and result display
├── uploads/              # Temporary folder to store uploaded images



## ⚙️ Key Features

- ✅ Upload grayscale SAR images (`.jpg`, `.png`, `.bmp`)
- 🔄 Automatic preprocessing and classification of radar targets
- 🧠 Feature extraction using **Scattering2D Transform** (via Kymatio)
- 🛠️ Prediction via trained **Convolutional Neural Network**
- 🌐 Flask-powered web interface with simple upload & display workflow


## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/irfansulfi11/MSTAR-Tank-Detection.git
cd MSTAR-Tank-Detection


### 2️⃣ Set Up the Environment

Ensure Python ≥ 3.8 is installed. Then run:


pip install -r requirements.txt


If `requirements.txt` is not available:


pip install flask tensorflow kymatio opencv-python joblib numpy

### 3️⃣ Run the App

python app.py


Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)


## 🧠 Model Overview

* **Model Type**: Convolutional Neural Network (CNN)
* **Input**: 64x64 grayscale SAR images
* **Feature Extraction**: Scattering2D via Kymatio
* **Model File**: `radarmodel.h5`
* **Label Mapping**: `label.pkl`

---

## 🖼️ Testing with Sample Images

1. Place SAR samples in the `uploads/` folder
2. Upload via the web UI
3. Ensure images are grayscale and in `.jpg`, `.jpeg`, `.png`, or `.bmp` format
4. View predicted results in real-time

---

## 📌 Notes

* The app automatically deletes uploaded images after prediction
* Input images must be grayscale for accurate results
* Developed as part of defense AI research at **LRDE – DRDO**

---

## 🤝 Acknowledgements

Special thanks to our mentors and the research team at **Electronics & Radar Development Establishment (LRDE), DRDO** for their support and guidance.

---

📌 *Star the repo or fork it to build your own radar target classifier!*

