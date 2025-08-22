<p align="center">
  <a href="https://github.com/Brajesh31/MSTAR-Tank-Detection">
    <img src="https://raw.githubusercontent.com/Brajesh31/asset/main/mstar-tank-banner.png" alt="MSTAR Tank Detection Banner">
  </a>
</p>

<div align="center">

# 🛡️ MSTAR Tank Detection using Scattering Transform + CNN 📡

**A lightweight, web-based radar target classification system that detects and classifies military tanks from SAR (Synthetic Aperture Radar) images using deep learning techniques.**

</div>

<p align="center">
  <img src="https://img.shields.io/github/stars/Brajesh31/MSTAR-Tank-Detection?style=for-the-badge&color=gold" alt="Stars">
  &nbsp;
  <img src="https://img.shields.io/github/last-commit/Brajesh31/MSTAR-Tank-Detection?style=for-the-badge&color=blue" alt="Last Commit">
  &nbsp;
  <img src="https://img.shields.io/github/license/Brajesh31/MSTAR-Tank-Detection?style=for-the-badge&color=green" alt="License">
</p>

---

## ## ✨ Project Overview

This project provides a web-based system to classify military targets from Synthetic Aperture Radar (SAR) imagery. It combines the powerful feature extraction capabilities of the **Scattering2D Transform** with a trained **Convolutional Neural Network (CNN)** for accurate classification.

The entire system is wrapped in a clean **Flask** web interface, allowing users to easily upload a SAR image and receive an instant prediction. This implementation was developed by **Brajesh** and is based on research concepts from defense AI applications at LRDE, DRDO.

---
## ## ⭐ Core Features

* **SAR Image Upload**: Supports grayscale `.jpg`, `.png`, and `.bmp` formats.
* **Automated Processing**: Automatically preprocesses images for the model.
* **Advanced Feature Extraction**: Uses the **Scattering2D Transform** (via the Kymatio library) to create robust feature sets.
* **Deep Learning Prediction**: Classifies targets using a pre-trained CNN model.
* **Web-Based UI**: A simple, clean interface for uploading images and viewing results.

---
## ## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Backend** | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/) |
| **Deep Learning** | [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) |
| **Feature Extraction**| **Kymatio (Scattering Transforms)** |
| **Data Handling** | [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) **Joblib** |

---
## ## 🧠 How It Works

The classification pipeline is straightforward:
1.  **Upload**: The user uploads a grayscale SAR image via the Flask web UI.
2.  **Preprocess**: The image is loaded using OpenCV and resized to the model's required input shape (64x64).
3.  **Feature Extraction**: The **Scattering2D Transform** is applied to the image to extract stable and informative features.
4.  **Prediction**: The extracted features are fed into the pre-trained CNN model (`radarmodel.h5`) for classification.
5.  **Display**: The predicted class label is displayed to the user on the web page.

---
## ## 📁 Project Structure
