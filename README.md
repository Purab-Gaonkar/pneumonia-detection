# Pneumonia Detection AI

An advanced deep learning application for detecting pneumonia from chest X-ray images. This project utilizes an ensemble of state-of-the-art Convolutional Neural Networks (CNNs) including DenseNet121, ResNet50V2, and MobileNetV2. It also features an Out-of-Distribution (OOD) detector to verify if the uploaded image is a valid X-ray.

## Features
- **Deep Learning Ensemble**: Combines multiple CNN architectures for high-accuracy predictions.
- **OOD Detection**: Uses an Isolation Forest algorithm to reject invalid/random non-X-ray images.
- **Premium Web Interface**: A sleek, responsive frontend built with HTML, CSS (Glassmorphism design), and JavaScript.
- **FastAPI Backend**: Asynchronous and robust API serving the TensorFlow models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Purab-Gaonkar/pneumonia-detection
   cd pneumonia_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Important**: The dataset is not included in this repository. Place your dataset directory relative to the project or adjust the `DATASET_DIR` variable in `train.py`.

## Usage

### 1. Training the Models
Before running the application, you must train the ensemble and OOD models. A GPU is highly recommended.
```bash
python train.py
```
This will generate `pneumonia_ensemble_model.h5` and `ood_isolation_forest.pkl` in the `models/` directory.

### 2. Starting the Backend Server
Once the models are trained, start the FastAPI server:
```bash
python app.py
```

### 3. Accessing the Web Interface
Open your web browser and navigate to:
```
http://localhost:8000
```
Drag and drop a chest X-ray image into the interface to see the AI's diagnosis.

## Disclaimer
This AI tool is for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
