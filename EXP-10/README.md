# Glaucoma Detection from Retinal Fundus Images using a Multi-Layer Perceptron (MLP) Neural Network

## Problem Statement

Glaucoma is one of the leading causes of irreversible blindness worldwide, caused by progressive damage to the optic nerve. It is often called the “silent thief of sight” because noticeable vision loss occurs only after significant and irreversible damage has already taken place. Manual screening of retinal fundus images by an ophthalmologist is accurate but time-consuming, subjective, and difficult to scale for large populations at risk, especially where access to eye-care specialists is limited.

There is therefore a need for an automated, fast, and accessible system that can analyse a retinal fundus image and classify it as Normal or Glaucomatous with an interpretable confidence score. This project designs and implements a web-based glaucoma image classification system that performs image preprocessing, handcrafted feature extraction, and classification using a Multi-Layer Perceptron (MLP) neural network. The trained model is evaluated using standard metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## Objectives

- To collect and preprocess a labelled dataset of Normal and Glaucomatous retinal fundus images.
- To extract meaningful handcrafted features such as colour histograms, spatial gradients, and Sobel edge energy from each image.
- To design, train, and save a Multi-Layer Perceptron (MLP) classifier using the extracted features.
- To build a Flask-based backend and web frontend that allows users to upload a fundus image and get a diagnosis.
- To evaluate the trained model on an unseen test set and report Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- To persist every prediction in an SQLite database for record-keeping and review.

## Software / Hardware Required

- Operating System: Windows / Linux / macOS
- IDE: Visual Studio Code
- Language: Python 3.x
- Backend Framework: Flask
- Machine Learning: scikit-learn
- Image Processing: OpenCV, Pillow, NumPy
- Database: SQLite3
- Frontend: HTML5, CSS3, JavaScript
- Dataset: Retinal fundus images labelled as Normal and Glaucoma
- Version Control: Git

## Project Directory Structure

The project is organised as a Flask-based application with a separate dataset, trained model, and utility folders.

```text
GlaucomaClassifier/
├── app.py
├── README.md
├── requirements.txt
├── train_model.py
├── check_dataset.py
├── check_duplicates.py
├── check_repeatability.py
├── diagnose_predictions.py
├── dataset/
│   ├── train/
│   │   ├── Normal/
│   │   └── Glaucoma/
│   └── val/
│       ├── Normal/
│       └── Glaucoma/
├── backend/
│   ├── db.py
│   ├── eyecare_history.db
│   ├── models/
│   │   ├── glaucoma_model.pkl
│   │   └── evaluation/
│   ├── ml/
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   ├── evaluate_model.py
│   │   └── preprocessing_current_backup.py
│   ├── routes/
│   │   ├── predict.py
│   │   └── __init__.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   │   └── history.html
│   └── uploads/
├── training/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── train_model_v2.py
│   └── training_history.py
└── venv/
```

## Procedure

1. Collect a labelled retinal fundus-image dataset and organise it into dataset/train and dataset/val folders.
2. Preprocess each image by resizing it, converting to RGB, applying Gaussian blur, enhancing contrast using CLAHE, and normalising pixel values to [0, 1].
3. Extract a 128-dimensional handcrafted feature vector from each image using colour histogram information, gradient magnitude, orientation, and edge energy statistics.
4. Train the MLP classifier using the extracted features and save the resulting model as backend/models/glaucoma_model.pkl.
5. Launch the Flask application with app.py and open the browser UI.
6. Upload a fundus image from the frontend, run prediction, and view the predicted class and confidence score.
7. Save prediction results to SQLite for later history review and analysis.
8. Evaluate the saved model on the validation/test set using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## Implementation Code

### 1. Image Preprocessing

```python
import cv2
import numpy as np


def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    img = img.astype(np.float32) / 255.0
    return img
```

### 2. Feature Extraction

```python
import cv2
import numpy as np


def extract_features(img):
    rgb = np.clip(img * 255, 0, 255).astype(np.uint8)
    color_features = []

    for channel in range(3):
        hist = cv2.calcHist([img.astype(np.float32)], [channel], None, [32], [0, 1]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        color_features.extend(hist)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    mag_hist, _ = np.histogram(magnitude, bins=14, range=(0, 1443))
    mag_hist = mag_hist.astype(np.float32)
    mag_hist /= mag_hist.sum() + 1e-8

    angle = (np.degrees(np.arctan2(gy, gx)) + 360) % 360
    angle_hist, _ = np.histogram(angle, bins=16, range=(0, 360))
    angle_hist = angle_hist.astype(np.float32)
    angle_hist /= angle_hist.sum() + 1e-8

    edge_mean = float(np.mean(magnitude)) / 1443.0
    edge_std = float(np.std(magnitude)) / 1443.0

    features = np.concatenate([
        np.asarray(color_features, dtype=np.float32),
        mag_hist,
        angle_hist,
        np.array([edge_mean, edge_std], dtype=np.float32)
    ])
    return features
```

### 3. Model Training

```python
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np

X = np.load('processed/features.npy')
y = np.load('processed/labels.npy')

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=100,
    random_state=42,
)

model.fit(X, y)
joblib.dump({"model": model, "classes": ["Normal", "Glaucoma"], "feature_count": 128},
            'backend/models/glaucoma_model.pkl')
```

### 4. Prediction API

```python
from flask import Blueprint, request, jsonify
import joblib
from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

predict_bp = Blueprint('predict', __name__)
model = joblib.load('backend/models/glaucoma_model.pkl')
CLASSES = ['Normal', 'Glaucoma']

@predict_bp.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    path = f'backend/uploads/{file.filename}'
    file.save(path)
    img = preprocess_image(path)
    features = extract_features(img).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    pred_idx = proba.argmax()

    return jsonify({
        'prediction': CLASSES[pred_idx],
        'confidence': round(float(proba[pred_idx]) * 100, 1),
        'glaucoma_prob': round(float(proba[1]) * 100, 1),
        'normal_prob': round(float(proba[0]) * 100, 1)
    })
```

### 5. Prediction History Database

```python
import sqlite3
from datetime import datetime

conn = sqlite3.connect('backend/eyecare_history.db')
cur = conn.cursor()
cur.execute('''
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL,
        created_at TEXT
    )
''')
cur.execute(
    'INSERT INTO prediction_history (filename, prediction, confidence, created_at) VALUES (?, ?, ?, ?)',
    ('sample.png', 'Glaucoma', 92.4, datetime.now().isoformat(timespec='seconds'))
)
conn.commit()
```

## Installation and Run Instructions

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the web application

```bash
python app.py
```

Then open the browser at:

```text
http://127.0.0.1:5000
```

## Notes

- The system uses the saved model at backend/models/glaucoma_model.pkl for inference.
- Prediction records are stored in backend/eyecare_history.db.
- The application includes evaluation endpoints and prediction-history endpoints for reporting and monitoring.
- The project is designed for a glaucoma screening workflow where a clinician can review flagged retinal fundus images quickly.

## Conclusion

This project demonstrates a practical and efficient method for automated glaucoma detection using retinal fundus images. By combining image preprocessing, handcrafted feature extraction, and an MLP-based classifier, the system can support earlier screening of glaucoma and assist clinicians in making timely diagnostic decisions.
