# Deepfake Image Detection: Classical Feature-Based vs. CNN

This project evaluates and compares two distinct machine learning approaches for deepfake frontal-face image detection: a lightweight, classical feature-engineered pipeline (FAST keypoints + BRIEF descriptors + dlib region grouping + Random Forest) and a modern deep learning convolutional neural network (Transfer Learning on PyTorch ResNet18).

It is built as a complete, single-container web application: **FastAPI backend** + **vanilla HTML/CSS/JS frontend**, ready for deployment to platforms like Render via Docker.

---

## 🚀 Features & Architecture

- **Dual Model Inference:** Evaluate uploaded frontal face images against both models simultaneously or individually.
- **Shared Preprocessing:** Automated face detection and cropping using `dlib` with standardized 10% margins and 256x256 resizing.
- **Explainable Metrics:**
  - **Classical:** Displays the relative feature importance of facial sub-regions (eyes, nose, mouth, eyebrows, whole face).
  - **CNN:** Generates gradient-based saliency overlay heatmaps highlighting what the neural network focuses on.
- **Camera Focus Bounding Box:** Visual reticle overlays drawn dynamically on the frontend canvas.
- **Clean UI:** Cool, clinical design tailored for forensic reporting (no shadows, IBM Plex Mono technical metrics layout).

---

## 📊 Evaluation & Tradeoffs (Results So Far)

The models were evaluated on the identical test split (242 images total, balanced 50/50 real/fake):

| Metric | Classical (FAST+BRIEF+RF) | CNN (ResNet18 Transfer) |
| :--- | :--- | :--- |
| **Train Accuracy** | 97.87% (visible overfitting) | 76.60% |
| **Test Accuracy** | 66.94% | 68.60% |
| **Precision** | 75.47% | 67.68% |
| **Recall (Detect Fakes)** | 59.70% | 82.84% (better coverage) |
| **Training Time** | 0.30 seconds | 174.65 seconds (on CPU) |
| **Avg. Inference Speed** | **6.31 ms / image** | 30.78 ms / image |

### Explainability Findings
- **Classical Bounding Box Importance:** The nose region (22.38%) and the whole face (21.56%) were the strongest indicators of manipulation in the Random Forest tree splits, followed by the right eyebrow (14.12%).
- **Errors Complementarity:** Out of the 242 test images, the CNN correctly predicted 49 fakes that the Classical model missed, and the Classical model predicted 45 fakes that the CNN missed. This strongly justifies running both pipelines.

---

## 🛠️ Local Setup

### 1. Requirements
- Python 3.11 or 3.12 (Recommended for `dlib` wheel compatibility)
- CMake & C++ Build Tools (if compiling dlib from source, though precompiled wheels are recommended on Windows)

### 2. Installation
1. Clone the repository.
2. Initialize and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Upgrade pip and install standard packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
   *Note: If installing dlib on Windows fails, download a matching precompiled wheel (e.g. from `z-mahmud22/Dlib_Windows_Python3.x`) and run `pip install <wheel_path>`.*
4. Ensure the face predictor model file is extracted under:
   `app/models/shape_predictor_68_face_landmarks.dat`

### 3. Running Preprocessing & Training
If you want to re-run dataset processing or model training locally:
- **Prepare dataset (crops, balances, splits):** `python training/prepare_dataset.py`
- **Train Classical pipeline:** `python training/train_classical.py`
- **Train CNN pipeline:** `python training/train_cnn.py`
- **Generate Saliency Heatmaps:** `python training/generate_explainability.py`

### 4. Running the Web App locally
Start the ASGI Uvicorn server:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 🐳 Dockerization & Deployment

To build and run the container locally:
```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

Deploy directly on Render as a **Web Service** with the **Docker** runtime. Render will parse the `Dockerfile` automatically, pull the model artifacts inside `app/models/`, and bind to the correct `$PORT`.
