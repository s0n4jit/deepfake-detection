# Architecture: Deepfake Detection Web App

**Goal:** a single deployable web app (Docker → Render) that lets a user upload a face image and get a real/fake verdict, backed by the two models from the PRD (classical FAST+BRIEF+Random Forest, and a fine-tuned CNN).

---

## 1. Tech Stack (recommended)

| Layer | Choice | Why |
|---|---|---|
| Backend | **FastAPI** (Python) | Your models (sklearn, PyTorch/TF, OpenCV, dlib) are all Python — same language for ML + API means no serialization bridge, no second runtime. FastAPI is fast to write, has automatic docs, and async support for handling uploads. |
| Model serving | In-process, loaded at startup | With only 2 models and modest traffic (student project demo), a separate model-serving layer (Triton, TorchServe) is unnecessary overhead. Load both models once when the app starts; keep them in memory. |
| Frontend | **Plain HTML + CSS + vanilla JS** (served by FastAPI via Jinja2/static files) | No build step, no Node toolchain, no separate deploy target — one container, one Dockerfile, deploys cleanly to Render. A React/Vite frontend is nicer but adds a second build pipeline you don't have time for in 3 days; only go there if you have spare time on Day 3. |
| Classical ML | OpenCV (`cv2`), `dlib`, scikit-learn, `joblib` | Matches the reference paper's pipeline exactly. |
| CNN | PyTorch + `torchvision` (pretrained ResNet18/MobileNetV2) | Lighter to containerize than TensorFlow for a single small model; smaller image size. |
| Containerization | **Docker**, single image | Render deploys directly from a Dockerfile — one image running FastAPI (serving both API + static frontend) is the simplest path. |
| Hosting | **Render** (Web Service, Docker runtime) | You already picked this — free/low tier is enough for a demo; supports Docker deploys directly from a GitHub repo. |
| Process server | `uvicorn` (with `--workers 1` for a small demo) | Standard ASGI server for FastAPI. |

**Note on GPU on Render:** Render's standard web services are CPU-only. Your CNN training happens locally on your RTX 3050 — you train once, export the model file (`.pt`), and the *deployed* app only runs inference (forward pass), which is fine on CPU for single-image requests. Don't plan on training or GPU inference happening on Render itself.

---

## 2. High-Level Architecture

```
                        ┌─────────────────────────────┐
                        │        User's Browser        │
                        │  (upload image, view result) │
                        └───────────────┬───────────────┘
                                        │ HTTPS
                                        ▼
                        ┌─────────────────────────────┐
                        │     Render Web Service        │
                        │   (single Docker container)   │
                        │                               │
                        │  ┌─────────────────────────┐  │
                        │  │      FastAPI App         │  │
                        │  │                          │  │
                        │  │  Static frontend (/, /ui)│  │
                        │  │  API routes (/api/*)      │  │
                        │  │                          │  │
                        │  │  ┌────────────────────┐  │  │
                        │  │  │ Preprocessing        │  │  │
                        │  │  │ (face detect/crop)   │  │  │
                        │  │  └─────────┬──────────┘  │  │
                        │  │            ▼              │  │
                        │  │  ┌────────────────────┐  │  │
                        │  │  │ Model A: FAST+BRIEF  │  │  │
                        │  │  │ + Random Forest      │  │  │
                        │  │  │ (loaded from .pkl)   │  │  │
                        │  │  └────────────────────┘  │  │
                        │  │  ┌────────────────────┐  │  │
                        │  │  │ Model B: CNN         │  │  │
                        │  │  │ (loaded from .pt)    │  │  │
                        │  │  └────────────────────┘  │  │
                        │  └─────────────────────────┘  │
                        └─────────────────────────────┘
```

---

## 3. Request Flow (upload → verdict)

1. User opens the site → served the static HTML/JS page (drag-and-drop or file-picker upload widget).
2. User selects an image and picks which model to use (dropdown: "Classical", "CNN", or "Both") → JS sends a `POST /api/scan` with the file (multipart/form-data) and the chosen mode.
3. FastAPI route receives the file into memory (no need to write to disk for a single image).
4. **Preprocessing step:**
   - Decode image bytes → OpenCV/PIL image.
   - Run dlib face detector. If no face found → return a clear "no face detected" response immediately (don't run models on garbage input).
   - Crop to the face region, resize as needed per model (grayscale for classical, 224×224 RGB for CNN).
5. **Inference step:**
   - If "Classical": run FAST → BRIEF → region grouping → feature vector → `random_forest.predict_proba()`.
   - If "CNN": run the fine-tuned model's forward pass → softmax probability.
   - If "Both": run both, return both scores.
6. **Response:** JSON with verdict (`real`/`fake`), confidence score(s), and which model(s) were used. Optionally include which facial region contributed most (for the classical model — you already have feature importance from training) as a simple explainability note.
7. Frontend renders the result: verdict badge, confidence bar, model used.

---

## 4. API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves the frontend (upload page) |
| `POST` | `/api/scan` | Accepts an image file + `model` param (`classical` \| `cnn` \| `both`) → returns verdict JSON |
| `GET` | `/api/health` | Simple health check (used by Render, and useful for you to confirm models loaded correctly at startup) |
| `GET` | `/api/models` | (optional) returns metadata about loaded models — version, training accuracy, etc., for display on the site |

Example response from `/api/scan`:
```json
{
  "face_detected": true,
  "results": {
    "classical": { "verdict": "fake", "confidence": 0.71 },
    "cnn": { "verdict": "fake", "confidence": 0.88 }
  }
}
```

---

## 5. Folder Structure

```
deepfake-detector/
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── README.md
│
├── app/
│   ├── main.py                 # FastAPI app entrypoint, route definitions
│   ├── config.py               # paths, constants (model file locations, thresholds)
│   │
│   ├── models/                 # trained model artifacts (checked in or downloaded at build time)
│   │   ├── random_forest.pkl
│   │   ├── cnn_model.pt
│   │   └── shape_predictor_68_face_landmarks.dat
│   │
│   ├── pipelines/
│   │   ├── preprocessing.py    # face detection/cropping, shared by both models
│   │   ├── classical_pipeline.py   # FAST + BRIEF + region grouping + RF inference
│   │   └── cnn_pipeline.py         # CNN loading + inference
│   │
│   ├── schemas.py               # Pydantic request/response models
│   └── static/                  # frontend
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── training/                    # NOT part of the deployed image — your local training code
│   ├── prepare_dataset.py
│   ├── train_classical.py       # reproduces the paper's compute_metric.py / testing_models.py
│   ├── train_cnn.py
│   └── notebooks/                # exploratory work, comparison charts
│
└── tests/
    ├── test_api.py
    └── test_pipelines.py
```

**Why `training/` is separate and excluded from the Docker image:** training needs your GPU, the full dataset, and heavier libraries (full PyTorch with CUDA) — none of that belongs in the small, CPU-only deployed container. The Docker image only needs the *already-trained* model files plus lightweight inference code. Keep `training/` out of `.dockerignore`-excluded paths so your image stays small and builds fast on Render.

---

## 6. Dockerfile (shape, not final code)

```dockerfile
FROM python:3.11-slim

# system deps needed for dlib/opencv build
RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Heads-up on `dlib`:** it compiles from source unless you use a prebuilt wheel, and this can slow down your Render build significantly or even time out on free tier. Two mitigations:
- Use `dlib-binary` (a prebuilt wheel package) instead of `dlib` in `requirements.txt` if compatible with your Python version, or
- Swap dlib for `mediapipe` or `MTCNN` for face detection in the *deployed* app only (keep dlib for local training/landmark work if you need the 68-point regions there) — this avoids a slow/fragile build step on Render entirely.

---

## 7. Render Deployment Notes

- Connect your GitHub repo to Render as a **Web Service**, runtime **Docker**.
- Render auto-builds from your `Dockerfile` on push.
- Set the port Render expects to match your `EXPOSE`/`uvicorn` port (Render also passes a `$PORT` env var — safer to bind to `0.0.0.0:$PORT` in your CMD rather than hardcoding 8000).
- Free tier services on Render spin down when idle and cold-start on the next request — expect a delay (10-30s) on the first request after inactivity; worth mentioning in your demo/report so it doesn't look broken.
- Keep model files reasonably small — a Random Forest `.pkl` and a small CNN `.pt` (ResNet18/MobileNetV2 fine-tuned head) should both be well under typical free-tier size/memory limits.

---

## 8. Frontend Flow (simple version)

```
index.html
 ├── <input type="file"> or drag-and-drop zone
 ├── <select> model choice (Classical / CNN / Both)
 ├── "Scan" button → fetch('/api/scan', { method: 'POST', body: formData })
 └── Result panel:
      - Verdict badge (Real / Fake, color-coded)
      - Confidence bar(s)
      - "No face detected" state handled explicitly
```

No framework needed for this scope — `fetch` + `FormData` + a bit of DOM manipulation covers the whole flow in under 100 lines of JS.

---

## 9. Summary of Key Decisions

- **One backend, one language (Python/FastAPI)** — avoids a second stack just for a demo.
- **One Docker image, both API and frontend** — simplest possible Render deploy.
- **Training kept fully separate from the deployed app** — GPU/heavy deps never enter the container.
- **CPU-only inference in production** — both models are small enough that this is fine for single-image requests.
- **Face-detection library choice (dlib vs mediapipe/MTCNN) is a build-time tradeoff** — decide early on Day 1 so it doesn't cost you Render build time later.
