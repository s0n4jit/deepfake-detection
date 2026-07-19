# PRD: Deepfake Image Detection — Classical Feature-Based vs CNN Comparison

**Duration:** 3 working days
**Type:** Cybersecurity + Machine Learning (summer internship project)
**Hardware:** RTX 3050 4GB VRAM (local), CPU for classical pipeline

---

## 1. Problem Statement

Deepfake images/videos are increasingly used for fraud, harassment, blackmail, and disinformation. Most published detection methods rely on deep neural networks (CNNs/RNNs), which are accurate but computationally expensive to train and deploy. An alternative line of research uses classical computer vision features (keypoint detectors + descriptors) with lightweight classifiers, trading some accuracy for speed and interpretability.

This project reproduces and evaluates both approaches on the same dataset to produce a direct, evidence-based comparison of accuracy, training cost, and explainability — rather than assuming one is strictly better.

## 2. Reference Work

Adapting the methodology from: *"Deepfake Detection"* (Santa Clara University project report, Mishra & Lan), which itself builds on **FFR-FD: Effective and Fast Detection of DeepFakes Based on Feature Point Defects** (Wang et al., 2020).

Reference pipeline: FAST (keypoint detection) → BRIEF (binary descriptors) → dlib face/region detection → per-region descriptor averaging + feature-count column → Random Forest classifier. Reported baseline: ~66% test accuracy, ~88% train accuracy (visible overfitting gap).

## 3. Goals

- **G1:** Reproduce the classical FAST+BRIEF+Random Forest pipeline on a small, balanced, frontal-face image dataset.
- **G2:** Fine-tune a small pretrained CNN (transfer learning) on the same dataset split, for a fair comparison.
- **G3:** Compare both approaches on: accuracy, training/inference time, and explainability (feature importance vs black-box).
- **G4:** Ship a simple working demo (upload an image → get a verdict) using the better-performing (or both, toggleable) model.

## 4. Non-Goals

- Video-level temporal analysis (frame sequences, LSTM/RNN) — out of scope for 3 days.
- Training a CNN from scratch — not feasible on this timeline/VRAM; transfer learning only.
- Side-profile or non-frontal face handling — explicitly out of scope (matches the reference paper's own limitation).
- Full-scale benchmark datasets (full DFDC, full Celeb-DF) — a small labeled subset is used instead.
- Production-grade deployment, authentication, or scaling — this is a proof-of-concept demo.

## 5. Users & Use Case

- **Primary user for the demo:** anyone wanting a quick sanity-check on whether a face image looks synthetically generated.
- **Primary audience for the comparison:** instructor/evaluator assessing understanding of both classical ML feature engineering and modern deep learning tradeoffs.

## 6. Data

- **Source:** small labeled subset (a few hundred images, balanced real/fake) from a public deepfake image/video-frame dataset (e.g. Kaggle deepfake dataset or extracted frames from DFDC/Celeb-DF samples).
- **Preprocessing:**
  - Frame/image → grayscale for classical pipeline; RGB 224×224 for CNN pipeline.
  - Face detection via dlib; drop images with no detectable face.
  - Restrict to frontal faces only (per reference paper's own finding that side-profile faces hurt performance).
  - Balance real vs fake classes (undersample majority class); use class weighting during training as a backup.
- **Split:** same train/test split (e.g. 70/30) used for both pipelines to keep the comparison fair.

## 7. Methodology

### 7.1 Pipeline A — Classical (FAST + BRIEF + Random Forest)
1. Detect keypoints per image using FAST.
2. Compute binary descriptors per keypoint using BRIEF.
3. Detect face + facial sub-regions (eyes, nose, mouth, etc.) using dlib's 68-point landmark predictor.
4. Group keypoints by region; average descriptors per region; append keypoint-count column.
5. Concatenate into one feature vector per image.
6. Standardize features; train a Random Forest classifier (optionally compare against a stacked Random Forest + SVM ensemble, as in the reference paper).

### 7.2 Pipeline B — CNN (Transfer Learning)
1. Load a pretrained lightweight backbone (ResNet18 or MobileNetV2).
2. Freeze base layers; replace/fine-tune the final classification layer(s) for binary output.
3. Train on the same face-cropped, balanced dataset (batch size 8–16, 224×224 input — sized for 4GB VRAM).
4. Use standard augmentation (flip, slight color jitter) to reduce overfitting on a small dataset.

### 7.3 Evaluation
- Accuracy, precision/recall, confusion matrix for both pipelines on the same held-out test set.
- Training time and inference time per image, measured directly.
- Explainability: feature importance (Random Forest) vs a basic saliency/Grad-CAM overlay (CNN), shown side by side on a few example images.

## 8. Success Criteria

- Both pipelines run end-to-end on the same dataset split without manual intervention.
- Classical pipeline result is in the same ballpark as the reference paper (~60–70% test accuracy) — this is expected, not a failure.
- CNN pipeline shows a measurable accuracy improvement over the classical pipeline (realistic target: ~75–85% test accuracy given the small dataset size and 3-day timeline).
- A working demo accepts an uploaded image and returns a verdict + confidence from at least one of the two models.
- Final write-up honestly reports the accuracy/cost/explainability tradeoff — including where the CNN wins and where the classical approach's speed/interpretability still has value.

## 9. Timeline (3 Days)

**Day 1 — Setup & Data**
- Environment setup: OpenCV, dlib (+ 68-point landmark model file), PyTorch/TensorFlow for CNN.
- Download and prepare a small balanced subset of labeled deepfake images.
- Run face detection; drop undetectable/non-frontal faces; confirm class balance.

**Day 2 — Model Training**
- Build and run Pipeline A (classical): feature extraction → Random Forest → baseline accuracy.
- Build and run Pipeline B (CNN): pretrained backbone fine-tuning on GPU → accuracy.
- Log training time and inference time for both.

**Day 3 — Comparison, Demo, Write-up**
- Generate comparison metrics (accuracy, time, explainability visuals) side by side.
- Build a simple Streamlit (or similar) upload-and-scan demo using the trained model(s).
- Write up findings: what worked, what didn't, honest limitations (dataset size, frontal-face-only scope, overfitting risk).

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| dlib installation issues eat Day 1 | Try installation early; have a fallback face detector (MTCNN) ready |
| Dataset too large to process in time | Use a small, pre-filtered subset (a few hundred images), not full DFDC |
| CNN overfits on small dataset | Use frozen base + light fine-tuning, augmentation, and early stopping |
| Class imbalance skews accuracy | Balance dataset via undersampling; use class-weighted loss/training as backup |
| Side-profile faces break pipeline | Explicitly restrict scope to frontal faces (documented non-goal) |
| Results don't beat classical baseline | Still a valid, reportable finding — document honestly rather than overclaiming |

## 11. Deliverables

- Source code: `feature_pipeline.py` (classical), `cnn_pipeline.py` (transfer learning), `demo_app.py` (Streamlit demo).
- Trained model files (`.pkl` for Random Forest, `.pt`/`.h5` for CNN).
- Comparison report/slides: accuracy tables, confusion matrices, training/inference time, explainability visuals.
- Final project write-up covering methodology, results, and honest discussion of tradeoffs and limitations.
