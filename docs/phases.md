# Project Phases: Deepfake Detection (Classical vs CNN)

Each phase has a goal, tasks, and an exit checkpoint — don't move to the next phase until the checkpoint is genuinely met, not just "mostly done." This keeps the 3-day timeline honest.

---

## Phase 0 — Environment & Repo Setup
**Time budget:** 1–2 hours (start of Day 1)

**Tasks**
- Create the repo with the folder structure from `architecture.md`.
- Set up a Python virtual environment.
- Install core deps: OpenCV, scikit-learn, PyTorch, FastAPI, uvicorn.
- Decide face-detection library now (dlib vs mediapipe/MTCNN) — per `architecture.md`, this affects Docker build time later. Install and test it in isolation (detect a face in one sample image) before moving on.
- Initialize Git, first commit (empty structure + `.gitignore` for datasets/model files/venv).

**Exit checkpoint**
- [ ] `import cv2, dlib (or mediapipe), sklearn, torch, fastapi` all run with no errors.
- [ ] Face detector successfully finds a face in at least one test image.

---

## Phase 1 — Dataset Preparation
**Time budget:** rest of Day 1

**Tasks**
- Download a small subset of a labeled deepfake dataset (per PRD Section 6).
- Run face detection across the subset; drop images with no detectable face.
- Filter out clearly non-frontal/side-profile faces (per PRD non-goals).
- Check class balance (real vs fake counts); undersample the majority class or note the imbalance for class-weighting later.
- Create the fixed train/test split now — **save the split (e.g. a list of file paths per set) to disk** so both pipelines use the identical split.

**Exit checkpoint**
- [ ] A balanced (or documented-imbalanced) set of frontal-face images, labeled.
- [ ] One saved train/test split file that both Phase 2 and Phase 3 will load — not regenerated separately in each.
- [ ] Rough count logged: N real / N fake / N dropped (no face or side-profile).

---

## Phase 2 — Classical Pipeline (FAST + BRIEF + Random Forest)
**Time budget:** first half of Day 2

**Tasks**
- Implement feature extraction: FAST keypoints → BRIEF descriptors → dlib region grouping → averaged descriptor + keypoint-count column (per the reference paper's method).
- Standardize features; train Random Forest on the Phase 1 train split.
- Evaluate on the test split: accuracy, precision/recall, confusion matrix.
- Log training time.
- Export the trained model (`random_forest_v1.pkl`).
- (Optional, if time allows) try the stacked Random Forest + SVM ensemble from the reference paper for comparison.

**Exit checkpoint**
- [ ] Classical pipeline runs end-to-end on the saved split without manual fixes.
- [ ] Accuracy/confusion matrix recorded (expect ~60–70%, per Rules Section 2 — this is fine).
- [ ] Model file exported and reloadable (test: load it fresh in a new script and predict on one sample).

---

## Phase 3 — CNN Pipeline (Transfer Learning)
**Time budget:** second half of Day 2

**Tasks**
- Load a pretrained backbone (ResNet18 or MobileNetV2); freeze base layers.
- Replace/fine-tune the final classification layer(s) for binary output.
- Train on the **same** Phase 1 train split, batch size 8–16, 224×224 input (sized for 4GB VRAM).
- Apply basic augmentation (flip, slight color jitter) to reduce overfitting.
- Evaluate on the same test split: accuracy, precision/recall, confusion matrix.
- Log training time and GPU memory usage (watch for OOM — reduce batch size if needed).
- Export the trained model (`cnn_v1.pt`).

**Exit checkpoint**
- [ ] CNN trains without OOM errors on the RTX 3050.
- [ ] Accuracy/confusion matrix recorded (target ~75–85%, per Rules Section 2).
- [ ] Model file exported and reloadable for inference alone (no training code needed to run it).
- [ ] Confirmed: CNN was evaluated on the identical test split as the classical model (not a different sample).

---

## Phase 4 — Comparison & Explainability
**Time budget:** first part of Day 3, ~1–2 hours

**Tasks**
- Build a single comparison table: accuracy, precision, recall, training time, inference time per image, for both models.
- Classical model: extract and note feature importance (which regions/features mattered most — you already get this from the Random Forest).
- CNN: generate a basic saliency map or Grad-CAM overlay on a few example images.
- Pick 3–5 example images (some correctly classified, some misclassified by one or both models) to use as concrete illustrations in the report/demo.

**Exit checkpoint**
- [ ] One comparison table, both models, same metrics, side by side.
- [ ] At least one explainability visual per model.
- [ ] A short honest note on where each model wins/loses (don't skip this even if results aren't flattering).

---

## Phase 5 — Web App (Backend + Frontend)
**Time budget:** middle of Day 3, ~2–3 hours

**Tasks**
- Implement FastAPI routes (`/`, `/api/scan`, `/api/health`, `/api/models`) per `architecture.md`.
- Wire in both models, loaded once at startup.
- Implement the shared preprocessing step (face detect/crop) used by both pipelines.
- Implement the error-handling table from `rules.md` Section 5 (bad file, no face, oversized upload, etc.) — test each case manually.
- Build the plain HTML/CSS/JS frontend: upload widget, model-choice dropdown, result display.

**Exit checkpoint**
- [ ] Local run (`uvicorn app.main:app`) works end-to-end: upload → verdict displayed in browser.
- [ ] Each error case in `rules.md` Section 5 manually tested and confirmed to fail gracefully, not crash.
- [ ] No face detected case confirmed to short-circuit before reaching either model.

---

## Phase 6 — Dockerize & Deploy
**Time budget:** end of Day 3, ~1–2 hours

**Tasks**
- Write the Dockerfile per `architecture.md` (watch the dlib build-time warning — swap to a prebuilt wheel or mediapipe/MTCNN now if it's slowing the build).
- Build and run the image locally first — confirm it works identically to the local (non-Docker) run before pushing.
- Push repo to GitHub; connect to Render as a Docker Web Service.
- Set the app to bind to `0.0.0.0:$PORT` (Render's expected port convention).
- Deploy; test the live URL with a real upload.
- Note expected cold-start delay in your demo notes (per `rules.md`).

**Exit checkpoint**
- [ ] Docker image builds successfully and runs locally.
- [ ] Deployed Render URL accepts an upload and returns a correct-looking verdict.
- [ ] Both models confirmed working on the deployed version (not just locally) — test both dropdown options.

---

## Phase 7 — Report & Submission
**Time budget:** final stretch of Day 3

**Tasks**
- Write up the final report: methodology, dataset, both pipelines, comparison table, explainability visuals, honest discussion of results (including if 90% wasn't hit, and why).
- Cite the reference paper(s) properly.
- Include the live Render URL and a few screenshots/example runs in case the live demo has a cold-start hiccup during presentation.
- Final check against `rules.md` Section 8 (Definition of Done) before submitting.

**Exit checkpoint**
- [ ] Report covers all Definition of Done items from `rules.md`.
- [ ] Live demo tested one final time, end to end, right before submission/presentation.

---

## If You're Running Behind

Cut in this order (least damaging to most):
1. Drop the optional stacked Random Forest + SVM ensemble (Phase 2) — plain Random Forest is enough.
2. Simplify the frontend to a single form with no model dropdown — just run "both" always.
3. Skip Grad-CAM for the CNN (Phase 4) — feature importance from the classical model alone is still a valid explainability angle.
4. If Docker/Render deployment (Phase 6) is genuinely at risk, demo locally via `uvicorn` and be upfront about deployment being in progress — a working local app with an honest report beats a broken or rushed deployment.

**Never cut:** the same train/test split across both models (Phase 1), and honest reporting of real numbers (Rules Section 2) — these are the two things that make the comparison legitimate.
