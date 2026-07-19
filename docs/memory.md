# Project Memory: Deepfake Detection (Classical vs CNN)

**Purpose of this file:** a single source of truth for where the project stands. Update it at the end of every work session — before you stop, not after you forget. If you (or an AI assistant) pick this project back up later, read this file first instead of re-explaining the whole project from scratch.

---

## Project Summary

Internship project (cybersecurity + ML), 3-day build. Detects deepfake face images two ways — a classical FAST+BRIEF+Random Forest pipeline (reproducing a Santa Clara University project report, itself based on the FFR-FD paper) and a fine-tuned CNN (ResNet18/MobileNetV2 via transfer learning) — then compares them honestly on accuracy, speed, and explainability. Shipped as a web app: FastAPI backend + plain HTML/JS frontend, one Docker image, deployed on Render.

**Reference docs already written** (all in `/mnt/user-data/outputs/` unless moved into the repo):
- `PRD.md` — goals, non-goals, methodology, success criteria, timeline, risks
- `architecture.md` — tech stack, request flow, API routes, folder structure, Docker/Render notes
- `rules.md` — what to achieve, what to avoid, error handling table, AI-assistance boundaries, accuracy-honesty rules
- `phases.md` — the 8-phase execution plan with exit checkpoints
- `design.md` — frontend visual direction, tokens, layout, copy voice
- `challenges_and_scaling.md` — details on package compilation conflicts, memory bottlenecks, and VPS scaling roadmap

---

## Key Decisions Log

Record decisions here as they're made, with a one-line reason — so later-you doesn't re-litigate something already settled.

| Decision | Reason |
|---|---|
| Deepfake image detection (not APK/scam/network projects) | Had a research paper to anchor methodology; classical CV approach is genuinely 3-day feasible |
| Do both classical (FAST+BRIEF+RF) AND CNN, compare them | Stronger project than either alone; RTX 3050 makes CNN fine-tuning fast enough to fit |
| FastAPI backend, plain HTML/JS frontend, single Docker image | One language, one deploy target, no separate frontend build pipeline to manage in 3 days |
| Deploy on Render via Docker | User's choice; CPU-only on Render — GPU work stays local, only exported model files get deployed |
| Frontal faces only, no video/audio, no multi-face handling | Matches the reference paper's own documented limitation; keeps scope realistic |
| Target ~90%+ accuracy, but don't force it | Reference paper's own classical model hit ~66%; realistic CNN target is 75–85% on this timeline. 90%+ is a stretch goal, not achieved via test-set leakage or cherry-picking (see `rules.md` §2) |
| dlib for face/landmark detection | Successfully installed dlib via a precompiled wheel for Python 3.13 on Windows; matches the reference paper's exact requirements |
| Dataset subset: DeepFake00 to DeepFake04 | Provides a manageable subset (~2000 images before filtering) for 3-day development |
| Configure default port 8080 & APP_URL self-ping | Meets target environment specifications and prevents Free tier spin-down |

---

## Current Status

**Phase:** Phase 5 complete. Proceeding to Phase 6.

**Last updated:** 2026-07-20

### Done
- [x] Project idea selected and scoped (deepfake detection, classical vs CNN comparison)
- [x] PRD written
- [x] Architecture/tech stack decided and documented
- [x] Rules/guardrails documented
- [x] Phase-by-phase execution plan written
- [x] Frontend design direction written
- [x] Phase 0 — Environment & repo setup
- [x] Phase 1 — Dataset preparation
- [x] Phase 2 — Classical pipeline (FAST+BRIEF+RF)
- [x] Phase 3 — CNN pipeline (transfer learning)
- [x] Phase 4 — Comparison & explainability
- [x] Phase 5 — Web app (backend + frontend)

### Not Started
- [ ] Phase 6 — Dockerize & deploy
- [ ] Phase 7 — Report & submission


---

## Open Questions / Blockers

Track anything unresolved here so it doesn't get silently forgotten between sessions.

- None.

---

## How to Update This File

At the end of each work session:
1. Move finished checklist items from "Not Started" to "Done" under **Current Status**.
2. Update the **Last updated** line.
3. Add any new entries to **Key Decisions Log** if a real choice was made (not just "tried X, it worked" — only decisions that future-you needs to not re-debate).
4. Add anything unresolved to **Open Questions / Blockers**, and remove it once resolved.
5. If actual results come in (accuracy numbers, training times), add a **Results So Far** section below this line and keep it current — don't let numbers go stale in the other docs while this one has the latest truth.

---

## Results So Far

### Dataset Stats (Phase 1)
- Total images checked: 7230
- Dropped (no face): 2514
- Dropped (non-frontal): 1722
- Valid REAL images: 403
- Valid FAKE images: 2591
- Balanced Dataset: 806 images total (403 REAL / 403 FAKE)
- Split: 564 train images (70%), 242 test images (30%)

### Classical Pipeline (Phase 2)
- Model: FAST + BRIEF + Region Grouping + Random Forest (`random_forest_v1.pkl`)
- Train Accuracy: 97.87%
- Test Accuracy: 66.94%
- Test Precision: 75.47%
- Test Recall: 59.70%
- Test F1-Score: 66.67%
- Confusion Matrix:
  - True Negatives (Real as Real): 82
  - False Positives (Real as Fake): 26
  - False Negatives (Fake as Real): 54
  - True Positives (Fake as Fake): 80
- Training Time (Random Forest): 0.30 seconds
- Total Pipeline Extraction + Training Time: 7.87 seconds
- Average Inference Time per image: 6.31 ms

### CNN Pipeline (Phase 3)
- Model: Fine-tuned ResNet18 fc layer (`cnn_model_v1.pt`)
- Train Accuracy: 76.60%
- Test Accuracy: 68.60%
- Test Precision: 67.68%
- Test Recall: 82.84%
- Test F1-Score: 74.50%
- Confusion Matrix:
  - True Negatives (Real as Real): 55
  - False Positives (Real as Fake): 53
  - False Negatives (Fake as Real): 23
  - True Positives (Fake as Fake): 111
- Training Time: 174.65 seconds (CPU)
- Average Inference Time per image: 30.78 ms

### Comparison & Explainability (Phase 4)
- **Error overlap in test set (242 images):**
  - Both correct (Real): 46 occurrences
  - Both correct (Fake): 71 occurrences
  - Both wrong (Real as Fake): 17 occurrences
  - Both wrong (Fake as Real): 14 occurrences
  - CNN correct, RF wrong: 49 occurrences
  - RF correct, CNN wrong: 45 occurrences
- **Classical feature importance (top regions):**
  - nose: 22.38%
  - whole_face: 21.56%
  - right_eyebrow: 14.12%
  - mouth: 11.24%
  - right_eye: 9.95%
  - left_eye: 7.79%
  - left_eyebrow: 7.26%
  - inner_mouth: 5.69%
- **CNN explainability:** Generated and saved saliency overlay heatmaps under `docs/explainability/`.

---

## Future Roadmap

Key directions for future development:
1. **Audio & Video Support:** Expand detection coverage to support video temporal checking (e.g. frame sequence analysis using LSTMs or 3D-CNNs) and audio deepfake detection.
2. **CNN Accuracy Refinement:** Fine-tune the backbone models further (e.g. unfreezing deeper layers of ResNet18/EfficientNet) with larger datasets on GPU to increase the overall detection accuracy.
3. **UI/UX Polish:** Enhance the clinical reporting interface, add visualization layers for region importance, and implement responsive improvements for mobile devices.





