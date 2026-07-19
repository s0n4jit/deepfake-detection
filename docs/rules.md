# Project Rules: Deepfake Detection (Classical vs CNN)

This document sets the working rules for the project — what to achieve, what to avoid, how errors must be handled, and the boundaries for using AI tools (Claude/ChatGPT/Copilot) while building it. Read this before writing code, not after.

---

## 1. What We're Trying to Achieve

- A working, end-to-end pipeline for **both** models (classical FAST+BRIEF+Random Forest, and fine-tuned CNN) trained on the same dataset split.
- A deployed web app (Render, Docker) that accepts an image upload and returns a verdict.
- An honest, evidence-backed comparison of the two approaches (accuracy, speed, explainability) in the final report.
- A codebase and report clean enough that you can explain every line and every number to your instructor without hand-waving.

## 2. On the 90%+ Accuracy Target — Read This First

**Be honest with yourself about this number before you build anything.** The reference paper's own classical pipeline reports ~66% test accuracy. A fine-tuned CNN on a small (few-hundred-image) dataset in 3 days is realistically in the **75–85% range** — that was the honest estimate we landed on earlier, and it still stands.

Rules to follow here:
- **Do not tune, filter, or resample the test set to hit 90%.** If your test accuracy is inflated because you leaked data between train/test, removed "hard" examples, or evaluated on a test set too similar to training data, the number is meaningless and will fall apart under any follow-up question.
- **Do not report training accuracy as if it were test accuracy.** The reference paper itself shows a large train/test gap (88% vs 66%) — expect something similar, and report both numbers, not just the flattering one.
- **If you legitimately hit 90%+ on a proper held-out test set, that's great — verify it isn't due to an overly easy/small/imbalanced test split before you trust it.**
- **If you don't hit 90%, that is a valid, reportable outcome.** The value of this project is the honest comparison and the methodology, not a specific number. An instructor who understands ML will trust a well-documented 82% far more than a suspicious, unexplained 96%.
- If you want to *push toward* higher accuracy, the legitimate levers are: more/better data (not just more epochs), class balancing, data augmentation, and only unfreezing more CNN layers once you confirm you're not overfitting — not post-hoc test-set cherry-picking.

## 3. What To Do

- **Use the same train/test split** for both models — this is the only way the comparison is meaningful.
- **Report both models' confusion matrices**, not just accuracy — precision/recall matter more for a real detector (a false "real" verdict on an actual deepfake is worse than the reverse).
- **Log training time and inference time** for both pipelines from the start — don't try to reconstruct this after the fact.
- **Version your model files** (e.g. `random_forest_v1.pkl`, `cnn_v1.pt`) so you can roll back if a later training run performs worse.
- **Keep `training/` code and dependencies out of the deployed Docker image** — the deployed app only needs inference code and the already-trained model files.
- **Handle "no face detected" as a first-class case**, not an exception that crashes the request.
- **Cite the reference paper properly** in your report (Wang et al. FFR-FD, plus the Santa Clara project report you adapted from) — you're building on their work, not claiming the original method as your own idea.

## 4. What To Avoid

- **Don't train or run inference on Render using a GPU** — Render's standard web service is CPU-only; all GPU work happens locally and only the exported model file goes into the container.
- **Don't commit the full dataset to your Git repo** — it's large, not yours to redistribute in most cases, and bloats the repo. Commit only a small sample (if licensing allows) or a script that downloads it.
- **Don't skip face-detection validation** — an image with no detectable face should never silently reach the model as if it were a valid input.
- **Don't leave classical and CNN preprocessing inconsistent** — if the classical pipeline crops faces one way and the CNN pipeline crops them differently, your "comparison" isn't actually comparing the same input, and the result is invalid.
- **Don't hardcode secrets, API keys, or file paths specific to your machine** — use environment variables/config, since this needs to run in a Docker container on Render, not just on your laptop.
- **Don't let scope creep in** — video-level analysis, audio deepfakes, multi-face images, and side-profile faces are explicitly out of scope per the PRD. If you find yourself building any of these, stop and check the PRD's non-goals section first.
- **Don't claim state-of-the-art performance** in your write-up. Your framing should match the reference paper's own honest framing: a lightweight/interpretable alternative with a known accuracy tradeoff, not a breakthrough.

## 5. Error Handling Requirements

The app must handle these cases explicitly, with a clear response — never a raw stack trace or a silent wrong answer:

| Case | Required behavior |
|---|---|
| Uploaded file is not a valid image | Reject with a clear "invalid file type" message before any processing starts |
| No face detected in the image | Return a distinct "no face detected" result — do not run either model on it |
| Multiple faces detected | Explicitly out of scope (per PRD) — pick a documented behavior (e.g. use the largest/most central face, or reject with a "multiple faces not supported" message) and be consistent |
| Side-profile / non-frontal face | Either detect and reject with a clear message, or document as a known limitation if detection isn't reliable |
| File too large | Enforce a max upload size (e.g. 5–10MB) and reject larger files with a clear message, not a server crash |
| Model file missing/failed to load at startup | App should fail fast and loudly at startup (visible in Render logs), not silently serve broken predictions |
| Classical and CNN models disagree | Don't silently pick one — if returning "both," show both verdicts and let the result speak for itself |
| Render cold start / slow first request | Document this as expected behavior in your report/demo notes, not a bug |

General rule: **fail loudly and specifically, not silently and generically.** A user (or your instructor) should always know *why* something didn't work.

## 6. Data & Ethics Rules

- Only use publicly available, appropriately licensed deepfake datasets for training (per the PRD's listed sources) — don't scrape or use content you don't have rights to use for this purpose.
- Don't use real people's images beyond what's in the established public datasets, and don't add your own real photos of identifiable people (classmates, family) as "real" or "fake" training examples.
- Be clear in your report that this tool is a student proof-of-concept, not a certified forensic tool — it should not be presented as something that can definitively prove an image is fake in any legal, journalistic, or accusatory context.

## 7. Boundaries for AI-Assisted Work (Claude, ChatGPT, Copilot, etc.)

Since you'll likely use AI tools to help write parts of this:

- **Verify any accuracy numbers, benchmark claims, or dataset statistics an AI tool gives you** — don't put a number in your report that you haven't actually produced yourself by running the code.
- **Don't let an AI tool's code go into the project unread.** You need to be able to explain every part of the pipeline to your instructor — if you can't explain a chunk of code, don't ship it as-is; ask for it to be simplified or explain it to yourself first.
- **Don't ask an AI tool to fabricate a research citation, dataset source, or benchmark comparison.** If it's not sure, it should say so — treat a confident-sounding but unverifiable claim as a reason to check, not a reason to trust.
- **AI tools are good for boilerplate (Dockerfile shape, FastAPI route skeletons, standard preprocessing code) — use them there.** They are not a substitute for you understanding *why* the classical pipeline underperforms the CNN, since that understanding is the actual point of the project.
- **If an AI tool suggests a way to "boost" accuracy that involves test-set leakage, cherry-picked splits, or reporting the better of multiple runs without disclosure — reject it.** That's the exact failure mode flagged in Section 2.
- Final report and explanations should be in your own words. Using AI to draft/organize your report is fine; submitting AI-written analysis you don't understand or agree with is not.

## 8. Definition of Done

- [ ] Both pipelines trained on the same data split, both model files exported.
- [ ] Confusion matrix + accuracy + training/inference time recorded for both.
- [ ] Web app deployed on Render via Docker, handles all cases in Section 5.
- [ ] No face / invalid file / oversized file cases tested manually at least once each.
- [ ] Report written with honest numbers, proper citation of the reference paper, and a clear discussion of the accuracy/cost/explainability tradeoff — including if the 90% target wasn't hit, and why.
