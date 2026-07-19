# Project Report: Deepfake Image Detection — Classical vs. CNN

**Date:** July 20, 2026  
**Internship deliverable:** 3-Day Cybersecurity + Machine Learning Deliverable  
**Author:** s0n4jit  
**Supervisor:** Department of Computer Science & Engineering  

---

## 1. Problem Statement & Reference Work
Deepfake images are increasingly generated using sophisticated GANs/diffusion architectures, making face forgery a significant threat. Standard detection methods employ large deep neural networks (CNNs/Transformers) which offer high accuracy but are computationally expensive.

This project reproduces and evaluates a lightweight, interpretable alternative described in **"FFR-FD: Effective and Fast Detection of DeepFakes Based on Feature Point Defects" (Wang et al., 2020)** and compares it directly against a modern deep learning backbone (**ResNet18**) on the same dataset split. 

---

## 2. Dataset Preparation & Preprocessing
To ensure a fair comparison, the dataset was processed, balanced, and split identically for both pipelines:
- **Source Subset:** Folders `DeepFake00` through `DeepFake04` (~7,230 raw files).
- **Face Extraction:** Bounding box detection using `dlib` (largest face selected for multi-face instances).
- **Frontal Face Filter:** Calculated symmetry ratio using landmark coordinates (distance from nose tip to leftmost cheek vs. rightmost cheek). Non-frontal faces (ratio < 0.55) were dropped.
- **Class Balancing:** Balanced via undersampling of the majority class (`FAKE` generated frames).
- **Dataset Split:** 70% Training set (564 images), 30% Testing set (242 images).

**Preprocessing Stats:**
- Total files checked: 7,230
- Dropped (no face): 2,514
- Dropped (non-frontal): 1,722
- Balanced Dataset: 806 images total (403 REAL / 403 FAKE)

---

## 3. Methodology & Pipelines

### Pipeline A: Classical Feature-Based
1. Run **FAST** corner detector on the 256x256 cropped grayscale face image.
2. Compute **BRIEF** descriptors (32 bytes per keypoint).
3. Detect the 68 facial landmarks. Map keypoints to 7 facial sub-regions (mouth, inner mouth, right eyebrow, left eyebrow, right eye, left eye, nose) and the whole face (region 8).
4. Average descriptors per region and append keypoint counts.
5. Standardize features and train a **Random Forest Classifier** (100 trees).

### Pipeline B: CNN Transfer Learning
1. Load a pre-trained **ResNet18** backbone.
2. Freeze all feature extraction layers.
3. Replace the final fully connected layer (`fc`) with a linear output mapping to 2 classes (REAL / FAKE).
4. Apply Random Horizontal Flips and Color Jitter to training images.
5. Train on CPU/GPU for 10 epochs using the Adam optimizer.

---

## 4. Evaluation Metrics & Comparison

| Metric | Classical (FAST+BRIEF+RF) | CNN (ResNet18 Transfer) |
| :--- | :--- | :--- |
| **Train Accuracy** | 97.87% | 76.60% |
| **Test Accuracy** | 66.94% | **68.60%** |
| **Precision (Test)** | **75.47%** | 67.68% |
| **Recall (Test)** | 59.70% | **82.84%** |
| **F1-Score** | 66.67% | **74.50%** |
| **Training Time** | **0.30 seconds** | 174.65 seconds (on CPU) |
| **Average Inference Latency** | **6.31 ms / image** | 30.78 ms / image |

### Confusion Matrices
- **Classical Model:**
  ```
  [[82  26]   <-- Real (82 Correct, 26 False Positives)
   [54  80]]  <-- Fake (54 False Negatives, 80 Correct)
  ```
- **CNN Model:**
  ```
  [[ 55  53]  <-- Real (55 Correct, 53 False Positives)
   [ 23 111]] <-- Fake (23 False Negatives, 111 Correct)
  ```

---

## 5. Explainability & Analysis

### Classical Feature Importance
The Random Forest weights reveal the relative contribution of each region to the final classification:
1. **Nose region:** 22.38%
2. **Whole face region:** 21.56%
3. **Right eyebrow:** 14.12%
4. **Mouth:** 11.24%
5. **Right eye:** 9.95%
6. **Left eye:** 7.79%
7. **Left eyebrow:** 7.26%
8. **Inner mouth:** 5.69%

### CNN Saliency Heatmaps
Using backpropagation gradients, saliency overlay heatmaps were generated under `docs/explainability/`. These visualize pixels that most strongly influence predictions, indicating the CNN heavily inspects local textures around the nose, cheeks, and blending borders (warping artifacts).

---

## 6. Key Tradeoffs & Discussion
- **The Accuracy vs. Cost Trade-off:** The CNN outperformed the Classical model in raw accuracy (68.60% vs. 66.94%) and offered dramatically better coverage for fakes (Recall of 82.84% vs. 59.70%). However, this comes at the cost of being 580x slower to train and 5x slower to run in production.
- **Error Complementarity:** Interestingly, the CNN was correct where RF failed in 49 test cases, and RF was correct where the CNN failed in 45 cases. Using **Both Pipelines** in the web app provides a multi-view forensic consensus that is more robust than either model independently.
- **Limitations:** The classical model shows a strong overfitting gap (97% train vs. 66% test) due to the limited resolution and uniform lighting of generated fakes.

---

## 7. Citations
- **[1]** Gaojian Wang, Qian Jiang, Xin Jin, Xiaohui Cui, *“FFR FD: Effective and Fast Detection of DeepFakes Based on Feature Point Defects”*, 2020.
- **[2]** Aman Mishra, Kevin Lan, *“Deepfake Detection”*, Santa Clara University, Department of Computer Science and Engineering Project Report, Group No. 02.
