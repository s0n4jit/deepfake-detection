import os
import json
import time
import pickle
import numpy as np
import cv2
import dlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Configuration
SPLIT_PATH = "dataset/split.json"
SHAPE_PREDICTOR_PATH = "app/models/shape_predictor_68_face_landmarks.dat"
MODEL_EXPORT_PATH = "app/models/random_forest_v1.pkl"

# Initialize detectors
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Define facial landmarks regions (imutils style)
FACIAL_LANDMARKS_IDXS = {
    "mouth": (48, 68),
    "inner_mouth": (60, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 36)
}

def rect_contains(rect, point):
    """
    Check if rect (x, y, w, h) contains point (px, py)
    """
    x, y, w, h = rect
    px, py = point
    return x < px < x + w and y < py < y + h

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # 1. Detect FAST keypoints and compute BRIEF descriptors
    kp = fast.detect(img, None)
    kp, des = brief.compute(img, kp)
    
    descriptor_size = brief.descriptorSize()  # usually 32 bytes
    metric = np.zeros((8, descriptor_size), dtype=np.float32)
    counts = np.zeros((8, 1), dtype=np.float32)
    
    if des is None or len(des) == 0:
        # Return flattened zeros if no descriptors found
        return np.zeros(8 * (descriptor_size + 1), dtype=np.float32)
        
    # 2. Get landmarks (assuming the entire cropped image is the face)
    h_img, w_img = img.shape[:2]
    rect = dlib.rectangle(0, 0, w_img, h_img)
    shape = predictor(img, rect)
    
    # Convert shape to numpy array
    landmarks = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        landmarks[i] = [shape.part(i).x, shape.part(i).y]
        
    # Bounding box of the entire face
    whole_face_rect = (0, 0, w_img, h_img)
    
    # Group keypoints/descriptors by region
    for j, kp_point in enumerate(kp):
        pt = kp_point.pt
        des_vector = des[j]
        
        # Whole face is region 7
        if rect_contains(whole_face_rect, pt):
            metric[7, :] += des_vector
            counts[7, 0] += 1.0
            
        # Check sub-regions
        for idx, (name, (start, end)) in enumerate(FACIAL_LANDMARKS_IDXS.items()):
            region_pts = landmarks[start:end]
            if len(region_pts) == 0:
                continue
                
            rx, ry, rw, rh = cv2.boundingRect(region_pts)
            # Expand region slightly as in reference paper (10% padding)
            padded_rect = (
                rx - rw // 10,
                ry - rh // 10,
                int(1.1 * rw),
                int(1.1 * rh)
            )
            
            if rect_contains(padded_rect, pt):
                metric[idx, :] += des_vector
                counts[idx, 0] += 1.0
                
    # Average descriptors per region
    for i in range(8):
        cnt = counts[i, 0]
        if cnt > 0:
            metric[i, :] /= cnt
            
    # Concatenate counts column to make 33-column rows
    metric = np.concatenate((metric, counts), axis=1)
    
    return metric.flatten()

def train_classical():
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
        
    train_recs = split["train"]
    test_recs = split["test"]
    
    print(f"Extracting features for {len(train_recs)} training images...")
    X_train = []
    y_train = []
    
    start_time = time.time()
    for rec in train_recs:
        feat = extract_features(rec["path"])
        if feat is not None:
            X_train.append(feat)
            y_train.append(0 if rec["label"] == "REAL" else 1)
            
    print(f"Extracting features for {len(test_recs)} testing images...")
    X_test = []
    y_test = []
    for rec in test_recs:
        feat = extract_features(rec["path"])
        if feat is not None:
            X_test.append(feat)
            y_test.append(0 if rec["label"] == "REAL" else 1)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    print("Training Random Forest Classifier...")
    rf_start = time.time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    rf_train_time = time.time() - rf_start
    
    # Evaluate
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
    cm = confusion_matrix(y_test, y_pred_test)
    
    print("\nResults:")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    total_time = time.time() - start_time
    print(f"\nTraining time for Random Forest: {rf_train_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    
    # Measure inference time per image
    inf_start = time.time()
    for x in X_test_scaled[:100]:
        clf.predict([x])
    avg_inf_time = (time.time() - inf_start) / 100.0
    print(f"Average Inference Time per image: {avg_inf_time * 1000.0:.2f} ms")
    
    # Export model + scaler together as a single pickle
    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)
    with open(MODEL_EXPORT_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "classifier": clf,
            "feature_regions": list(FACIAL_LANDMARKS_IDXS.keys())
        }, f)
    print(f"Saved trained classical model pipeline to {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    train_classical()
