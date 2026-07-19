import pickle
import cv2
import numpy as np
import dlib
from app.config import CLASSICAL_MODEL_PATH

# Load model data at startup
try:
    with open(CLASSICAL_MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    scaler = model_data["scaler"]
    classifier = model_data["classifier"]
except Exception as e:
    # Fail fast and loudly at startup if model file is missing
    raise RuntimeError(f"Failed to load classical model pipeline at {CLASSICAL_MODEL_PATH}. Error: {e}")

# Initialize OpenCV feature extractors
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Regional mappings identical to training
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
    x, y, w, h = rect
    px, py = point
    return x < px < x + w and y < py < y + h

def predict_classical(face_crop, shape):
    """
    Extract classical features from the cropped face using the shape landmarks,
    scale using the trained scaler, and run the Random Forest classifier.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    # 1. FAST keypoint detection + BRIEF descriptor calculation
    kp = fast.detect(gray, None)
    kp, des = brief.compute(gray, kp)
    
    descriptor_size = brief.descriptorSize()
    metric = np.zeros((8, descriptor_size), dtype=np.float32)
    counts = np.zeros((8, 1), dtype=np.float32)
    
    if des is not None and len(des) > 0:
        h_crop, w_crop = gray.shape[:2]
        
        # In the preprocessed face crop, run the shape landmarks.
        # Since the crop is resized to 256x256, we can run landmarks on this 256x256 image directly.
        crop_rect = dlib.rectangle(0, 0, w_crop, h_crop)
        # Note: shape was originally calculated on the full image, but we recalculate landmarks on the crop 
        # to ensure local coordinates map correctly.
        # Alternatively, we can project the original landmarks onto the cropped box.
        # Recalculating landmarks on the cropped face is more robust and self-contained!
        # However, to be 100% safe, we recalculate landmarks here:
        from app.pipelines.preprocessing import predictor
        crop_shape = predictor(gray, crop_rect)
        
        landmarks = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            landmarks[i] = [crop_shape.part(i).x, crop_shape.part(i).y]
            
        whole_face_rect = (0, 0, w_crop, h_crop)
        
        for j, kp_point in enumerate(kp):
            pt = kp_point.pt
            des_vector = des[j]
            
            if rect_contains(whole_face_rect, pt):
                metric[7, :] += des_vector
                counts[7, 0] += 1.0
                
            for idx, (name, (start, end)) in enumerate(FACIAL_LANDMARKS_IDXS.items()):
                region_pts = landmarks[start:end]
                if len(region_pts) == 0:
                    continue
                    
                rx, ry, rw, rh = cv2.boundingRect(region_pts)
                padded_rect = (
                    rx - rw // 10,
                    ry - rh // 10,
                    int(1.1 * rw),
                    int(1.1 * rh)
                )
                
                if rect_contains(padded_rect, pt):
                    metric[idx, :] += des_vector
                    counts[idx, 0] += 1.0
                    
        for i in range(8):
            cnt = counts[i, 0]
            if cnt > 0:
                metric[i, :] /= cnt
                
    metric = np.concatenate((metric, counts), axis=1)
    features = metric.flatten()
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    probs = classifier.predict_proba(features_scaled)[0]  # [prob_real, prob_fake]
    
    label_idx = np.argmax(probs)
    verdict = "real" if label_idx == 0 else "fake"
    confidence = float(probs[label_idx])
    
    return {
        "verdict": verdict,
        "confidence": confidence
    }
