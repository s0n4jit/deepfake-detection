import os
import cv2
import torch
import pickle
from app.pipelines.preprocessing import preprocess_image
from app.pipelines.classical_pipeline import predict_classical
from app.pipelines.cnn_pipeline import predict_cnn

# Test image path (replace with any image you want to test)
TEST_IMAGE = "dataset/DeepFake00/DeepFake00/aaqaifqrwn.jpg"

def test_single_image():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}. Please specify a valid image path.")
        return
        
    print(f"Loading test image: {TEST_IMAGE}...")
    with open(TEST_IMAGE, "rb") as f:
        img_bytes = f.read()
        
    print("Running preprocessing (face detection and crop)...")
    try:
        face_crop, box, shape = preprocess_image(img_bytes)
        print(f"Face detected successfully! Bounding Box: {box}")
    except Exception as e:
        print(f"Face detection failed: {e}")
        return
        
    print("\n--- Running Classical Pipeline (FAST+BRIEF+RF) ---")
    try:
        class_res = predict_classical(face_crop, shape)
        print(f"Classical Verdict: {class_res['verdict'].upper()} (Confidence: {class_res['confidence'] * 100:.2f}%)")
    except Exception as e:
        print(f"Classical inference failed: {e}")
        
    print("\n--- Running CNN Pipeline (ResNet18) ---")
    try:
        cnn_res = predict_cnn(face_crop)
        print(f"CNN Verdict: {cnn_res['verdict'].upper()} (Confidence: {cnn_res['confidence'] * 100:.2f}%)")
    except Exception as e:
        print(f"CNN inference failed: {e}")

if __name__ == "__main__":
    test_single_image()
