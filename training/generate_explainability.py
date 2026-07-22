import os
import json
import numpy as np
import cv2
import pickle
import torch
from torchvision import models, transforms
from PIL import Image

# Setup paths
SPLIT_PATH = "dataset/split.json"
CLASSICAL_MODEL_PATH = "app/models/random_forest_v1.pkl"
CNN_MODEL_PATH = "app/models/cnn_model_v1.pt"
EXPLAIN_DIR = "docs/explainability"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# Import feature extractor from classical_pipeline helper
from train_classical import extract_features

# Load models
with open(CLASSICAL_MODEL_PATH, "rb") as f:
    classical_data = pickle.load(f)
scaler = classical_data["scaler"]
rf_clf = classical_data["classifier"]

# Load CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(CNN_MODEL_PATH, map_location=device)

# Load ResNet18
print("Instantiating ResNet18...")
cnn_model = models.resnet18()
num_features = cnn_model.fc.in_features
if "fc.1.weight" in state_dict:
    cnn_model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.0),
        torch.nn.Linear(num_features, 2)
    )
else:
    cnn_model.fc = torch.nn.Linear(num_features, 2)
    
cnn_model.load_state_dict(state_dict)
cnn_model.to(device)
cnn_model.eval()

# CNN transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def generate_saliency_map(img_path, dest_path):
    # Load raw image
    raw_img = Image.open(img_path).convert("RGB")
    raw_img_resized = raw_img.resize((224, 224))
    
    # Transform for model
    tensor_img = cnn_transform(raw_img).unsqueeze(0).to(device)
    tensor_img.requires_grad_()
    
    # Forward pass
    output = cnn_model(tensor_img)
    pred_idx = output.argmax(dim=1).item()
    
    # Backward pass to get gradients
    score = output[0, pred_idx]
    score.backward()
    
    # Saliency map is maximum absolute gradients across channels
    saliency, _ = torch.max(tensor_img.grad.data.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()
    
    # Normalize to 0-255
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency_img = (saliency * 255).astype(np.uint8)
    
    # Apply color map
    saliency_heatmap = cv2.applyColorMap(saliency_img, cv2.COLORMAP_HOT)
    
    # Superimpose on original resized image
    raw_cv = cv2.cvtColor(np.array(raw_img_resized), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(raw_cv, 0.6, saliency_heatmap, 0.4, 0)
    
    # Save files
    cv2.imwrite(dest_path, overlay)
    return pred_idx

def analyze_test_set():
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
    test_recs = split["test"]
    
    cases = {
        "both_correct_real": [],
        "both_correct_fake": [],
        "both_wrong_real": [],  # Real classified as Fake by both
        "both_wrong_fake": [],  # Fake classified as Real by both
        "cnn_correct_rf_wrong": [],
        "rf_correct_cnn_wrong": []
    }
    
    print("Analyzing test set examples...")
    for idx, rec in enumerate(test_recs):
        img_path = rec["path"]
        true_label = 0 if rec["label"] == "REAL" else 1
        
        # Classical prediction
        feat = extract_features(img_path)
        if feat is None:
            continue
        feat_scaled = scaler.transform([feat])
        rf_pred = rf_clf.predict(feat_scaled)[0]
        
        # CNN prediction
        # Get raw image
        raw_img = Image.open(img_path).convert("RGB")
        tensor_img = cnn_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_pred = cnn_model(tensor_img).argmax(dim=1).item()
            
        rf_ok = (rf_pred == true_label)
        cnn_ok = (cnn_pred == true_label)
        
        # Categorize
        if rf_ok and cnn_ok:
            if true_label == 0:
                cases["both_correct_real"].append((img_path, rf_pred, cnn_pred))
            else:
                cases["both_correct_fake"].append((img_path, rf_pred, cnn_pred))
        elif not rf_ok and not cnn_ok:
            if true_label == 0:
                cases["both_wrong_real"].append((img_path, rf_pred, cnn_pred))
            else:
                cases["both_wrong_fake"].append((img_path, rf_pred, cnn_pred))
        elif cnn_ok and not rf_ok:
            cases["cnn_correct_rf_wrong"].append((img_path, rf_pred, cnn_pred, true_label))
        elif rf_ok and not cnn_ok:
            cases["rf_correct_cnn_wrong"].append((img_path, rf_pred, cnn_pred, true_label))
            
    # Save a few representatives for the report
    report_examples = []
    
    # 1. Both Correct Real
    if cases["both_correct_real"]:
        path = cases["both_correct_real"][0][0]
        dest = os.path.join(EXPLAIN_DIR, "ex_both_correct_real.jpg")
        generate_saliency_map(path, dest)
        report_examples.append({"type": "both_correct_real", "orig": path, "visual": dest})
        
    # 2. Both Correct Fake
    if cases["both_correct_fake"]:
        path = cases["both_correct_fake"][0][0]
        dest = os.path.join(EXPLAIN_DIR, "ex_both_correct_fake.jpg")
        generate_saliency_map(path, dest)
        report_examples.append({"type": "both_correct_fake", "orig": path, "visual": dest})
        
    # 3. Both Wrong Fake
    if cases["both_wrong_fake"]:
        path = cases["both_wrong_fake"][0][0]
        dest = os.path.join(EXPLAIN_DIR, "ex_both_wrong_fake.jpg")
        generate_saliency_map(path, dest)
        report_examples.append({"type": "both_wrong_fake", "orig": path, "visual": dest})
        
    # 4. CNN Correct RF Wrong
    if cases["cnn_correct_rf_wrong"]:
        path = cases["cnn_correct_rf_wrong"][0][0]
        dest = os.path.join(EXPLAIN_DIR, "ex_cnn_correct_rf_wrong.jpg")
        generate_saliency_map(path, dest)
        report_examples.append({"type": "cnn_correct_rf_wrong", "orig": path, "visual": dest})

    print(f"\nSaved explainability example visuals to: {EXPLAIN_DIR}")
    print(f"Stats:")
    for k, v in cases.items():
        print(f"  {k}: {len(v)} occurrences")

if __name__ == "__main__":
    analyze_test_set()
