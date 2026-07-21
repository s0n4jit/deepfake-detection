import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from app.config import CNN_MODEL_PATH

# Set device to CPU for production deployment
device = torch.device("cpu")

# Model cache to store loaded models in memory
_loaded_models = {}
DEFAULT_MODEL = "cnn_model_v1.pt"

def load_model_from_file(model_filename: str):
    """
    Dynamically loads and instantiates the model from the models folder based on state_dict inspection.
    """
    models_dir = "app/models"
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    state_dict = torch.load(model_path, map_location=device)
    
    # Detect model type from state_dict keys
    is_efficientnet = any(k.startswith("features.") for k in state_dict.keys())
    
    if is_efficientnet:
        print(f"[{model_filename}] Detected EfficientNet-B0 state_dict. Instantiating...")
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(num_features, 2)
        )
    else:
        print(f"[{model_filename}] Detected ResNet18 state_dict. Instantiating...")
        model = models.resnet18()
        num_features = model.fc.in_features
        if "fc.1.weight" in state_dict:
            model.fc = nn.Sequential(
                nn.Dropout(p=0.0),
                nn.Linear(num_features, 2)
            )
        else:
            model.fc = nn.Linear(num_features, 2)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_cnn_model(model_filename: str = None):
    """
    Retrieves model from cache, loading it if not cached.
    """
    if not model_filename:
        model_filename = DEFAULT_MODEL
        
    if model_filename not in _loaded_models:
        try:
            _loaded_models[model_filename] = load_model_from_file(model_filename)
        except Exception as e:
            # If default fails, raise error. Otherwise, fallback to default.
            if model_filename == DEFAULT_MODEL:
                raise RuntimeError(f"Failed to load default CNN model. Error: {e}")
            print(f"Error loading {model_filename}: {e}. Falling back to default.")
            return get_cnn_model(DEFAULT_MODEL)
            
    return _loaded_models[model_filename]

# Pre-load default model on startup to fail-fast if something is broken
try:
    get_cnn_model(DEFAULT_MODEL)
except Exception as e:
    raise RuntimeError(f"Startup check failed: default model loading failed. Error: {e}")

# Preprocessing transforms for ResNet18 / EfficientNet
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def predict_cnn(face_crop, model_filename: str = None):
    """
    Given a cropped face image (numpy BGR format), preprocess it,
    run the forward pass of selected CNN model on CPU, and return verdict + confidence.
    """
    # Fetch appropriate model
    model = get_cnn_model(model_filename)
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    img_rgb = cv2_to_pil(face_crop)
    
    # Preprocess
    tensor_img = cnn_transform(img_rgb).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    prob_real = float(probs[0])
    prob_fake = float(probs[1])
    
    if prob_real > prob_fake:
        return {"verdict": "real", "confidence": prob_real}
    else:
        return {"verdict": "fake", "confidence": prob_fake}

def cv2_to_pil(cv_img):
    import cv2
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
