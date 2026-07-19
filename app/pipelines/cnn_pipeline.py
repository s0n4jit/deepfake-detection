import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from app.config import CNN_MODEL_PATH

# Set device to CPU for production deployment
device = torch.device("cpu")

# Initialize and load model at startup
try:
    cnn_model = models.resnet18()
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    state_dict = torch.load(CNN_MODEL_PATH, map_location=device)
    cnn_model.load_state_dict(state_dict)
    cnn_model.to(device)
    cnn_model.eval()
except Exception as e:
    # Fail fast and loudly at startup if model file is missing
    raise RuntimeError(f"Failed to load CNN model at {CNN_MODEL_PATH}. Error: {e}")

# Preprocessing transforms for ResNet18
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def predict_cnn(face_crop):
    """
    Given a cropped face image (numpy BGR format), preprocess it,
    run the forward pass of ResNet18 model on CPU, and return verdict + confidence.
    """
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    img_rgb = cv2_to_pil(face_crop)
    
    # Preprocess
    tensor_img = cnn_transform(img_rgb).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = cnn_model(tensor_img)
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
