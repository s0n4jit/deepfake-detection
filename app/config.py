import os
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Model Paths
CLASSICAL_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_v1.pkl")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model_v1.pt")
SHAPE_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

# Application Settings
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
