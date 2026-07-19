import cv2
import dlib
import numpy as np
from app.config import SHAPE_PREDICTOR_PATH

# Initialize dlib frontal face detector and shape predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except Exception as e:
    # Fail fast and loudly at startup if model file is missing
    raise RuntimeError(f"Failed to load dlib shape predictor at {SHAPE_PREDICTOR_PATH}. Error: {e}")

class NoFaceDetectedError(Exception):
    pass

class MultipleFacesError(Exception):
    pass

def preprocess_image(image_bytes: bytes):
    """
    Decode image, run face detection, crop the primary face with padding,
    and return the cropped face along with bounding box coordinates for overlay.
    """
    # Decode image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format or corrupted file.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        raise NoFaceDetectedError("No face detected in the image.")
        
    # Standard behavior: pick the largest face
    rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    
    # Get coordinates for UI bounding box overlay
    box = {
        "x": int(rect.left()),
        "y": int(rect.top()),
        "w": int(rect.right() - rect.left()),
        "h": int(rect.bottom() - rect.top())
    }
    
    # Get shape/landmarks for pose verification (optional checking or reference)
    shape = predictor(gray, rect)
    
    # Crop face region with 10% padding
    h, w = img.shape[:2]
    l = max(0, rect.left() - int(0.1 * (rect.right() - rect.left())))
    t = max(0, rect.top() - int(0.1 * (rect.bottom() - rect.top())))
    r = min(w, rect.right() + int(0.1 * (rect.right() - rect.left())))
    b = min(h, rect.bottom() + int(0.1 * (rect.bottom() - rect.top())))
    
    face_crop = img[t:b, l:r]
    if face_crop.size == 0:
        raise NoFaceDetectedError("Cropped face region is empty.")
        
    # Resize face crop to standard 256x256
    face_crop_resized = cv2.resize(face_crop, (256, 256))
    
    return face_crop_resized, box, shape
