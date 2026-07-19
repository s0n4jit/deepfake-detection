import os
import cv2
import numpy as np
import pytest
from app.pipelines.preprocessing import preprocess_image, NoFaceDetectedError
from app.pipelines.classical_pipeline import predict_classical
from app.pipelines.cnn_pipeline import predict_cnn

def test_preprocessing_no_face():
    # Create a solid black image (no face)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".jpg", img)
    
    with pytest.raises(NoFaceDetectedError):
        preprocess_image(img_bytes.tobytes())
