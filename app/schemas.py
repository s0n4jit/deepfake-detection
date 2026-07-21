from pydantic import BaseModel
from typing import Dict, Optional

class ModelResult(BaseModel):
    verdict: str
    confidence: float

class ScanResponse(BaseModel):
    face_detected: bool
    box: Optional[Dict[str, int]] = None
    results: Dict[str, ModelResult]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

class ModelInfo(BaseModel):
    name: str
    type: str
    train_accuracy: float
    test_accuracy: float
    version: Optional[str] = None

class ModelsInfoResponse(BaseModel):
    cnn: ModelInfo
