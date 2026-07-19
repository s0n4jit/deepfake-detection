import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "models_loaded": True}

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "classical" in data
    assert "cnn" in data
    assert data["classical"]["train_accuracy"] == 0.9787

def test_scan_invalid_extension():
    # Test uploading an invalid file extension (like .txt)
    file_content = b"fake file content"
    files = {"file": ("test.txt", file_content, "text/plain")}
    response = client.post("/api/scan", files=files, data={"model": "both"})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
