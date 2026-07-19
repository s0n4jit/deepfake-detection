import os
import asyncio
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.config import STATIC_DIR, MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS
from app.schemas import ScanResponse, HealthResponse, ModelsInfoResponse
from app.pipelines.preprocessing import preprocess_image, NoFaceDetectedError
from app.pipelines.classical_pipeline import predict_classical
from app.pipelines.cnn_pipeline import predict_cnn

async def ping_self(app_url: str):
    url = f"{app_url.rstrip('/')}/healthz"
    # Delay initial ping slightly to allow server to bind fully
    await asyncio.sleep(15)
    while True:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=10).read())
        except Exception:
            pass
        await asyncio.sleep(240)  # Ping every 4 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_url = os.getenv("APP_URL")
    if app_url:
        asyncio.create_task(ping_self(app_url))
    yield

app = FastAPI(
    title="Deepfake Image Detector",
    description="Cybersecurity + ML deliverable. Classical FAST+BRIEF+RF vs ResNet18 comparison.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Size Limit Middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/scan":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE / (1024 * 1024):.0f}MB."}
            )
    return await call_next(request)

# Serve Frontend static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index file not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/healthz", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    # Since imports fail-fast at startup, if we reached here, models are successfully loaded
    return {"status": "healthy", "models_loaded": True}

@app.get("/api/models", response_model=ModelsInfoResponse)
async def get_models_info():
    return {
        "classical": {
            "name": "Classical Pipeline (FAST+BRIEF+RF)",
            "type": "Keypoint engineered features + Random Forest",
            "train_accuracy": 0.9787,
            "test_accuracy": 0.6694
        },
        "cnn": {
            "name": "CNN Pipeline (ResNet18)",
            "type": "Deep Transfer Learning on cropped faces",
            "train_accuracy": 0.7660,
            "test_accuracy": 0.6860
        }
    }

@app.post("/api/scan", response_model=ScanResponse)
async def scan_image(
    file: UploadFile = File(...),
    model: str = Form("cnn")  # "classical", "cnn", or "both"
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} images are accepted."
        )
        
    # Read file contents
    contents = await file.read()
    
    try:
        # Preprocessing (Shared)
        face_crop, box, shape = preprocess_image(contents)
    except NoFaceDetectedError as e:
        return {
            "face_detected": False,
            "box": None,
            "results": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Run predictions
    results = {}
    model_choice = model.lower()
    
    if model_choice in ("classical", "both"):
        results["classical"] = predict_classical(face_crop, shape)
        
    if model_choice in ("cnn", "both"):
        results["cnn"] = predict_cnn(face_crop)
        
    return {
        "face_detected": True,
        "box": box,
        "results": results
    }
