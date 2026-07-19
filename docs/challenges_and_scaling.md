# Engineering Challenges & Scaling Roadmap

This document outlines the package compatibility and deployment challenges encountered during the development of this internship deliverable, and provides a clear technical roadmap for scaling the application to a dedicated Virtual Private Server (VPS) or cloud GPU instance.

---

## 1. Summary of Development & Build Challenges

During the 3-day development phase, several environment conflicts emerged due to package versioning, OS architecture mismatches, and platform resource limits:

### A. Python 3.13 Compatibility & C++ Compilations
- **The Issue:** Python 3.13 is a very recent release. Heavy ML packages like `dlib` (compiled C++ template library) did not have prebuilt Python 3.13 binary wheels (`.whl`) on PyPI for Windows or Linux.
- **Local Resolution:** Installed a community-compiled Python 3.13 Windows wheel (`dlib-20.0.99-cp313`).
- **Docker/Render Resolution:** Standard `pip install dlib` on Debian compiles from source. This C++ compilation is extremely memory-intensive. On Render's Free tier, compiler multi-threading crashed due to out-of-memory (OOM) exhaustion (exceeding 8GB RAM). We resolved this by forcing sequential, single-core compilation in the `Dockerfile` using:
  ```dockerfile
  ENV CMAKE_BUILD_PARALLEL_LEVEL=1
  ENV MAKEFLAGS="-j1"
  ```

### B. OpenCV "Main" vs. "Contrib" Packages
- **The Issue:** The classical pipeline groups keypoints by facial landmark regions using FAST and BRIEF descriptors. BRIEF is part of the `xfeatures2d` module. 
- **The Conflict:** Standard `opencv-python-headless` (chosen to keep the Docker image small by excluding heavy GUI dependencies) does not package the `contrib` extra/non-free modules. This led to a runtime `AttributeError` for `cv2.xfeatures2d`.
- **Resolution:** Updated `requirements.txt` to require `opencv-contrib-python-headless`, which includes the patent-restricted descriptors while remaining lightweight.

### C. PEP 440 Operator Constraints on PyTorch CPU Wheels
- **The Issue:** PyTorch hosts CPU-only binaries on a custom index (`https://download.pytorch.org/whl/cpu`). These wheels are tagged with a local version label (e.g., `+cpu`).
- **The Conflict:** Modern versions of `pip` enforce PEP 440 constraints, which forbid using relational operators (like `>=`) with local version tags.
- **Resolution:** Modified `requirements.txt` to use explicit exact match operators (`==`):
  ```text
  torch==2.13.0+cpu
  torchvision==0.28.0+cpu
  ```

---

## 2. Scaling to a Full-Scale VPS Deployment

While Render's Free tier is excellent for a lightweight prototype, hosting a production-grade, high-accuracy deepfake detection service (including video frame-sequence CNNs and audio spectrogram check) requires a dedicated **Virtual Private Server (VPS)** or cloud GPU instance (e.g. DigitalOcean, Linus, AWS EC2, or RunPod).

### Benefits of a Dedicated VPS/GPU Instance:
1. **No Cold Starts:** The FastAPI app remains constantly active in RAM.
2. **GPU Acceleration:** Allows real-time inference on CNNs and video analysis.
3. **No Build Resource Constraints:** Multi-core parallel builds will not crash due to memory.

### How to Upgrade the Requirements & Code for Production:

#### 1. Transitioning PyTorch to GPU (CUDA)
To utilize a GPU, you must update the PyTorch index URL in `requirements.txt` to pull CUDA-enabled wheels (e.g. CUDA 12.1):
```text
# Replace the CPU index url in requirements.txt with:
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.6.0+cu121
torchvision==0.21.0+cu121
```

#### 2. Updating the Dockerfile to Use a CUDA Base Image
Change the base image from `python:3.11-slim` to an official NVIDIA CUDA base image to compile/run GPU kernels, and enable parallel multi-core compilation:
```dockerfile
# Replace the top of your Dockerfile with:
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable full multi-core build since VPS resources are dedicated
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
ENV MAKEFLAGS="-j4"
```

#### 3. Update Device Mappings in Python Code
Currently, our production model loaders are hardcoded to CPU:
- **`app/pipelines/cnn_pipeline.py`:** Update `device = torch.device("cpu")` to `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **`app/pipelines/preprocessing.py`:** Can remain the same, as dlib face detection is fast enough on CPU for single images, but you can also configure CUDA support for dlib by installing `dlib` with CUDA enabled (`DLIB_USE_CUDA=1`).
