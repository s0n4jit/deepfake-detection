FROM python:3.11-slim

# Install system dependencies required to build and run dlib & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Limit parallel compilation to avoid compiler memory exhaustion during dlib build
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAKEFLAGS="-j1"

# Copy requirements and install CPU-optimized packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (app/models/ contains the exported trained models)
COPY app/ ./app/

# Download shape predictor landmark model at build time to keep Git repo light
RUN python -c "import urllib.request, bz2, os; \
    dest = 'app/models/shape_predictor_68_face_landmarks.dat'; \
    os.makedirs(os.path.dirname(dest), exist_ok=True); \
    print('Downloading shape predictor...'); \
    urllib.request.urlretrieve('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', dest + '.bz2'); \
    print('Decompressing shape predictor...'); \
    open(dest, 'wb').write(bz2.BZ2File(dest + '.bz2').read()); \
    os.remove(dest + '.bz2'); \
    print('Shape predictor ready!')"


# Expose port and start ASGI server bound to Render's dynamic port environment variable
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
