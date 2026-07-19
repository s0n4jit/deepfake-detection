FROM python:3.11-slim

# Install system dependencies required to build and run dlib & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install CPU-optimized packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (app/models/ contains the exported trained models)
COPY app/ ./app/

# Expose port and start ASGI server bound to Render's dynamic port environment variable
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
