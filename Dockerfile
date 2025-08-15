FROM python:3.12-bookworm

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \ 
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the correct application entry point
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
