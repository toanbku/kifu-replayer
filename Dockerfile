# --- Kifu Replayer: backend + static frontend ---
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for OpenCV + Tesseract.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tesseract-ocr libtesseract-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r ./backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000
ENV PORT=8000

# Run uvicorn from inside the backend dir so module imports resolve.
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
