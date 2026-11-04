# Production image for the agentic RAG API.
# Docling, ONNX FlashRank, and Chroma all need a few system libraries.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHROMA_DIR=/data/chroma \
    CHECKPOINT_PATH=/data/checkpoints/rag.sqlite \
    GUARDRAILS_DIR=/app/guardrails \
    HF_HOME=/data/hf \
    HOME=/home/appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .
COPY rag ./rag
COPY guardrails ./guardrails
COPY documents ./documents

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/chroma /data/checkpoints /data/hf /app/uploads \
    && chown -R appuser:appuser /app /data /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
