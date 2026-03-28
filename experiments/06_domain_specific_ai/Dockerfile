# Multi-stage Dockerfile for Game Dev AI Assistant

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_app.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy application code
COPY app.py .
COPY multimodal_assistant.py .
COPY rag_pipeline.py .
COPY game_dev_assistant.py .

# Create directories
RUN mkdir -p uploads vector_store_web

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/status')"

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
