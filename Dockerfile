FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: requirements.txt includes numpy<2.0.0 to avoid compatibility issues with ChromaDB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY papers/ ./papers/

# Create necessary directories
RUN mkdir -p data/chroma logs

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Note: The actual startup command is defined in docker-compose.yml
# It runs PDF ingestion before starting the API server:
# 1. python scripts/ingest_pdfs.py --clear-existing
# 2. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 