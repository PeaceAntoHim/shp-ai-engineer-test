# Use Python 3.11 slim image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dri \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/output_logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run application
CMD ["python", "main.py"]
