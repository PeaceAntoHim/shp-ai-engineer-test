# Use Python 3.11 slim image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in correct order
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    curl \
    procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify gcc installation
RUN gcc --version

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Update requirements.txt to use pre-compiled wheels where possible
COPY requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/output_logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run application
CMD ["python", "main.py"]