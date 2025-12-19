# WeakIdent PDE-Selector
# Reproducible research environment with PySINDy integration

FROM python:3.11-slim

LABEL maintainer="WeakIdent Research Team"
LABEL description="PDE identification and selector framework"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e ".[pysindy,dev]"

# Run tests by default
CMD ["pytest", "tests/", "-v", "--tb=short"]
