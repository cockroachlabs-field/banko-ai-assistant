# Multi-stage build for Banko AI Assistant
# Optimized for both amd64 and arm64 architectures

# Stage 1: Base image with system dependencies
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (including poppler and tesseract for receipt OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build stage - install dependencies
FROM base AS builder

WORKDIR /build

# Copy only dependency files first for better caching
COPY pyproject.toml ./

# Create cache directory structure
RUN mkdir -p /root/.cache

# Install build dependencies and the package
# Install CPU-only PyTorch first to avoid CUDA bloat (saves ~3GB per architecture)
RUN pip install --upgrade pip setuptools wheel && \
    pip install build && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

# Pre-download sentence transformer models to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    echo "Model downloaded successfully" && \
    ls -la /root/.cache/

# Stage 3: Runtime stage - minimal production image
FROM base AS runtime

# Create non-root user for security
RUN useradd -m -u 1000 bankoai && \
    mkdir -p /app && \
    chown -R bankoai:bankoai /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cache directories (sentence transformers models)
RUN mkdir -p /home/bankoai/.cache
COPY --from=builder /root/.cache /home/bankoai/.cache

# Copy application code
COPY --chown=bankoai:bankoai . .

# Copy entrypoint script
COPY --chown=root:root docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Install the package in the runtime stage
RUN pip install --no-deps -e .

# Bake the embedding model into the image so airgap deployments never
# reach for Hugging Face. Roughly 90MB, cached under /root/.cache.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Fix permissions for cache directories
RUN chown -R bankoai:bankoai /home/bankoai/.cache

# Switch to non-root user
USER bankoai

# Expose the application port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/api/health/live || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - gunicorn with threaded workers. The app runs
# Flask-SocketIO in threading mode (long-polling transport), and gunicorn
# 26 removed the eventlet worker class the old command asked for.
# One worker: SocketIO sessions and the in-process coach queue don't
# share state across processes. sh -c so the bind honors PORT.
CMD ["sh", "-c", "exec gunicorn --worker-class gthread -w 1 --threads 16 --bind 0.0.0.0:${PORT:-5000} --timeout 120 --access-logfile - --error-logfile - 'banko_ai.web.app:create_app()'"]
