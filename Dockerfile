# ── NeuroAgent Backend — Dockerfile ────────────────────────────────────────────
# Multi-stage build: slim final image, no dev tools in prod.
#
# Build:  docker build -t neuroagent .
# Run:    docker run --env-file .env -p 8000:8000 neuroagent

# ── Stage 1: builder ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels (lxml, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Install only the pre-built wheels — no compiler needed
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*

# Copy application source
COPY app/ ./app/
COPY pyproject.toml .

# Create log directory (loguru may write here)
RUN mkdir -p logs && chown -R appuser:appgroup /app

USER appuser

# Expose the FastAPI port (Railway injects $PORT)
EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
