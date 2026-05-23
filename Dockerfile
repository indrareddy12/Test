# ============================================================
# Greencare AI — Unified Dockerfile
# All four Lego services use the same image; the entrypoint
# command is specified per-service in docker-compose.yml.
# ============================================================

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────
# libgl1 and libglib2.0-0 are required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Source code ────────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ────────────────────────────────────────────────────
RUN mkdir -p temp_uploads pending_review final_database rejected lego2_temp

# ── Default healthcheck (overridden per-service) ──────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (overridden per service in docker-compose.yml)
CMD ["uvicorn", "lego1_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
