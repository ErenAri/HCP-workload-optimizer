# ---------------------------------------------------------------------------
# Multi-stage Docker build for HPC Workload Optimizer
# ---------------------------------------------------------------------------

# ---- Stage 1: builder – install Python dependencies into a virtual-env ----
FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Stage 2: runtime – lean image with only what we need -----------------
FROM python:3.12-slim AS runtime

# Install curl for the health-check probe
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy project source and install the package itself
COPY pyproject.toml README.md ./
COPY python/ python/
COPY schemas/ schemas/
COPY configs/ configs/

RUN pip install --no-cache-dir .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

ENTRYPOINT ["hpcopt", "serve", "api", "--host", "0.0.0.0", "--port", "8080"]
