# TransitEye Headless & Batch Production Container
FROM python:3.11-slim

# Prevent prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV OMP_NUM_THREADS=1

# Install required system libraries for headless Qt6, BLAS, and PDF validation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libdbus-1-3 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency specifications first for optimal Docker layer caching
COPY pyproject.toml uv.lock README.md NOTICE ./

# Sync dependencies in a frozen environment
RUN uv sync --locked --all-extras

# Copy source tree and manifests
COPY src/ ./src/
COPY desktop/ ./desktop/
COPY configs/ ./configs/
COPY manifests/ ./manifests/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Default CLI entrypoint
ENTRYPOINT ["uv", "run", "transiteye"]
CMD ["--help"]
