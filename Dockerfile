# Base image with uv and Python 3.13
# Stage 1: get uv binary with Python tooling
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS uv

# Stage 2: CUDA runtime image matching PyTorch cu126 wheels
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Install runtime libraries needed by OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	libegl1 libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
	&& rm -rf /var/lib/apt/lists/*

# Copy uv static binary from stage 1
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

# Configure uv to manage Python (downloads 3.13 per pyproject requires-python)
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
	UV_PYTHON_PREFERENCE=managed \
	UV_LINK_MODE=copy

# Set working directory
WORKDIR /home/OCTA-seg

# Copy only dependency metadata first for better Docker layer caching
COPY pyproject.toml ./

# Create and populate a project-local virtual environment with uv
# This reads dependencies from pyproject.toml and installs Python 3.13 if needed
RUN uv sync --no-dev

# Ensure the virtual environment is on PATH
ENV PATH="/home/OCTA-seg/.venv/bin:$PATH"

# Copy the rest of the repository
COPY . .

# Sync again to install the local project into the environment (uses cached deps)
RUN uv sync --no-dev

# Make entrypoint script executable if present
RUN chmod +x /home/OCTA-seg/docker/dockershell.sh || true

RUN echo "Successfully built image!"

ENTRYPOINT ["/home/OCTA-seg/docker/dockershell.sh"]
