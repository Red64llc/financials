# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

# Install the project into `/app`
WORKDIR /workspace

