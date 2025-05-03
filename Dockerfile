# Use a Python image with uv pre-installed
FROM downloads.unstructured.io/unstructured-io/unstructured:latest

# Intall uv package manage
# The installer requires curl (and certificates) to download the release archive
# Use Alpine's package manager (apk) instead of apt-get
# Workaround: download and install uv without curl or apk
RUN wget -qO- https://astral.sh/uv/install.sh | bash

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

#PORT for streamlit
EXPOSE 8501

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["uv", "run", "streamlit", "run", "src/financials/app.py", "--server.headless", "true"]


