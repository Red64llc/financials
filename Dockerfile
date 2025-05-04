# Use a Python image with uv pre-installed
FROM downloads.unstructured.io/unstructured-io/unstructured:latest

# Intall uv package manage
# The installer requires curl (and certificates) to download the release archive
# Use Alpine's package manager (apk) instead of apt-get
# Workaround: download and install uv without curl or apk
RUN wget -qO- https://astral.sh/uv/install.sh | bash

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Explicitly set user to root
USER root

# Copy the project into the image
COPY . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app


# Create output directory in the container
RUN mkdir -p /app/output/images
# Change ownership to the user running the build (likely root)
RUN chown -R notebook-user:notebook-user .

# Install dependencies and the package itself in development mode
RUN uv sync --locked && uv pip install -e .

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

#PORT for streamlit
EXPOSE 8501
