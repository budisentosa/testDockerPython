FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_ENABLE_LAB=1

# Install system dependencies needed for data science packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running Jupyter
RUN useradd -m -s /bin/bash jupyter

# Set working directory
WORKDIR /home/jupyter/work

# Copy requirements.txt
COPY --chown=jupyter:jupyter requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy Jupyter configuration
COPY --chown=jupyter:jupyter jupyter_notebook_config.py /home/jupyter/.jupyter/jupyter_notebook_config.py

# Add jupyter configuration directory
RUN mkdir -p /home/jupyter/.jupyter && \
    chown -R jupyter:jupyter /home/jupyter

# Switch to non-root user
USER jupyter

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
