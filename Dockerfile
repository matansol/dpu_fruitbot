FROM python:3.10-slim

# Install system dependencies including Qt5 for procgen build
RUN apt-get update && apt-get install -y \
    fontconfig \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    qtbase5-dev \
    qt5-qmake \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/*

# Set up the virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and build custom procgen from source with C++ extensions
COPY procgen/ ./procgen/
COPY setup.py .
COPY README.md .
# Install in non-editable mode to trigger C++ build
RUN python setup.py install

# Copy application files
COPY app.py .
COPY dpu_clf.py .
COPY common/ ./common/
COPY static/ ./static/
COPY templates/ ./templates/
COPY models/ ./models/

# Create .env file placeholder (override with volume mount or env vars)
RUN touch .env

# Expose the port your app runs on
EXPOSE 8000

# Set environment variable to prevent OpenMP errors
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run your application with Gunicorn & Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "app:socket_app"]
