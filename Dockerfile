# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables for CPU optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MKL_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_DYNAMIC=TRUE

# Add Intel oneAPI repository
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

# Install system dependencies including Intel MKL optimization libraries and libvips for pyvips
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libgomp1 \
    libnuma1 \
    libvips-dev \
    libvips42 \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel \
    intel-oneapi-openmp \
    && rm -rf /var/lib/apt/lists/*

# Configure MKL environment after installation
ENV MKL_INTERFACE_LAYER=LP64 \
    MKL_THREADING_LAYER=GNU \
    LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid CUDA dependencies
RUN pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create directories for mounting
RUN mkdir -p /app/images /app/output

# Set proper permissions
RUN chmod +x main.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Default command
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]