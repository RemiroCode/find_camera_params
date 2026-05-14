# Use an Ubuntu 22.04 CUDA base image (includes nvcc and Python 3.10)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent apt from prompting for timezone/inputs during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python 3.10, and pip
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender-dev \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python so standard commands work
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# --- SOLUCIÓN ERROR LIGHTGLUE ---
# Actualizamos pip y setuptools para que pueda leer pyproject.toml modernos
RUN pip install --upgrade pip setuptools wheel

# --- OPTIMIZACIÓN DE TIEMPO (Caché de Docker) ---
# Instalamos las librerías gigantes (Torch, Bpy) ANTES de copiar requirements.txt. 
# De este modo, si modificas el requirements.txt en el futuro, 
# Docker usará la caché y no volverá a descargar estos 1.3 GB de datos.
RUN pip install --default-timeout=1000 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
RUN pip install --default-timeout=1000 bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

# Ahora sí, copiamos el requirements e instalamos el resto
COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt
RUN pip install fastapi uvicorn requests

# Copiar el resto del código
COPY . .

# Compilar operaciones de RDD (Force CUDA compilation)
ENV FORCE_CUDA=1

# Le decimos a PyTorch que compile para las GPUs más comunes: 
# 7.0 (V100), 7.5 (RTX 2000/T4), 8.0/8.6 (RTX 3000/A100), 8.9 (RTX 4000)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"

# Y por fin compilamos sin aislamiento
RUN cd RDD/models/ops && pip install --no-build-isolation -e .

EXPOSE 8000

# Arrancar nuestra pequeña API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]