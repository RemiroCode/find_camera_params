FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

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

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install --default-timeout=1000 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
RUN pip install --default-timeout=1000 bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

RUN apt-get update && apt-get install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6

COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt
RUN pip install fastapi uvicorn requests

COPY . .
ENV FORCE_CUDA=1

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"

RUN cd RDD/models/ops && pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]