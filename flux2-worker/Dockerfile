FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI
WORKDIR /root/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod handler dependencies
RUN pip install --no-cache-dir runpod huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# Set working directory back to app
WORKDIR /app

# Start handler
CMD ["python3", "-u", "handler.py"]
