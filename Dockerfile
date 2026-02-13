FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    requests \
    pillow \
    torch \
    torchvision \
    torchaudio \
    git+https://github.com/huggingface/transformers.git

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI

# Install ComfyUI requirements
RUN cd /root/ComfyUI && pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /root/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/root/handler.py"]
