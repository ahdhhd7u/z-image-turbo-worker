FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /root

# Install ComfyUI dependencies
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    comfy-cli==1.5.3 \
    huggingface_hub[hf_transfer]==0.34.4 \
    requests

# Install ComfyUI
RUN comfy --skip-prompt install --fast-deps --nvidia

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# Copy handler
COPY handler.py /root/handler.py

CMD ["python", "-u", "/root/handler.py"]
