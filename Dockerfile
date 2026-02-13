FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install compatible versions
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    safetensors \
    pillow \
    huggingface_hub \
    sentencepiece

# Pre-download the model during build (optional but speeds up cold start)
# Uncomment if you want to bake the model into the image (~12GB larger image)
# RUN python -c "from diffusers import ZImagePipeline; ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo')"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

CMD ["python", "-u", "handler.py"]
