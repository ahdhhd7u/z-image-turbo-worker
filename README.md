# SDXL-Turbo RunPod Worker

Custom RunPod serverless handler for SDXL-Turbo image generation.

## Build & Deploy

### Option 1: Build locally and push to Docker Hub

1. **Create a Docker Hub account** at https://hub.docker.com

2. **Login to Docker Hub:**
```bash
docker login
```

3. **Build the image:**
```bash
docker build -t YOUR_DOCKERHUB_USERNAME/z-image-turbo:latest .
```

4. **Push to Docker Hub:**
```bash
docker push YOUR_DOCKERHUB_USERNAME/z-image-turbo:latest
```

### Option 2: Use GitHub Actions (automated)

Push this folder to a GitHub repo and set up GitHub Actions to build and push automatically.

## Create RunPod Serverless Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click **+ New Endpoint**
3. Select **Custom** (not a template)
4. Enter your Docker image: `YOUR_DOCKERHUB_USERNAME/z-image-turbo:latest`
5. Configure:
   - **GPU**: 24GB+ (RTX 4090, A100, H100)
   - **Container Disk**: 20GB (for model cache)
   - **Idle Timeout**: 5 seconds
6. Click **Create**

## API Usage

```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 9,
    "guidance_scale": 0.0,
    "seed": 42
  }
}
```

### Parameters:
- `prompt` (required): Text description
- `width`: Image width (default: 1024)
- `height`: Image height (default: 1024)  
- `num_inference_steps`: Steps (default: 9, Z-Image-Turbo is fast)
- `guidance_scale`: CFG scale (default: 0.0 for Turbo models)
- `seed`: Random seed for reproducibility
