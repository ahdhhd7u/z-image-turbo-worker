# FLUX.2-dev RunPod Worker

RunPod serverless worker for FLUX.2-dev image generation.

## Model Info
- **Model**: black-forest-labs/FLUX.2-dev
- **VRAM**: 24GB minimum
- **Recommended GPU**: RTX A5000, L40, A6000

## Deployment

1. **Build Docker image** (GitHub Actions will do this automatically on push)
   - Image: `dudushot/flux2-dev:latest`

2. **Create RunPod Endpoint**:
   - Go to https://console.runpod.io/serverless
   - Click "New Endpoint"
   - Container Image: `dudushot/flux2-dev:v1`
   - GPU: Select 24GB+ GPU (RTX A5000 or L40)
   - Container Disk: 50GB
   - Active Workers: 0 (serverless)
   - Max Workers: 3
   - Idle Timeout: 5 seconds
   - Execution Timeout: 600 seconds

3. **Save Endpoint ID** to your bot's `.env` file

## API Input

```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg": 3.5,
    "seed": 12345
  }
}
```

## Default Settings
- Steps: 20
- CFG: 3.5
- Sampler: euler
- Scheduler: simple
