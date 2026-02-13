# Qwen-Image-2512 RunPod Worker

RunPod serverless worker for Qwen-Image-2512 image generation.

## ⚠️ Important Note

Qwen-Image-2512 is a newer model that may not have native ComfyUI support yet. This worker is **experimental** and may require:
- Custom ComfyUI nodes
- Diffusers pipeline instead of ComfyUI
- Additional setup steps

**Test this worker carefully before deploying to production.**

## Model Info
- **Model**: Qwen/Qwen-Image-2512
- **VRAM**: 16-24GB
- **Recommended GPU**: RTX A5000, L40, A6000

## Deployment

1. **Build Docker image** (GitHub Actions will do this automatically on push)
   - Image: `dudushot/qwen-image-2512:latest`

2. **Create RunPod Endpoint**:
   - Go to https://console.runpod.io/serverless
   - Click "New Endpoint"
   - Container Image: `dudushot/qwen-image-2512:v1`
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
    "steps": 30,
    "cfg": 7.0,
    "seed": 12345
  }
}
```

## Default Settings
- Steps: 30
- CFG: 7.0
- Sampler: euler
- Scheduler: normal

## Troubleshooting

If this worker fails to generate images:
1. Check ComfyUI logs for missing nodes
2. Verify Qwen-Image-2512 model format compatibility
3. May need to use diffusers pipeline instead of ComfyUI
4. Consider using FLUX.2-dev as a proven alternative
