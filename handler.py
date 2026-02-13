import runpod
import torch
import base64
import io
from PIL import Image

# Global model - loaded once on cold start
pipe = None


def load_model():
    """Load Z-Image-Turbo model."""
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import ZImagePipeline

    print("Loading Z-Image-Turbo model...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    print("Model loaded successfully!")
    return pipe


def handler(event):
    """RunPod serverless handler for Z-Image-Turbo."""
    try:
        input_data = event.get("input", {})
        
        prompt = input_data.get("prompt", "a beautiful sunset")
        negative_prompt = input_data.get("negative_prompt", "")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 9)
        guidance_scale = input_data.get("guidance_scale", 0.0)
        seed = input_data.get("seed", None)

        # Load model
        model = load_model()

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(int(seed))

        # Generate image
        print(f"Generating: {prompt[:50]}... ({width}x{height}, steps={num_inference_steps})")
        
        result = model(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image_base64": img_base64,
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
