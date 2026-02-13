import runpod
import os
import subprocess
import time
import json
import requests
import random
from pathlib import Path
from huggingface_hub import hf_hub_download

WORKER_VERSION = "v14"

# Track if models are downloaded
models_downloaded = False

def download_models():
    """Download Z-Image-Turbo models from HuggingFace"""
    global models_downloaded
    
    if models_downloaded:
        return
    
    print("📦 Downloading Z-Image-Turbo models...")
    
    hf_token = os.getenv("HF_TOKEN")
    
    models = [
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "target_dir": "/root/ComfyUI/models/diffusion_models",
            "target_name": "z_image_turbo_bf16.safetensors",
        },
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
            "target_dir": "/root/ComfyUI/models/text_encoders",
            "target_name": "qwen_3_4b.safetensors",
        },
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/vae/ae.safetensors",
            "target_dir": "/root/ComfyUI/models/vae",
            "target_name": "ae.safetensors",
        }
    ]
    
    for model in models:
        print(f"📥 Downloading {model['target_name']}...")
        cached_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            cache_dir="/cache",
            token=hf_token
        )
        
        Path(model["target_dir"]).mkdir(parents=True, exist_ok=True)
        target_path = f"{model['target_dir']}/{model['target_name']}"
        subprocess.run(f"ln -sf {cached_path} {target_path}", shell=True, check=True)
        print(f"   ✅ {model['target_name']} ready")
    
    models_downloaded = True
    print("🎉 All models downloaded!")


# Start ComfyUI server
comfy_process = None

def start_comfyui():
    """Start ComfyUI server in background"""
    global comfy_process

    if comfy_process is not None and comfy_process.poll() is not None:
        comfy_process = None

    if comfy_process is None:
        print("🌐 Starting ComfyUI server...")
        comfy_process = subprocess.Popen(
            ["python", "-u", "/root/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"],
        )

        # Wait for server to be ready
        for _ in range(120):
            try:
                resp = requests.get("http://127.0.0.1:8188/system_stats", timeout=1)
                if resp.status_code == 200:
                    print("✅ ComfyUI server ready!")
                    return
            except:
                time.sleep(1)

        print("⚠️ ComfyUI server may not be fully ready")


def handler(event):
    """RunPod handler for Z-Image-Turbo via ComfyUI API"""
    try:
        print(f"🔖 Worker version: {WORKER_VERSION}")
        # Download models first
        download_models()
        
        # Ensure ComfyUI is running
        start_comfyui()
        
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "a beautiful sunset")
        negative_prompt = input_data.get(
            "negative_prompt",
            "worst quality, low quality, blurry, ugly, bad anatomy, watermark, text, signature",
        )
        width = int(input_data.get("width") or 1024)
        height = int(input_data.get("height") or 1024)
        steps = input_data.get("steps")
        if steps is None:
            steps = input_data.get("num_inference_steps")
        steps = int(steps or 4)
        cfg = input_data.get("cfg")
        if cfg is None:
            cfg = input_data.get("guidance_scale")
        cfg = float(cfg or 1.0)
        seed = input_data.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        
        # Z-Image-Turbo workflow (matches official ComfyUI template)
        workflow = {
            "28": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "z_image_turbo_bf16.safetensors",
                    "weight_dtype": "default",
                },
            },
            "11": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {
                    "model": ["28", 0],
                    "shift": 3,
                },
            },
            "30": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_4b.safetensors",
                    "type": "lumina2",
                    "device": "default",
                },
            },
            "27": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["30", 0],
                    "text": prompt,
                },
            },
            "33": {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["27", 0],
                },
            },
            "13": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["11", 0],
                    "positive": ["27", 0],
                    "negative": ["33", 0],
                    "latent_image": ["13", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "res_multistep",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "29": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ae.safetensors"},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["29", 0],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "z_image",
                    "images": ["8", 0],
                },
            },
        }
        
        # Queue prompt
        print(f"🎨 Generating: {prompt[:50]}...")
        resp = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=180
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI /prompt failed ({resp.status_code}): {resp.text}")

        try:
            result = resp.json()
        except Exception:
            raise RuntimeError(f"ComfyUI /prompt returned non-JSON: {resp.text}")
        prompt_id = result.get("prompt_id")
        
        if not prompt_id:
            raise RuntimeError("No prompt_id returned")
        
        # Poll for completion
        for _ in range(120):
            time.sleep(1)
            hist_resp = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=30)
            if hist_resp.status_code == 200:
                hist_data = hist_resp.json()
                if prompt_id in hist_data:
                    outputs = hist_data[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_info in node_output["images"]:
                                filename = img_info.get("filename")
                                subfolder = img_info.get("subfolder", "")
                                
                                # Get image
                                img_url = f"http://127.0.0.1:8188/view?filename={filename}&subfolder={subfolder}&type=output"
                                img_resp = requests.get(img_url, timeout=60)
                                img_resp.raise_for_status()
                                
                                import base64
                                img_base64 = base64.b64encode(img_resp.content).decode("utf-8")
                                
                                return {
                                    "status": "success",
                                    "image_base64": img_base64,
                                }
        
        raise RuntimeError("Timeout waiting for image generation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
        }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
