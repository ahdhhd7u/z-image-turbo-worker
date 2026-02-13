import runpod
import os
import subprocess
import time
import json
import requests
import random
from pathlib import Path
from huggingface_hub import hf_hub_download

# Track if models are downloaded
models_downloaded = False

def download_models():
    """Download Z-Image-Turbo models from HuggingFace"""
    global models_downloaded
    
    if models_downloaded:
        return
    
    print("📦 Downloading FLUX.1-schnell models...")
    
    hf_token = os.getenv("HF_TOKEN")
    
    models = [
        {
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "filename": "flux1-schnell.safetensors",
            "target_dir": "/root/ComfyUI/models/unet",
            "target_name": "flux1-schnell.safetensors",
        },
        {
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "t5xxl_fp16.safetensors",
            "target_dir": "/root/ComfyUI/models/clip",
            "target_name": "t5xxl_fp16.safetensors",
        },
        {
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "clip_l.safetensors",
            "target_dir": "/root/ComfyUI/models/clip",
            "target_name": "clip_l.safetensors",
        },
        {
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "filename": "ae.safetensors",
            "target_dir": "/root/ComfyUI/models/vae",
            "target_name": "ae.sft",
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
    
    if comfy_process is None:
        print("🌐 Starting ComfyUI server...")
        comfy_process = subprocess.Popen(
            ["python", "/root/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        for _ in range(60):
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
        # Download models first
        download_models()
        
        # Ensure ComfyUI is running
        start_comfyui()
        
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "a beautiful sunset")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        steps = input_data.get("steps", 10)
        cfg = input_data.get("cfg", 1.1)
        seed = input_data.get("seed", random.randint(0, 2**32 - 1))
        
        # Use FLUX.1-schnell workflow (proven and stable)
        workflow = {
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["11", 0]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["13", 0],
                    "vae": ["10", 0]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "flux",
                    "images": ["8", 0]
                }
            },
            "10": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "ae.sft"
                }
            },
            "11": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "t5xxl_fp16.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux"
                }
            },
            "12": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "flux1-schnell.safetensors",
                    "weight_dtype": "default"
                }
            },
            "13": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["12", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["27", 0]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["11", 0]
                }
            },
            "27": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            }
        }
        
        # Queue prompt
        print(f"🎨 Generating: {prompt[:50]}...")
        resp = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=180
        )
        resp.raise_for_status()
        result = resp.json()
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
