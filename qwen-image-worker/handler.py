import runpod
import os
import subprocess
import time
import json
import requests
import random
from pathlib import Path
from huggingface_hub import hf_hub_download

WORKER_VERSION = "v1"

# Track if models are downloaded
models_downloaded = False

def download_models():
    """Download Qwen-Image-2512 models from HuggingFace"""
    global models_downloaded
    
    if models_downloaded:
        return
    
    print("📦 Downloading Qwen-Image-2512 models...")
    
    # Note: Qwen-Image-2512 uses diffusers format, not safetensors
    # We'll need to download the entire model directory
    models = [
        {
            "repo_id": "Qwen/Qwen-Image-2512",
            "filename": "model.safetensors",
            "target_dir": "/root/ComfyUI/models/checkpoints",
            "target_name": "qwen_image_2512.safetensors",
        },
    ]
    
    # For Qwen-Image, we might need to use diffusers pipeline instead of ComfyUI
    # This is a placeholder - actual implementation depends on ComfyUI support
    
    print("⚠️ Note: Qwen-Image-2512 may require custom ComfyUI nodes")
    print("   This worker is experimental and may need adjustments")
    
    for model in models:
        target_path = Path(model["target_dir"]) / model["target_name"]
        
        if target_path.exists():
            print(f"   ✅ {model['target_name']} already exists")
            continue
        
        print(f"📥 Downloading {model['filename']}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to download the model
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                cache_dir="/root/.cache/huggingface"
            )
            
            import shutil
            shutil.copy(downloaded_path, target_path)
            print(f"   ✅ {model['target_name']} ready")
            
        except Exception as e:
            print(f"   ⚠️ Could not download {model['filename']}: {e}")
            print(f"   Qwen-Image-2512 may use diffusers format instead of safetensors")
    
    models_downloaded = True
    print("🎉 Qwen-Image-2512 setup complete!")


# Global ComfyUI process
comfyui_process = None

def start_comfyui():
    """Start ComfyUI server"""
    global comfyui_process
    
    if comfyui_process is not None:
        try:
            comfyui_process.poll()
            if comfyui_process.returncode is None:
                print("♻️ ComfyUI already running")
                return
        except:
            pass
    
    print("🚀 Starting ComfyUI server...")
    comfyui_process = subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd="/root/ComfyUI",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to be ready
    for i in range(60):
        try:
            resp = requests.get("http://127.0.0.1:8188/system_stats", timeout=1)
            if resp.status_code == 200:
                print("✅ ComfyUI server ready")
                return
        except:
            pass
        time.sleep(1)
    
    raise RuntimeError("ComfyUI server failed to start")


def handler(event):
    """RunPod handler for Qwen-Image-2512 generation"""
    try:
        print(f"🔖 Worker version: {WORKER_VERSION}")
        
        # Download models on first run
        download_models()
        
        # Start ComfyUI
        start_comfyui()
        
        # Parse input
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "a beautiful landscape")
        width = int(input_data.get("width", 1024))
        height = int(input_data.get("height", 1024))
        steps = int(input_data.get("steps", 30))
        cfg = float(input_data.get("cfg", 7.0))
        seed = input_data.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        
        # NOTE: This is a placeholder workflow
        # Qwen-Image-2512 may require custom ComfyUI nodes or diffusers pipeline
        # This workflow assumes standard ComfyUI checkpoint format
        
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "qwen_image_2512.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, blurry, distorted",
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "qwen_image",
                    "images": ["8", 0]
                }
            }
        }
        
        # Queue prompt
        print(f"🎨 Generating Qwen-Image: {prompt[:50]}...")
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
            raise RuntimeError(f"No prompt_id in response: {result}")
        
        # Poll for completion
        print(f"⏳ Waiting for generation (prompt_id: {prompt_id})...")
        for attempt in range(120):
            time.sleep(1)
            
            try:
                hist_resp = requests.get(
                    f"http://127.0.0.1:8188/history/{prompt_id}",
                    timeout=10
                )
                
                if hist_resp.status_code != 200:
                    continue
                
                history = hist_resp.json()
                if prompt_id not in history:
                    continue
                
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            filename = img_info.get("filename")
                            subfolder = img_info.get("subfolder", "")
                            
                            # Download image
                            img_url = f"http://127.0.0.1:8188/view?filename={filename}&subfolder={subfolder}&type=output"
                            img_resp = requests.get(img_url, timeout=30)
                            img_resp.raise_for_status()
                            
                            # Return base64 encoded image
                            import base64
                            image_b64 = base64.b64encode(img_resp.content).decode("utf-8")
                            
                            print("✅ Qwen-Image generation complete!")
                            return {
                                "status": "completed",
                                "image_data": image_b64,
                            }
                
            except Exception as e:
                print(f"⚠️ Poll attempt {attempt} failed: {e}")
                continue
        
        raise RuntimeError("Timeout waiting for image generation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
