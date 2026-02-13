import fal
from fal.container import ContainerImage
from fal.toolkit import Image
from fastapi import Response, HTTPException
from pathlib import Path
import json
import uuid
import base64
import requests
import websocket
import traceback
import os
import copy
import random
import tempfile
import time
import subprocess
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from typing import Literal
from comfy_models import MODEL_LIST
from workflow import WORKFLOW_JSON

# -------------------------------------------------
# Container setup
# -------------------------------------------------
PWD = Path(__file__).resolve().parent
dockerfile_path = f"{PWD}/Dockerfile"
custom_image = ContainerImage.from_dockerfile(dockerfile_path)

COMFY_HOST = "127.0.0.1:8188"

# -------------------------------------------------
# Presets
# -------------------------------------------------
PRESETS = {
    "imperfect_skin": {"cfg": 0.1, "denoise": 0.34, "resolution": 2048},
    "high_end_skin": {"cfg": 1.1, "denoise": 0.30, "resolution": 3072},
    "smooth_skin": {
        "cfg": 1.1,
        "denoise": 0.30,
        "resolution": 2048,
        "prompt_override": True,
        "positive_prompt": (
            "ultra realistic portrait of [subject], flawless clear face, "
            "smooth radiant skin texture, fine pores, balanced complexion, "
            "healthy glow, cinematic lighting"
        ),
        "negative_prompt": (
            "freckles, spots, blemishes, acne, pigmentation, redness, "
            "rough skin, waxy skin, plastic texture, airbrushed"
        )
    },
    "portrait": {"cfg": 0.5, "denoise": 0.35, "resolution": 2048},
    "mid_range": {"cfg": 1.4, "denoise": 0.40, "resolution": 2048},
    "full_body": {"cfg": 1.5, "denoise": 0.30, "resolution": 2048},
}

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def download_if_missing(url, path):
    if os.path.exists(path):
        return
    ensure_dir(path)
    
    # Add Hugging Face authentication if HF_TOKEN_k is available
    headers = {}
    hf_token = os.environ.get("HF_TOKEN_k") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def check_server(url, retries=500, delay=0.1):
    for _ in range(retries):
        try:
            if requests.get(url).status_code == 200:
                return True
        except:
            pass
        time.sleep(delay)
    return False

def fal_image_to_base64(img: Image) -> str:
    pil = img.to_pil()
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def image_url_to_base64(image_url: str) -> str:
    """Download image from URL and convert to base64."""
    response = requests.get(image_url)
    response.raise_for_status()
    pil = PILImage.open(BytesIO(response.content))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def upload_images(images):
    for img in images:
        blob = base64.b64decode(img["image"])
        files = {"image": (img["name"], BytesIO(blob), "image/png")}
        r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        r.raise_for_status()

# -------------------------------------------------
# Input Model (conditional UI already applied)
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(
        ...,
        title="Input Image",
        description="URL of the image to enhance and upscale."
    )

    skin_preset: Literal[
        "none",
        "imperfect_skin",
        "high_end_skin",
        "smooth_skin",
        "portrait",
        "mid_range",
        "full_body"
    ] = Field(default="none", title="Skin Preset")

    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        title="Skin Realism",
        json_schema_extra={"visible_when": {"skin_preset": "none"}}
    )

    skin_refinement: int = Field(
        default=30,
        ge=0,
        le=100,
        title="Skin Refinement",
        json_schema_extra={"visible_when": {"skin_preset": "none"}}
    )

    seed: int = Field(default=123456789, title="Random Seed")

    upscale_resolution: Literal[
        1024, 1280, 1536, 1792,
        2048, 2304, 2560, 2816, 3072
    ] = Field(
        default=2048,
        title="Upscaler Resolution",
        json_schema_extra={"visible_when": {"skin_preset": "none"}}
    )

# -------------------------------------------------
# Output Model
# -------------------------------------------------
class SkinFixOutput(BaseModel):
    images: list[Image] = Field(
        description="Output images from skin fix processing"
    )

# -------------------------------------------------
# App
# -------------------------------------------------
class SkinFixApp(
    fal.App,
    keep_alive=100,
    min_concurrency=0,
    max_concurrency=5,
    name="skin-fix",
):
    """Skin Fix - Advanced skin refinement and upscaling."""
    
    image = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]

    # 🔒 CRITICAL
    private_logs = True

    def setup(self):
        # Print GPU info
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True
            ).strip()
            print(f"🖥️ GPU Type: {gpu_info}")
        except Exception as e:
            print(f"⚠️ Could not detect GPU: {e}")

        # Download models
        for model in MODEL_LIST:
            download_if_missing(model["url"], model["path"])

        # Symlink models
        for model in MODEL_LIST:
            ensure_dir(model["target"])
            if not os.path.exists(model["target"]):
                os.symlink(model["path"], model["target"])

        # Start ComfyUI (NO --log-stdout)
        self.comfy = subprocess.Popen(
            [
                "python", "-u", "/comfyui/main.py",
                "--disable-auto-launch",
                "--disable-metadata",
                "--listen", "--port", "8188"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            job = copy.deepcopy(WORKFLOW_JSON)
            workflow = job["input"]["workflow"]

            # -------------------------------------------------
            # 1️⃣ Download and read input image resolution (KEY PART)
            # -------------------------------------------------
            # Download image from URL
            image_b64 = image_url_to_base64(input.image_url)
            
            # Get resolution from downloaded image
            pil_img = PILImage.open(BytesIO(base64.b64decode(image_b64)))
            w, h = pil_img.size
            input_image_resolution = max(w, h)

            # Upload image
            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{
                "name": image_name,
                "image": image_b64
            }])
            workflow["32"]["inputs"]["image"] = image_name

            sampler = workflow["29"]["inputs"]

            # -------------------------------------------------
            # 2️⃣ Determine target resolution (NO DOWNSCALE)
            # -------------------------------------------------
            if input.skin_preset != "none":
                p = PRESETS[input.skin_preset]
                target_resolution = max(p["resolution"], input_image_resolution)
                sampler["cfg"] = p["cfg"]
                sampler["denoise"] = p["denoise"]

                if p.get("prompt_override"):
                    workflow["26"]["inputs"]["part1"] = p["positive_prompt"]
                    workflow["25"]["inputs"]["text"] = p["negative_prompt"]
            else:
                sampler["cfg"] = input.cfg
                sampler["denoise"] = 0.30 + (input.skin_refinement / 100.0) * 0.10
                target_resolution = max(input.upscale_resolution, input_image_resolution)

            # Apply seed
            sampler["seed"] = input.seed

            # -------------------------------------------------
            # 3️⃣ Apply resolution to BOTH nodes
            # -------------------------------------------------
            workflow["30"]["inputs"]["new_resolution"] = target_resolution
            workflow["31"]["inputs"]["vae_tile_size"] = target_resolution

            # Always randomize SeedVR2 internal seed
            workflow["30"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

            # -------------------------------------------------
            # 4️⃣ Run ComfyUI
            # -------------------------------------------------
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

            resp = requests.post(
                f"http://{COMFY_HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30
            )
            
            # Log detailed error if request fails
            if resp.status_code != 200:
                error_detail = resp.text
                print(f"ComfyUI Error Response: {error_detail}")
                raise HTTPException(status_code=500, detail=f"ComfyUI rejected workflow: {error_detail}")
            
            prompt_id = resp.json()["prompt_id"]

            while True:
                out = ws.recv()
                if not out.strip().startswith('{'):
                    continue  # Skip non-JSON messages (progress messages)
                msg = json.loads(out)
                if msg.get("type") == "executing" and msg["data"]["node"] is None:
                    break

            history = requests.get(
                f"http://{COMFY_HOST}/history/{prompt_id}"
            ).json()

            images = []
            for node in history[prompt_id]["outputs"].values():
                for img in node.get("images", []):
                    params = (
                        f"filename={img['filename']}"
                        f"&subfolder={img.get('subfolder','')}"
                        f"&type={img['type']}"
                    )
                    r = requests.get(f"http://{COMFY_HOST}/view?{params}")
                    pil_image = PILImage.open(BytesIO(r.content))
                    output_image = Image.from_pil(pil_image, format="png")
                    images.append(output_image)

            ws.close()
            
            # Set billing headers
            response.headers["x-fal-billable-units"] = str(len(images))
            
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
