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
import sys
import logging
import warnings
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from comfy_models import MODEL_LIST
from workflow import WORKFLOW_JSON

# Suppress urllib and fal toolkit warnings
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("fal.toolkit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="fal.toolkit")

# -------------------------------------------------
# Container setup
# -------------------------------------------------
PWD = Path(__file__).resolve().parent
dockerfile_path = f"{PWD}/Dockerfile"
custom_image = ContainerImage.from_dockerfile(dockerfile_path)

COMFY_HOST = "127.0.0.1:8188"
DEBUG_LOGS = os.environ.get("FAL_DEBUG") == "1"

def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)

# -------------------------------------------------
# Presets
# -------------------------------------------------
PRESETS = {
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
    "imperfect_skin": {"cfg": 0.1, "denoise": 0.34, "resolution": 2048},
    "portrait": {"cfg": 0.5, "denoise": 0.35, "resolution": 2048},
    "mid_range": {"cfg": 1.4, "denoise": 0.40, "resolution": 2048},
    "full_body": {"cfg": 1.5, "denoise": 0.30, "resolution": 2048},
    "high_end_skin": {"cfg": 1.1, "denoise": 0.30, "resolution": 3072},
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

def run_workflow(workflow: dict, timeout_seconds: int = 300) -> dict:
    """Execute a ComfyUI workflow and return its history entry."""
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.settimeout(timeout_seconds)
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    try:
        resp = requests.post(
            f"http://{COMFY_HOST}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")

        prompt_id = resp.json()["prompt_id"]
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            out = ws.recv()
            if not out.strip().startswith('{'):
                continue
            msg = json.loads(out)
            if msg.get("type") == "executing" and msg["data"]["node"] is None:
                break
        else:
            raise TimeoutError("ComfyUI workflow execution timed out")

        history = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30).json()
        return history[prompt_id]
    finally:
        ws.close()

# -------------------------------------------------
# Input Model
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(
        ...,
        title="Input Image",
        description="URL of the image to enhance and upscale."
    )

    mode: Literal["preset", "custom"] = Field(
        default="preset",
        title="Configuration Mode",
        description="Choose 'preset' to use predefined settings or 'custom' for manual control"
    )

    preset_name: Optional[Literal[
        "smooth_skin",
        "imperfect_skin",
        "portrait",
        "mid_range",
        "full_body",
        "high_end_skin"
    ]] = Field(
        default="smooth_skin",
        title="Preset",
        description="Select a preset (only active when mode is 'preset')"
    )

    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        title="Skin Realism",
        description="Adjust skin realism (only active when mode is 'custom')"
    )

    skin_refinement: int = Field(
        default=30,
        ge=0,
        le=100,
        title="Skin Refinement",
        description="Adjust skin refinement level (only active when mode is 'custom')"
    )

    seed: int = Field(default=123456789, title="Random Seed")

    upscale_resolution: Literal[
        1024, 1280, 1536, 1792,
        2048, 2304, 2560, 2816, 3072
    ] = Field(
        default=2048,
        title="Upscaler Resolution",
        description="Target resolution for upscaling (only active when mode is 'custom')"
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
    name="skin-new",
):
    """Skin Fix - Advanced skin refinement and upscaling."""
    
    image = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]

    # 🔒 CRITICAL
    private_logs = True  # Set to True if logs may contain sensitive info (e.g. image URLs)

    def setup(self):
        # Print GPU info
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True
            ).strip()
            debug_log(f"🖥️ GPU Type: {gpu_info}")
        except Exception as e:
            debug_log(f"⚠️ Could not detect GPU: {e}")

        # Download models
        for model in MODEL_LIST:
            download_if_missing(model["url"], model["path"])

        # Symlink models
        for model in MODEL_LIST:
            ensure_dir(model["target"])
            if not os.path.exists(model["target"]):
                os.symlink(model["path"], model["target"])

        # Preflight: verify face_parsing node is present and importable
        try:
            node_dir = "/comfyui/custom_nodes/comfyui_face_parsing"
            debug_log(f"🧩 face_parsing node dir exists: {os.path.isdir(node_dir)}")
            if "/comfyui" not in sys.path:
                sys.path.insert(0, "/comfyui")
            import importlib
            fp_module = importlib.import_module("custom_nodes.comfyui_face_parsing")
            node_map = getattr(fp_module, "NODE_CLASS_MAPPINGS", {})
            has_parser = "FaceParsingResultsParser(FaceParsing)" in node_map
            debug_log(f"🧩 face_parsing node registered: {has_parser}")
        except Exception as e:
            debug_log(f"❌ face_parsing import failed: {e}")

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

        # Verify ComfyUI registered the face_parsing node
        try:
            info = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
            info.raise_for_status()
            nodes = info.json()
            if "FaceParsingResultsParser(FaceParsing)" not in nodes:
                raise RuntimeError("FaceParsingResultsParser(FaceParsing) not in object_info")
            debug_log("✅ ComfyUI reports FaceParsingResultsParser(FaceParsing) is available")
        except Exception as e:
            raise RuntimeError(f"ComfyUI missing face_parsing node: {e}")

        # Warmup: execute one synthetic run in setup so model/node loading isn't charged to first request.
        try:
            warmup_job = copy.deepcopy(WORKFLOW_JSON)
            warmup_workflow = warmup_job["input"]["workflow"]

            warmup_image = PILImage.new("RGB", (1024, 1024), color=(127, 127, 127))
            warmup_buf = BytesIO()
            warmup_image.save(warmup_buf, format="PNG")
            warmup_name = f"warmup_{uuid.uuid4().hex}.png"
            upload_images([{
                "name": warmup_name,
                "image": base64.b64encode(warmup_buf.getvalue()).decode()
            }])
            warmup_workflow["545"]["inputs"]["image"] = warmup_name

            warmup_sampler = warmup_workflow["510"]["inputs"]
            warmup_sampler["cfg"] = PRESETS["smooth_skin"]["cfg"]
            warmup_sampler["denoise"] = PRESETS["smooth_skin"]["denoise"]
            warmup_sampler["seed"] = random.randint(0, 2**32 - 1)

            warmup_workflow["548"]["inputs"]["resolution"] = 1024
            warmup_workflow["548"]["inputs"]["max_resolution"] = 4096
            warmup_workflow["548"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            warmup_workflow["549"]["inputs"]["encode_tile_size"] = 1024
            warmup_workflow["549"]["inputs"]["decode_tile_size"] = 1024

            run_workflow(warmup_workflow, timeout_seconds=420)
            debug_log("✅ Warmup workflow completed")
        except Exception as e:
            # Do not block startup on warmup, but surface context for debugging.
            debug_log(f"⚠️ Warmup workflow failed: {e}")

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
            workflow["545"]["inputs"]["image"] = image_name

            sampler = workflow["510"]["inputs"]

            # -------------------------------------------------
            # 2️⃣ Apply settings based on mode
            # -------------------------------------------------
            if input.mode == "preset":
                # Use preset settings
                p = PRESETS[input.preset_name]
                target_resolution = max(p["resolution"], input_image_resolution)
                sampler["cfg"] = p["cfg"]
                sampler["denoise"] = p["denoise"]

                if p.get("prompt_override"):
                    workflow["506"]["inputs"]["part1"] = p["positive_prompt"]
                    workflow["507"]["inputs"]["text"] = p["negative_prompt"]
            else:
                # Use custom settings
                sampler["cfg"] = input.cfg
                sampler["denoise"] = 0.30 + (input.skin_refinement / 100.0) * 0.10
                target_resolution = max(input.upscale_resolution, input_image_resolution)

            # Apply seed
            sampler["seed"] = input.seed

            # -------------------------------------------------
            # 3️⃣ Apply resolution to SeedVR2 nodes
            # -------------------------------------------------
            workflow["548"]["inputs"]["resolution"] = target_resolution
            workflow["548"]["inputs"]["max_resolution"] = 4096
            workflow["549"]["inputs"]["encode_tile_size"] = min(1024, target_resolution)
            workflow["549"]["inputs"]["decode_tile_size"] = min(1024, target_resolution)

            # Always randomize SeedVR2 internal seed
            workflow["548"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

            # -------------------------------------------------
            # 4️⃣ Run ComfyUI
            # -------------------------------------------------
            try:
                history_entry = run_workflow(workflow)
            except Exception as e:
                debug_log(f"ComfyUI Error Response: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            images = []
            for node in history_entry["outputs"].values():
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
            
            # Set billing headers
            response.headers["x-fal-billable-units"] = str(len(images))
            
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
