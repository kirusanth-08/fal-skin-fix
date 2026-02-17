import os
import sys
import subprocess
import importlib.metadata as ilmd


def _ensure_requirements():
    cur_dir = os.path.dirname(__file__)
    req_path = os.path.join(cur_dir, "requirements.txt")
    if not os.path.exists(req_path):
        return

    try:
        with open(req_path, "r", encoding="utf-8") as f:
            required = [
                ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")
            ]

        def _pkg_name(spec: str) -> str:
            name = spec
            for sep in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
                if sep in name:
                    name = name.split(sep, 1)[0]
            if "[" in name:
                name = name.split("[", 1)[0]
            return name.strip()

        missing = []
        for spec in required:
            name = _pkg_name(spec)
            try:
                ilmd.version(name)
            except ilmd.PackageNotFoundError:
                missing.append(spec)

        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception:
        pass


_ensure_requirements()

import folder_paths

try:
    from torchvision.datasets.utils import download_url as _tv_download_url

    def _download(url: str, dst_dir: str, filename: str):
        _tv_download_url(url, dst_dir, filename)

except Exception:
    import urllib.request
    import shutil

    def _download(url: str, dst_dir: str, filename: str):
        os.makedirs(dst_dir, exist_ok=True)
        tmp_path = os.path.join(dst_dir, filename + ".part")
        final_path = os.path.join(dst_dir, filename)
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp_path, final_path)


models_path = folder_paths.models_dir
face_parsing_path = os.path.join(models_path, "face_parsing")
ultralytics_bbox_path = os.path.join(models_path, "ultralytics", "bbox")

os.makedirs(face_parsing_path, exist_ok=True)
os.makedirs(ultralytics_bbox_path, exist_ok=True)

folder_paths.add_model_folder_path("ultralytics_bbox", ultralytics_bbox_path)

if not os.path.exists(os.path.join(face_parsing_path, "model.safetensors")):
    _download(
        "https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.safetensors?download=true",
        face_parsing_path,
        "model.safetensors",
    )
if not os.path.exists(os.path.join(face_parsing_path, "config.json")):
    _download(
        "https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json?download=true",
        face_parsing_path,
        "config.json",
    )
if not os.path.exists(os.path.join(face_parsing_path, "preprocessor_config.json")):
    _download(
        "https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json?download=true",
        face_parsing_path,
        "preprocessor_config.json",
    )

if not os.path.exists(os.path.join(ultralytics_bbox_path, "face_yolov8m.pt")):
    _download(
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
        ultralytics_bbox_path,
        "face_yolov8m.pt",
    )

from .face_parsing_nodes import NODE_CLASS_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS"]
