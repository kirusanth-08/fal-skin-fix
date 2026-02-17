FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ARG COMFYUI_VERSION=latest
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

# ---------------------------------------------------------
# System Packages
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git git-lfs wget curl \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ca-certificates build-essential \
    && git lfs install \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip python3-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3
RUN python3.12 -m ensurepip --upgrade
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# Install ComfyUI via comfy-cli
# ---------------------------------------------------------
WORKDIR /opt
RUN pip install comfy-cli
RUN yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia

# ---------------------------------------------------------
# 🔴 CRITICAL: Use ComfyUI venv for EVERYTHING
# ---------------------------------------------------------
ENV VIRTUAL_ENV=/comfyui/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# PyTorch (installed INSIDE ComfyUI venv)
# ---------------------------------------------------------
RUN pip install torch==2.7.0 \
    -f https://download.pytorch.org/whl/cu128/torch_stable.html

# ---------------------------------------------------------
# Runtime Dependencies (all inside venv)
# ---------------------------------------------------------
RUN pip install \
    requests \
    websocket-client \
    websockets \
    accelerate \
    transformers \
    opencv-python \
    insightface \
    onnxruntime-gpu==1.18.0 \
    sageattention

# ---------------------------------------------------------
# Custom Nodes
# ---------------------------------------------------------

WORKDIR /comfyui/custom_nodes

# Face Parsing (required for EXCLUSION node)
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git \
    && pip install -r comfyui_face_parsing/requirements.txt

# Florence2
RUN git clone https://github.com/kijai/ComfyUI-Florence2.git \
    && pip install -r ComfyUI-Florence2/requirements.txt

# SeedVR2
RUN comfy --workspace /comfyui node install seedvr2_videoupscaler

# LayerStyle
RUN comfy --workspace /comfyui node install ComfyUI_LayerStyle_Advance

# Custom Scripts
RUN comfy --workspace /comfyui node install comfyui-custom-scripts

# ComfyRoll
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
    && pip install -r ComfyUI_Comfyroll_CustomNodes/requirements.txt || true

# KJNodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git \
    && pip install -r ComfyUI-KJNodes/requirements.txt

# Post Processing
RUN git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git || true

# Masquerade
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git || true

# rgthree
RUN git clone https://github.com/rgthree/rgthree-comfy.git \
    && pip install -r rgthree-comfy/requirements.txt

# ---------------------------------------------------------
# Verify face parsing actually imports (fail fast)
# ---------------------------------------------------------
RUN python - <<'PY'
import sys
print("Python:", sys.executable)
import importlib
try:
    importlib.import_module("custom_nodes.comfyui_face_parsing")
    print("FaceParsing module import OK")
except Exception as e:
    print("FaceParsing import failed:", e)
    raise
PY

# ---------------------------------------------------------
# fal runtime deps
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

ENV HF_HOME=/fal-volume/models/huggingface

WORKDIR /comfyui
EXPOSE 8188
