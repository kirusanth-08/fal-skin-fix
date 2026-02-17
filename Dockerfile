FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ARG COMFYUI_VERSION=latest
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ---------------------------------------------------------
# System & Python Setup
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git git-lfs wget curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates \
    && git lfs install \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip python3-distutils build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# PyTorch (CUDA 12.8)
# ---------------------------------------------------------
RUN pip install torch==2.7.0 -f https://download.pytorch.org/whl/cu128/torch_stable.html

# ---------------------------------------------------------
# ComfyUI Setup
# ---------------------------------------------------------
WORKDIR /opt
RUN pip install comfy-cli
RUN yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia

WORKDIR /comfyui

# ---------------------------------------------------------
# Extra dependencies
# ---------------------------------------------------------
RUN pip install requests websocket-client sageattention \
    accelerate transformers opencv-python insightface onnxruntime-gpu==1.18.0

# FIX: Add missing websocket packages for fal run
RUN pip install websocket-client websockets

# ---------------------------------------------------------
# Skin v03 / ComfyUI Custom Nodes
# ---------------------------------------------------------

# 1. ComfyRoll Custom Nodes (from git with specific hash)
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && cd /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && git checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca

# 2. ComfyUI Essentials v1.1.0 (from registry)
RUN comfy --workspace /comfyui node install comfyui_essentials

# 3. comfyui_face_parsing v1.0.5 (from git latest - CRITICAL for class names)
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git /comfyui/custom_nodes/comfyui_face_parsing \
    && pip install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt

# 4. ComfyUI LayerStyle Advance v2.0.37 (from registry)
RUN comfy --workspace /comfyui node install ComfyUI_LayerStyle_Advance

# 5. comfyui-custom-scripts v1.2.5 (from registry)
RUN comfy --workspace /comfyui node install comfyui-custom-scripts

# 6. SeedVR2 Video Upscaler v2.5.24 (from registry)
RUN comfy --workspace /comfyui node install seedvr2_videoupscaler

# 6. ComfyUI Florence2 (from git - not in snapshot registry)
RUN git clone https://github.com/kijai/ComfyUI-Florence2.git /comfyui/custom_nodes/ComfyUI-Florence2 \
    && pip install -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt

# 7. ComfyUI KJNodes (from git with specific hash)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
    && cd /comfyui/custom_nodes/ComfyUI-KJNodes \
    && git checkout 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 \
    && pip install -r requirements.txt

# 8. ComfyUI Post-Processing Nodes (from git - not in snapshot)
RUN git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git /comfyui/custom_nodes/ComfyUI-post-processing-nodes \
    && true

# 9. Masquerade Nodes (from git with specific hash)
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && cd /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && git checkout 432cb4d146a391b387a0cd25ace824328b5b61cf

# 10. rgthree – Power Lora Loader (from git with specific hash)
RUN git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy \
    && cd /comfyui/custom_nodes/rgthree-comfy \
    && git checkout 8ff50e4521881eca1fe26aec9615fc9362474931 \
    && pip install -r requirements.txt

# Optional nodes from snapshot (may not be essential but present in working env)
# 11. ComfyUI-Manager (from git with specific hash)
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /comfyui/custom_nodes/ComfyUI-Manager \
    && cd /comfyui/custom_nodes/ComfyUI-Manager \
    && git checkout 77377eeddb3d81867c062f1bee122a395e2e8278

# ---------------------------------------------------------
# fal Runtime Requirements
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

ENV HF_HOME=/fal-volume/models/huggingface

WORKDIR /comfyui
EXPOSE 8188