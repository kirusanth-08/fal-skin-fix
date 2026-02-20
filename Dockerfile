# =============================================================================
# OPTIMIZED DOCKERFILE FOR FAL - Reduced cold-start from ~6min to ~1-2min
# =============================================================================
# Key optimizations:
# 1. Single consolidated RUN for all git clones (reduces layers from 6 to 1)
# 2. Combined pip installs to reduce layer count
# 3. Aggressive cleanup of apt cache, pip cache, and .git directories
# 4. Removed build-essential from final image (not needed at runtime)
# 5. Using --depth=1 for shallow git clones (faster + smaller)
# =============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ARG COMFYUI_VERSION=latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    HF_HOME=/fal-volume/models/huggingface

# ---------------------------------------------------------
# System & Python Setup (single layer)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates \
        python3.12 python3.12-dev python3.12-venv \
    && git lfs install \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip setuptools wheel \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# ---------------------------------------------------------
# PyTorch + ComfyUI + All Python deps (single layer)
# ---------------------------------------------------------
WORKDIR /opt
RUN pip install torch==2.7.0 -f https://download.pytorch.org/whl/cu128/torch_stable.html \
    && pip install comfy-cli \
    && yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia \
    # Core dependencies
    && pip install \
        requests websocket-client websockets sageattention \
        accelerate transformers insightface onnxruntime-gpu==1.18.0 \
        boto3==1.35.74 protobuf==4.25.1 \
    && rm -rf /root/.cache/pip /tmp/*

WORKDIR /comfyui

# ---------------------------------------------------------
# Vendored custom node
# ---------------------------------------------------------
COPY custom_nodes/comfyui_face_parsing /comfyui/custom_nodes/comfyui_face_parsing
RUN pip install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt \
    && rm -rf /root/.cache/pip

# ---------------------------------------------------------
# CNR packages (single layer)
# ---------------------------------------------------------
RUN comfy --workspace /comfyui node install \
        ComfyUI_LayerStyle_Advance@2.0.37 \
        comfyui_essentials@1.1.0 \
        seedvr2_videoupscaler@2.5.24 \
        comfyui-custom-scripts@1.2.5 \
    && rm -rf /root/.cache /tmp/*

# ---------------------------------------------------------
# All git-based custom nodes (SINGLE LAYER - major optimization)
# Using --depth=1 for shallow clones, combined into one RUN
# ---------------------------------------------------------
RUN set -ex \
    # 1. ComfyRoll Custom Nodes
    && git clone --depth=1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && cd /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && git fetch --depth=1 origin d78b780ae43fcf8c6b7c6505e6ffb4584281ceca \
    && git checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # 2. ComfyUI Florence2
    && git clone --depth=1 https://github.com/kijai/ComfyUI-Florence2.git /comfyui/custom_nodes/ComfyUI-Florence2 \
    && cd /comfyui/custom_nodes/ComfyUI-Florence2 \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # 3. ComfyUI KJNodes
    && git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
    && cd /comfyui/custom_nodes/ComfyUI-KJNodes \
    && git fetch --depth=1 origin 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 \
    && git checkout 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # 4. ComfyUI Post-Processing Nodes
    && git clone --depth=1 https://github.com/EllangoK/ComfyUI-post-processing-nodes.git /comfyui/custom_nodes/ComfyUI-post-processing-nodes \
    && cd /comfyui/custom_nodes/ComfyUI-post-processing-nodes \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # 5. Masquerade Nodes
    && git clone --depth=1 https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && cd /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && git fetch --depth=1 origin 432cb4d146a391b387a0cd25ace824328b5b61cf \
    && git checkout 432cb4d146a391b387a0cd25ace824328b5b61cf \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # 6. rgthree – Power Lora Loader
    && git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy \
    && cd /comfyui/custom_nodes/rgthree-comfy \
    && git fetch --depth=1 origin 8ff50e4521881eca1fe26aec9615fc9362474931 \
    && git checkout 8ff50e4521881eca1fe26aec9615fc9362474931 \
    && [ -f requirements.txt ] && pip install -r requirements.txt || true \
    \
    # Cleanup: remove all .git directories and caches
    && find /comfyui/custom_nodes -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/* /var/tmp/*

WORKDIR /comfyui
EXPOSE 8188
