import os
import logging

import torch

import folder_paths
import comfy.sd

try:
    from flashpack import read_flashpack_file, iterate_from_flash_tensor
    _FLASHPACK_AVAILABLE = True
    _FLASHPACK_IMPORT_ERROR = None
except Exception as exc:
    _FLASHPACK_AVAILABLE = False
    _FLASHPACK_IMPORT_ERROR = exc


logger = logging.getLogger("flashpack_loader")


def _ensure_flashpack_extension() -> None:
    if ".flashpack" in folder_paths.supported_pt_extensions:
        return

    folder_paths.supported_pt_extensions.add(".flashpack")
    for _name, (_paths, exts) in folder_paths.folder_names_and_paths.items():
        if exts is folder_paths.supported_pt_extensions:
            exts.add(".flashpack")

    folder_paths.filename_list_cache.clear()


def _resolve_checkpoint_path(ckpt_name: str) -> str:
    for base_dir in folder_paths.get_folder_paths("checkpoints"):
        candidate = os.path.join(base_dir, ckpt_name)
        if candidate.lower().endswith(".flashpack"):
            if os.path.exists(candidate):
                return candidate
            continue

        flash_candidate = os.path.splitext(candidate)[0] + ".flashpack"
        if os.path.exists(flash_candidate):
            return flash_candidate

        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_name} (searched ComfyUI checkpoints folders)"
    )


def _flashpack_device() -> str:
    device = os.environ.get("FLASHPACK_DEVICE")
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_flashpack_state_dict(path: str) -> dict[str, torch.Tensor]:
    device = _flashpack_device()
    storage, meta = read_flashpack_file(path, device=device, silent=True)
    state_dict: dict[str, torch.Tensor] = {}
    for name, view in iterate_from_flash_tensor(storage, meta):
        state_dict[name] = view
    return state_dict


class FlashpackCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        _ensure_flashpack_extension()
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name: str):
        ckpt_path = _resolve_checkpoint_path(ckpt_name)

        if ckpt_path.lower().endswith(".flashpack"):
            if not _FLASHPACK_AVAILABLE:
                raise RuntimeError(
                    f"flashpack not available: {_FLASHPACK_IMPORT_ERROR}"
                )

            logger.info("Loading flashpack checkpoint: %s", ckpt_path)
            state_dict = _load_flashpack_state_dict(ckpt_path)
            out = comfy.sd.load_state_dict_guess_config(
                state_dict,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            return out[:3]

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


NODE_CLASS_MAPPINGS = {
    "FlashpackCheckpointLoader": FlashpackCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashpackCheckpointLoader": "Flashpack Checkpoint Loader",
}
