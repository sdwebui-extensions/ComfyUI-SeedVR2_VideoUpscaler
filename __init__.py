"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension
from .src.interfaces.video_upscaler import SeedVR2VideoUpscaler
from .src.interfaces.dit_model_loader import SeedVR2LoadDiTModel
from .src.interfaces.vae_model_loader import SeedVR2LoadVAEModel
from .src.interfaces.torch_compile_settings import SeedVR2TorchCompileSettings

NODE_CLASS_MAPPINGS = {
    "SeedVR2VideoUpscaler": SeedVR2VideoUpscaler,
    "SeedVR2LoadDiTModel": SeedVR2LoadDiTModel,
    "SeedVR2LoadVAEModel": SeedVR2LoadVAEModel,
    "SeedVR2TorchCompileSettings": SeedVR2TorchCompileSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2VideoUpscaler": "SeedVR2 Video Upscaler",
    "SeedVR2LoadDiTModel": "SeedVR2 (Down)Load DiT Model",
    "SeedVR2LoadVAEModel": "SeedVR2 (Down)Load VAE Model",
    "SeedVR2TorchCompileSettings": "SeedVR2 Torch Compile Settings"
}

__all__ = ["comfy_entrypoint", "SeedVR2Extension", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
