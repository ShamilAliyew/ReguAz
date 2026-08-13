from __future__ import annotations

from typing import Any


def select_inference_device(
    preferred: str | None = None, *, torch_module: Any | None = None
) -> str:
    """Prefer Apple MPS, then CUDA, and always provide a safe CPU fallback."""
    if preferred is not None:
        return preferred
    if torch_module is None:
        import torch as torch_module
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"
