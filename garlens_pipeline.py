"""Shared image preprocessing and embedding utilities for GarLens."""

from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from typing import Literal

import numpy as np
import torch
from PIL import Image
from rembg import remove
from scipy import ndimage
from transformers import AutoProcessor, CLIPModel

MODEL_NAME = "patrickjohncyh/fashion-clip"


def _load_hf_component(loader, model_name: str):
    """
    Prefer local cache first to avoid repeated network HEAD calls.
    Falls back to remote fetch on first-time setup.
    """
    try:
        return loader(model_name, local_files_only=True)
    except Exception:
        return loader(model_name)


def resolve_device(device_preference: Literal["auto", "cpu", "cuda"] = "auto") -> str:
    """Resolve the runtime device with a safe CPU fallback."""
    normalized = device_preference.lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_bundle(model_name: str = MODEL_NAME, device: str = "cpu"):
    """Load processor and CLIP model on the requested device."""
    processor = _load_hf_component(AutoProcessor.from_pretrained, model_name)
    model = _load_hf_component(CLIPModel.from_pretrained, model_name)
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    return processor, model


def decode_image_bytes(data: bytes) -> Image.Image:
    """Decode image bytes into RGB PIL image."""
    image = Image.open(BytesIO(data))
    return image.convert("RGB")


def remove_background(image_bytes: bytes) -> bytes:
    """Apply rembg to image bytes and return foreground bytes."""
    output = remove(image_bytes)
    if isinstance(output, bytes):
        return output

    # rembg may return PIL objects in some environments.
    buffer = BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


@torch.inference_mode()
def embed_image_bytes(
    image_bytes: bytes,
    processor,
    model: CLIPModel,
    device: str,
) -> list[float]:
    """Generate a normalized embedding vector from image bytes."""
    image = decode_image_bytes(image_bytes)
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    use_cuda_amp = device.startswith("cuda")
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda_amp
        else nullcontext()
    )
    with amp_context:
        features = model.get_image_features(**inputs)
    if isinstance(features, torch.Tensor):
        tensor = features
    elif hasattr(features, "image_embeds"):
        tensor = features.image_embeds
    elif hasattr(features, "pooler_output"):
        tensor = features.pooler_output
    elif hasattr(features, "last_hidden_state"):
        tensor = features.last_hidden_state[:, 0, :]
    else:
        raise TypeError(f"Unsupported image feature output type: {type(features)}")
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)
    return tensor[0].detach().cpu().tolist()


def analyze_foreground_complexity(
    image_bytes: bytes,
    min_component_ratio: float = 0.08,
) -> dict[str, float | int | bool]:
    """
    Analyze a foreground-masked image for multi-garment and pattern/color complexity.

    Returns flags/metrics suitable for cleaning reports and vector metadata.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    rgba = np.asarray(image)
    alpha = rgba[:, :, 3]
    mask = alpha > 16

    if not np.any(mask):
        return {
            "is_multi_garment": False,
            "component_count": 0,
            "is_multicolor": False,
            "dominant_color_count": 0,
            "color_entropy": 0.0,
            "is_patterned": False,
            "pattern_score": 0.0,
            "edge_density": 0.0,
        }

    labeled, _ = ndimage.label(mask.astype(np.uint8))
    area_counts = np.bincount(labeled.ravel())[1:]
    foreground_area = int(area_counts.sum()) or 1
    area_threshold = max(64, int(foreground_area * min_component_ratio))
    component_count = int((area_counts >= area_threshold).sum())
    is_multi_garment = component_count >= 2

    hsv = np.asarray(image.convert("HSV"))
    hue = hsv[:, :, 0][mask].astype(np.float32) * (360.0 / 255.0)
    sat = hsv[:, :, 1][mask].astype(np.float32) / 255.0
    val = hsv[:, :, 2][mask].astype(np.float32) / 255.0

    colorful = (sat > 0.2) & (val > 0.2)
    if np.any(colorful):
        hist = np.histogram(hue[colorful], bins=12, range=(0.0, 360.0))[0].astype(np.float32)
        dist = hist / max(hist.sum(), 1.0)
        dominant_color_count = int((dist >= 0.10).sum())
        entropy = float(-(dist[dist > 0] * np.log2(dist[dist > 0])).sum())
    else:
        dominant_color_count = 0
        entropy = 0.0

    is_multicolor = dominant_color_count >= 3 or entropy >= 1.8

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    gradient_mag = np.hypot(gx, gy)
    fg_gray = gray[mask]
    contrast = float(np.std(fg_gray) / 255.0) if fg_gray.size else 0.0
    edge_density = float((gradient_mag[mask] > 32.0).mean())
    pattern_score = (0.6 * contrast) + (0.4 * edge_density)
    is_patterned = pattern_score >= 0.24 or (is_multicolor and edge_density >= 0.20)

    return {
        "is_multi_garment": is_multi_garment,
        "component_count": component_count,
        "is_multicolor": is_multicolor,
        "dominant_color_count": dominant_color_count,
        "color_entropy": round(entropy, 4),
        "is_patterned": is_patterned,
        "pattern_score": round(float(pattern_score), 4),
        "edge_density": round(edge_density, 4),
    }
