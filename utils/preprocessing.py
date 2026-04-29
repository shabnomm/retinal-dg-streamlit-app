from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from utils.data_loader import load_preprocessing_config


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SUPPORTED_IMAGE_SUFFIXES: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if needed."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _load_config() -> dict:
    """Load preprocessing config from metadata."""
    config = load_preprocessing_config()

    image_size = config.get("image_size")
    mean = config.get("mean")
    std = config.get("std")

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Invalid preprocessing config: `image_size` must be a positive integer.")

    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("Invalid preprocessing config: `mean` must be a list of 3 values.")

    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("Invalid preprocessing config: `std` must be a list of 3 values.")

    return config


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_image_size() -> int:
    """Return configured image size."""
    config = _load_config()
    return int(config["image_size"])


def get_normalization_stats() -> tuple[list[float], list[float]]:
    """Return mean and std from config."""
    config = _load_config()
    return list(config["mean"]), list(config["std"])


def build_inference_transform() -> transforms.Compose:
    """
    Build the standard inference transform.

    Output:
        - resized image
        - normalized tensor
    """
    config = _load_config()
    image_size = int(config["image_size"])
    mean = list(config["mean"])
    std = list(config["std"])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_image_from_path(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk as RGB PIL Image.

    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if suffix unsupported or file is not a valid image
    """
    path = Path(image_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )

    try:
        image = Image.open(path)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image file: {path}") from exc

    return _ensure_rgb(image)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an image from uploaded bytes as RGB PIL Image.

    Raises:
        ValueError: if bytes are invalid or not decodable as image
    """
    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    return _ensure_rgb(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a normalized tensor of shape [1, C, H, W].
    """
    rgb_image = _ensure_rgb(image)
    transform = build_inference_transform()
    tensor = transform(rgb_image).unsqueeze(0)
    return tensor


def image_path_to_tensor(image_path: str | Path) -> torch.Tensor:
    """
    Load image from path and convert to normalized tensor.
    """
    image = load_image_from_path(image_path)
    return pil_to_tensor(image)


def image_bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """
    Load image from raw bytes and convert to normalized tensor.
    """
    image = load_image_from_bytes(image_bytes)
    return pil_to_tensor(image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to numpy array (H, W, C) in uint8 RGB format.
    """
    rgb_image = _ensure_rgb(image)
    return np.array(rgb_image, dtype=np.uint8)


def tensor_to_displayable_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor to a displayable uint8 RGB image.

    Supported input shapes:
        - [C, H, W]
        - [1, C, H, W]
    """
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected batch tensor with batch size 1.")
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError("Expected tensor with shape [C, H, W] or [1, C, H, W].")

    mean, std = get_normalization_stats()

    image_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * np.array(std)) + np.array(mean)
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255).astype(np.uint8)

    return image_np


def prepare_uploaded_image(uploaded_file) -> tuple[Image.Image, torch.Tensor]:
    """
    Prepare an uploaded Streamlit image file for inference.

    Returns:
        (pil_image, tensor)

    Raises:
        ValueError: if uploaded file is invalid
    """
    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    image_bytes = uploaded_file.read()
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    image = load_image_from_bytes(image_bytes)
    tensor = pil_to_tensor(image)
    return image, tensor


def get_preprocessing_summary() -> dict[str, object]:
    """
    Return a compact summary of preprocessing settings for display/debugging.
    """
    config = _load_config()
    return {
        "image_size": int(config["image_size"]),
        "mean": list(config["mean"]),
        "std": list(config["std"]),
        "color_mode": "RGB",
        "normalization": True,
    }