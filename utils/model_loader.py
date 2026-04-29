from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from torch import nn

from utils.data_loader import derive_config_path_from_model_path, resolve_model_path


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SUPPORTED_MODEL_TYPES: Final[set[str]] = {"Baseline", "DG", "Hybrid"}
SUPPORTED_DATASET_CONTEXTS: Final[set[str]] = {"ODIR", "RFMiD v1", "RFMiD v2"}


# ---------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------
def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return the best available device.

    Args:
        prefer_gpu: If True, use CUDA when available.

    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def validate_model_type(model_type: str) -> str:
    """Validate and normalize model type."""
    normalized = str(model_type).strip()
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported values: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}"
        )
    return normalized


def validate_dataset_context(dataset_context: str) -> str:
    """Validate and normalize dataset context."""
    normalized = str(dataset_context).strip()
    if normalized not in SUPPORTED_DATASET_CONTEXTS:
        raise ValueError(
            f"Unsupported dataset context: {dataset_context}. "
            f"Supported values: {', '.join(sorted(SUPPORTED_DATASET_CONTEXTS))}"
        )
    return normalized


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------
def load_torch_checkpoint(model_path: Path, map_location: torch.device | str = "cpu") -> object:
    """
    Load a PyTorch checkpoint safely.

    Supports:
    - plain state_dict
    - checkpoint dict containing 'state_dict' or similar
    """
    try:
        checkpoint = torch.load(model_path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from: {model_path}") from exc

    return checkpoint


def extract_state_dict(checkpoint: object) -> dict:
    """
    Extract a usable state_dict from common checkpoint formats.

    Supported:
    - raw state_dict
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - {'model': ...} if it looks like a state dict
    """
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]

        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

        # raw state_dict case
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Could not extract a valid state_dict from the checkpoint. "
        "Please verify the checkpoint structure."
    )


# ---------------------------------------------------------------------
# Model builder integration
# ---------------------------------------------------------------------
def build_model_from_type(model_type: str, dataset_context: str | None = None) -> nn.Module:
    """
    Build an uninitialized model architecture for the selected model type.

    Notes:
    - Baseline -> model_code/baseline_model.py
    - Hybrid   -> model_code/hybrid_model.py
    - DG       -> model_code/dg_model.py
                  DG is dataset-context aware:
                  ODIR -> Mixup
                  RFMiD v1 -> CORAL
                  RFMiD v2 -> DANN
    """
    model_type = validate_model_type(model_type)

    if model_type == "Baseline":
        from model_code import baseline_model as model_module

        for factory_name in ("build_model", "get_model", "create_model"):
            factory = getattr(model_module, factory_name, None)
            if callable(factory):
                model = factory()
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        f"`{factory_name}()` in {model_module.__name__} did not return an nn.Module."
                    )
                return model

    elif model_type == "Hybrid":
        from model_code import hybrid_model as model_module

        for factory_name in ("build_model", "get_model", "create_model"):
            factory = getattr(model_module, factory_name, None)
            if callable(factory):
                model = factory()
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        f"`{factory_name}()` in {model_module.__name__} did not return an nn.Module."
                    )
                return model

    elif model_type == "DG":
        from model_code import dg_model as model_module

        if dataset_context is not None:
            dataset_context = validate_dataset_context(dataset_context)
            dataset_factory = getattr(model_module, "build_model_for_dataset", None)
            if callable(dataset_factory):
                model = dataset_factory(dataset_context)
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        f"`build_model_for_dataset()` in {model_module.__name__} "
                        "did not return an nn.Module."
                    )
                return model

        for factory_name in ("build_model", "get_model", "create_model"):
            factory = getattr(model_module, factory_name, None)
            if callable(factory):
                model = factory()
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        f"`{factory_name}()` in {model_module.__name__} did not return an nn.Module."
                    )
                return model

    raise AttributeError(
        f"No valid model factory found for {model_type}. "
        "Expected one of: build_model(), get_model(), create_model()."
    )


# ---------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------
def load_model_from_path(
    model_path: str | Path,
    model_type: str,
    dataset_context: str | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model from a .pth path.

    Args:
        model_path: Path to .pth file
        model_type: One of Baseline / DG / Hybrid
        dataset_context: Required for DG dataset-aware model routing
        device: torch.device to move model onto
        strict: strict flag for state_dict loading

    Returns:
        nn.Module in eval mode
    """
    path = Path(model_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model_type = validate_model_type(model_type)
    if dataset_context is not None:
        dataset_context = validate_dataset_context(dataset_context)

    resolved_device = device or get_device(prefer_gpu=True)

    model = build_model_from_type(model_type, dataset_context=dataset_context)
    checkpoint = load_torch_checkpoint(path, map_location=resolved_device)
    state_dict = extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load state_dict into {model_type} model from {path.name}. "
            "This usually means the checkpoint architecture and the code architecture do not match."
        ) from exc

    model.to(resolved_device)
    model.eval()
    return model


def load_registered_model(
    model_type: str,
    dataset_context: str,
    device: torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model using model_registry.json.

    Args:
        model_type: Baseline / DG / Hybrid
        dataset_context: ODIR / RFMiD v1 / RFMiD v2

    Returns:
        Loaded nn.Module in eval mode
    """
    model_type = validate_model_type(model_type)
    dataset_context = validate_dataset_context(dataset_context)

    model_path = resolve_model_path(model_type, dataset_context)
    return load_model_from_path(
        model_path=model_path,
        model_type=model_type,
        dataset_context=dataset_context,
        device=device,
        strict=strict,
    )


# ---------------------------------------------------------------------
# Metadata / inspection helpers
# ---------------------------------------------------------------------
def get_model_file_info(model_type: str, dataset_context: str) -> dict[str, str]:
    """
    Return model + config path info for display/debugging.
    """
    model_type = validate_model_type(model_type)
    dataset_context = validate_dataset_context(dataset_context)

    model_path = resolve_model_path(model_type, dataset_context)
    config_path = derive_config_path_from_model_path(model_path)

    return {
        "model_type": model_type,
        "dataset_context": dataset_context,
        "model_path": str(model_path),
        "config_path": str(config_path),
    }


def warmup_model(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device | None = None,
) -> None:
    """
    Optional forward warmup to reduce first-inference delay.

    Safe to skip if not needed.
    """
    resolved_device = device or next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=resolved_device)

    with torch.no_grad():
        output = model(dummy)
        if isinstance(output, (tuple, list)):
            _ = output[0]
        else:
            _ = output


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def get_cache_key(model_type: str, dataset_context: str) -> str:
    """Generate a stable cache key."""
    return f"{validate_model_type(model_type)}::{validate_dataset_context(dataset_context)}"