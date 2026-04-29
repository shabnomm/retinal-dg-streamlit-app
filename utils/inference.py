from __future__ import annotations

from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.data_loader import load_class_labels


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DEFAULT_TOP_K: Final[int] = 3


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def validate_model(model: nn.Module) -> None:
    """Ensure model is a valid PyTorch module."""
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be a torch.nn.Module instance.")

    if not hasattr(model, "forward"):
        raise TypeError("Model does not have a valid forward method.")


def validate_tensor(input_tensor: torch.Tensor) -> None:
    """Ensure tensor shape is valid."""
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if input_tensor.ndim != 4:
        raise ValueError(
            f"Expected input tensor with shape [B, C, H, W], got shape {tuple(input_tensor.shape)}"
        )

    if input_tensor.shape[0] != 1:
        raise ValueError("Batch size must be 1 for inference.")


# ---------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------
def forward_pass(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Perform forward pass and return raw logits.
    """
    validate_model(model)
    validate_tensor(input_tensor)

    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Some models return tuple (logits, aux)
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Model output is not a tensor.")

    return output


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities using softmax.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [B, num_classes], got {tuple(logits.shape)}")

    probs = F.softmax(logits, dim=1)
    return probs


def get_top_k(
    probabilities: torch.Tensor,
    class_labels: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Extract top-k predictions.

    Returns:
        [
            {"class": "...", "confidence": 0.9234},
            ...
        ]
    """
    if probabilities.ndim != 2:
        raise ValueError("Expected probability tensor with shape [B, num_classes]")

    probs = probabilities[0]  # batch size = 1

    num_classes = probs.shape[0]
    k = min(top_k, num_classes)

    values, indices = torch.topk(probs, k=k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(values, indices):
        class_idx = int(idx.item())
        class_name = class_labels[class_idx] if class_idx < len(class_labels) else f"Class {class_idx}"

        results.append(
            {
                "class": class_name,
                "confidence": float(score.item()),
            }
        )

    return results


def predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Full prediction pipeline.

    Returns:
        {
            "predicted_class": "...",
            "confidence": float,
            "top_k": [...],
            "probabilities": np.ndarray
        }
    """
    class_labels = load_class_labels()

    logits = forward_pass(model, input_tensor)
    probabilities = logits_to_probabilities(logits)

    top_predictions = get_top_k(probabilities, class_labels, top_k=top_k)

    predicted_class = top_predictions[0]["class"]
    confidence = top_predictions[0]["confidence"]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_predictions,
        "probabilities": probabilities.detach().cpu().numpy()[0],
    }


# ---------------------------------------------------------------------
# Pretty formatting helpers (for Streamlit)
# ---------------------------------------------------------------------
def format_confidence(value: float) -> str:
    """Format confidence as percentage string."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def build_prediction_table(prediction_output: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert prediction output into table-friendly format.
    """
    table_rows = []
    for item in prediction_output["top_k"]:
        table_rows.append(
            {
                "Class": item["class"],
                "Confidence": format_confidence(item["confidence"]),
            }
        )
    return table_rows


def get_probability_distribution(
    prediction_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert probabilities into class-wise distribution.
    Useful for bar charts.
    """
    class_labels = load_class_labels()
    probs = prediction_output["probabilities"]

    distribution = []
    for idx, prob in enumerate(probs):
        label = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
        distribution.append(
            {
                "class": label,
                "probability": float(prob),
            }
        )

    return distribution


# ---------------------------------------------------------------------
# Optional debug helpers
# ---------------------------------------------------------------------
def debug_prediction_output(prediction_output: dict[str, Any]) -> None:
    """
    Print debug-friendly output in console/log.
    """
    print("Predicted class:", prediction_output["predicted_class"])
    print("Confidence:", prediction_output["confidence"])
    print("Top-k predictions:")
    for item in prediction_output["top_k"]:
        print(f"  - {item['class']}: {item['confidence']:.4f}")