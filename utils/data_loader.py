from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
UTILS_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = UTILS_DIR.parent

DATA_DIR: Final[Path] = ROOT_DIR / "data"
METRICS_DIR: Final[Path] = DATA_DIR / "metrics"
METADATA_DIR: Final[Path] = DATA_DIR / "metadata"

ASSETS_DIR: Final[Path] = ROOT_DIR / "assets"
CONFUSION_DIR: Final[Path] = ASSETS_DIR / "confusion_matrices"
CURVES_DIR: Final[Path] = ASSETS_DIR / "curves"
ROC_PR_DIR: Final[Path] = ASSETS_DIR / "roc_pr"
DIAGRAMS_DIR: Final[Path] = ASSETS_DIR / "diagrams"

MODELS_DIR: Final[Path] = ROOT_DIR / "models"
CONFIGS_DIR: Final[Path] = ROOT_DIR / "configs"


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------
def file_exists(path: Path) -> bool:
    """Return True if a path exists and is a file."""
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def dir_exists(path: Path) -> bool:
    """Return True if a path exists and is a directory."""
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def ensure_file(path: Path, label: str | None = None) -> Path:
    """
    Validate that a file exists and return the path.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not file_exists(path):
        display_name = label or str(path)
        raise FileNotFoundError(f"Required file not found: {display_name}")
    return path


# ---------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and return a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid or not a dictionary.
    """
    ensure_file(path)

    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in file: {path}")

    return data


def load_class_labels() -> list[str]:
    """
    Load the class label list from metadata.

    Expected file:
        data/metadata/class_labels.json

    Expected schema:
        { "classes": ["Normal", "Diabetic Retinopathy", ...] }
    """
    path = METADATA_DIR / "class_labels.json"
    data = load_json(path)

    classes = data.get("classes")
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(
            "Invalid `class_labels.json`: expected key `classes` with a list of strings."
        )

    return classes


def load_preprocessing_config() -> dict[str, Any]:
    """
    Load preprocessing configuration from metadata.

    Expected file:
        data/metadata/preprocessing_config.json
    """
    path = METADATA_DIR / "preprocessing_config.json"
    data = load_json(path)

    required_keys = {"image_size", "mean", "std"}
    missing = required_keys.difference(data.keys())
    if missing:
        raise ValueError(
            "Invalid `preprocessing_config.json`: missing keys "
            + ", ".join(sorted(missing))
        )

    return data


def load_model_registry() -> dict[str, Any]:
    """
    Load model registry from metadata.

    Expected file:
        data/metadata/model_registry.json
    """
    path = METADATA_DIR / "model_registry.json"
    data = load_json(path)

    required_top_keys = {"Baseline", "DG", "Hybrid"}
    missing = required_top_keys.difference(data.keys())
    if missing:
        raise ValueError(
            "Invalid `model_registry.json`: missing model groups "
            + ", ".join(sorted(missing))
        )

    return data


# ---------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file safely.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If parsing fails.
    """
    ensure_file(path)

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV file: {path}") from exc

    return df


def validate_columns(df: pd.DataFrame, required_columns: set[str], file_label: str) -> pd.DataFrame:
    """
    Validate required columns in a DataFrame and return the DataFrame unchanged.
    """
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


def load_in_domain_results() -> pd.DataFrame:
    """Load and validate the combined in-domain results CSV."""
    path = METRICS_DIR / "in_domain_results.csv"
    df = load_csv(path)

    required = {"dataset", "model", "accuracy", "f1", "auc", "precision", "recall"}
    return validate_columns(df, required, "in_domain_results.csv")


def load_lodo_results() -> pd.DataFrame:
    """Load and validate the baseline LODO results CSV."""
    path = METRICS_DIR / "lodo_results.csv"
    df = load_csv(path)

    required = {"held_out_dataset", "model", "accuracy", "f1", "auc", "precision", "recall"}
    return validate_columns(df, required, "lodo_results.csv")


def load_hybrid_results() -> pd.DataFrame:
    """Load and validate the hybrid results CSV."""
    path = METRICS_DIR / "hybrid_results.csv"
    df = load_csv(path)

    required = {"held_out_dataset", "model", "accuracy", "f1", "auc", "precision", "recall"}
    return validate_columns(df, required, "hybrid_results.csv")


def load_comparison_results() -> pd.DataFrame:
    """
    Load comparison results.

    This function intentionally supports either:
    - wide format: dataset, baseline_acc, dg_acc, hybrid_acc
    - long format: dataset, model, accuracy, ...
    """
    path = METRICS_DIR / "comparison_results.csv"
    df = load_csv(path)

    wide_required = {"dataset", "baseline_acc", "dg_acc", "hybrid_acc"}
    long_required = {"dataset", "model", "accuracy"}

    if wide_required.issubset(df.columns):
        return df

    if long_required.issubset(df.columns):
        return df

    raise ValueError(
        "comparison_results.csv must be either wide format "
        "(dataset, baseline_acc, dg_acc, hybrid_acc) "
        "or long format (dataset, model, accuracy, ...)."
    )


# ---------------------------------------------------------------------
# Registry-based model/config path helpers
# ---------------------------------------------------------------------
def resolve_model_path(model_type: str, dataset_context: str) -> Path:
    """
    Resolve model path from model registry.

    Args:
        model_type: One of "Baseline", "DG", "Hybrid"
        dataset_context: One of "ODIR", "RFMiD v1", "RFMiD v2"

    Returns:
        Absolute model file path.
    """
    registry = load_model_registry()

    if model_type not in registry:
        raise KeyError(f"Unknown model type: {model_type}")

    dataset_map = registry[model_type]
    if dataset_context not in dataset_map:
        raise KeyError(
            f"Dataset context `{dataset_context}` not found under model type `{model_type}`."
        )

    relative_path = Path(dataset_map[dataset_context])
    absolute_path = ROOT_DIR / relative_path

    return ensure_file(
        absolute_path,
        label=f"{model_type} model for {dataset_context}",
    )


def derive_config_path_from_model_path(model_path: Path) -> Path:
    """
    Derive config path from a model path by replacing:
    - root folder: models -> configs
    - suffix: .pth -> _config.json

    Example:
        models/hybrid/hybrid_lodo_test_on_odir.pth
        -> configs/hybrid/hybrid_lodo_test_on_odir_config.json
    """
    try:
        relative_model_path = model_path.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"Model path must be inside project root: {model_path}") from exc

    parts = list(relative_model_path.parts)
    if not parts or parts[0] != "models":
        raise ValueError(f"Expected model path under `models/`: {model_path}")

    parts[0] = "configs"
    config_relative = Path(*parts).with_suffix("")
    config_relative = config_relative.parent / f"{config_relative.name}_config.json"

    config_path = ROOT_DIR / config_relative
    return ensure_file(config_path, label=f"Config for model `{model_path.name}`")


# ---------------------------------------------------------------------
# Asset path helpers
# ---------------------------------------------------------------------
def dataset_to_asset_key(dataset_name: str) -> str:
    """
    Convert dataset display name to asset suffix.

    Supported:
        ODIR -> odir
        RFMiD v1 -> rfmid_v1
        RFMiD v2 -> rfmid_v2
    """
    mapping = {
        "ODIR": "odir",
        "RFMiD v1": "rfmid_v1",
        "RFMiD v2": "rfmid_v2",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return mapping[dataset_name]


def get_confusion_matrix_path(model_type: str, dataset_name: str, per_class: bool = False) -> Path:
    """
    Get confusion matrix asset path for a model type and dataset.
    """
    model_key = model_type.strip().lower()
    dataset_key = dataset_to_asset_key(dataset_name)

    if per_class:
        filename = f"cm_{model_key}_per_class_{dataset_key}.png"
    else:
        filename = f"cm_{model_key}_{dataset_key}.png"

    return CONFUSION_DIR / filename


def get_curve_path(model_type: str, dataset_name: str, metric_name: str) -> Path:
    """
    Get a curve asset path.

    Example metric names:
        loss, mauc, map, lr
    """
    model_key = model_type.strip().lower()
    dataset_key = dataset_to_asset_key(dataset_name)
    filename = f"curve_{model_key}_{metric_name}_{dataset_key}.png"
    return CURVES_DIR / filename


def get_roc_pr_path(model_type: str, dataset_name: str, asset_type: str) -> Path:
    """
    Get ROC/PR related asset path.

    Supported asset_type values:
        roc, pr, auc_ap, f1_pr_rc
    """
    model_key = model_type.strip().lower()
    dataset_key = dataset_to_asset_key(dataset_name)
    filename = f"{asset_type}_{model_key}_{dataset_key}.png"
    return ROC_PR_DIR / filename


def get_hybrid_diagnostics_path(dataset_name: str) -> Path:
    """
    Get hybrid diagnostics diagram path.
    """
    dataset_key = dataset_to_asset_key(dataset_name)
    filename = f"hybrid_diagnostics_{dataset_key}.png"
    return DIAGRAMS_DIR / filename