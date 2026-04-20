"""
models/factory.py
-----------------
Model factory — returns the correct model instance based on
model_type in model.yaml.

Usage:
    from factory import get_model
    model = get_model(model_cfg, num_classes=72, input_dim=768)
"""

from bilstm import BiLSTMClassifier


def get_model(
    model_cfg: dict,
    num_classes: int,
    input_dim: int,
):
    """
    Build and return a model based on model_cfg.

    Args:
        model_cfg:   loaded model.yaml dict
        num_classes: number of output classes
        input_dim:   embedding dimension (768 for WavLM base)

    Returns:
        model instance (subclass of BaseClassifier)
    """
    model_type = model_cfg.get("model_type", "bilstm")

    if model_type == "bilstm":
        cfg = model_cfg["bilstm"]
        return BiLSTMClassifier(
            input_dim   = input_dim,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
            num_classes = num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: bilstm"
        )