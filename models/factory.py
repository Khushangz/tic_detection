"""
models/factory.py
-----------------
Model factory — returns the correct model instance based on
model_type in model.yaml.

Usage:
    from factory import get_model
    model = get_model(model_cfg, num_classes=72, input_dim=768)

Supported models:
    bilstm        BiLSTM classifier
    cnn_bilstm    CNN + BiLSTM classifier
    linear_probe  Single linear layer baseline
"""

from bilstm import BiLSTMClassifier
from cnn_bilstm import CNNBiLSTMClassifier
from linear_probe import LinearProbe


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
    cfg        = model_cfg.get("bilstm", {})

    if model_type == "bilstm":
        return BiLSTMClassifier(
            input_dim   = input_dim,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
            num_classes = num_classes,
        )

    elif model_type == "cnn_bilstm":
        return CNNBiLSTMClassifier(
            input_dim   = input_dim,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
            num_classes = num_classes,
            cnn_dropout = cfg.get("cnn_dropout", 0.1),
        )

    elif model_type == "linear_probe":
        return LinearProbe(
            input_dim   = input_dim,
            num_classes = num_classes,
            dropout     = cfg.get("dropout", 0.0),
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: bilstm, cnn_bilstm, linear_probe"
        )