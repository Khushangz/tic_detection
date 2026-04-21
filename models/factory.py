from bilstm import BiLSTMClassifier
from cnn_bilstm import CNNBiLSTMClassifier


def get_model(
    model_cfg: dict,
    num_classes: int,
    input_dim: int,
):
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

    elif model_type == "cnn_bilstm":
        cfg = model_cfg["bilstm"]
        return CNNBiLSTMClassifier(
            input_dim   = input_dim,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
            num_classes = num_classes,
            cnn_dropout = cfg.get("cnn_dropout", 0.1),
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: bilstm, cnn_bilstm"
        )