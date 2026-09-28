from __future__ import annotations
import torch.nn as nn
from typing import Optional
from config import Config, ModelType
from model import PriorTransformer, ControlTransformer, TransformerTranslation


def model_from_config(config: Config, is_prior: bool) -> tuple[nn.Module, Optional[nn.Module]]:
    if config["model_type"] == ModelType.TRANSFORMER:
        if is_prior:
            model = PriorTransformer(
                input_T=config["input_T"],
                output_dim=config["output_dim"],
                hidden_dim=config["hidden_dim"],
                n_hidden=config["num_layers"],
                n_heads=config["transformer_n_heads"],
                d_ff=config["transformer_d_ff"],
                dropout=config["transformer_dropout"],
                diffusion_T=config["diffusion_T"],
            )
        else:
            model = ControlTransformer(
                input_T=config["input_T"],
                c_dim=config["input_dim"],
                output_dim=config["output_dim"],
                hidden_dim=config["hidden_dim"],
                n_hidden=config["num_layers"],
                n_heads=config["transformer_n_heads"],
                d_ff=config["transformer_d_ff"],
                dropout=config["transformer_dropout"],
                # --no-imu: 38-dim insole, IMU condition embeddings deleted.
                imu_available=bool(config.get("imu_available", True)),
            )
        model_translation = None
        # The translation model's sole condition is the IMU channels, so it is
        # only built when the config trains it (--no-imu configs set
        # "train_translation": false and evaluate pose only).
        if not is_prior and config.get("train_translation", True):
            model_translation = TransformerTranslation(
                c_dim=config["input_dim"],
                input_T=config["input_T"],
                output_dim=config["output_dim"],
                hidden_dim=config["hidden_dim"],
                n_hidden=1,
                n_heads=config["transformer_n_heads"],
                d_ff=config["transformer_d_ff"],
                dropout=config["transformer_dropout"],
                is_prior=is_prior,
            )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    return model, model_translation
