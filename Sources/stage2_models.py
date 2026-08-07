# Sources/stage2_models.py
# Stage-2 model: geometry/material parameters -> latent vector z

from __future__ import annotations

import torch
import torch.nn as nn


class FNNStage2(nn.Module):
    """
    Stage-2 FNN/MLP:
        input : normalized geometry/material vector [B, P]
        output: normalized latent vector [B, latent_dim]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 60,
        hidden_dims=(512, 512, 512, 512, 512, 256, 256),
        activation: str = "gelu",
        dropout: float = 0.06,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        act = activation.lower()

        if act == "relu":
            Act = nn.ReLU
        elif act == "gelu":
            Act = nn.GELU
        elif act == "tanh":
            Act = nn.Tanh
        elif act == "silu":
            Act = nn.SiLU
        else:
            raise ValueError("activation must be one of: relu, gelu, tanh, silu")

        layers = []
        prev = int(input_dim)

        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(int(h)))

            layers.append(Act())

            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))

            prev = int(h)

        layers.append(nn.Linear(prev, int(latent_dim)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_stage2_model(
    model_type: str,
    input_dim: int,
    latent_dim: int = 60,
    **kwargs,
) -> nn.Module:
    model_type = model_type.lower()

    if model_type in ["fnn", "mlp", "ann"]:
        return FNNStage2(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=kwargs.get("hidden_dims", (512, 512, 512, 512, 512, 256, 256)),
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.06),
            use_batchnorm=kwargs.get("use_batchnorm", False),
        )

    raise ValueError(f"Unknown model_type='{model_type}'. Use 'fnn', 'mlp', or 'ann'.")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))