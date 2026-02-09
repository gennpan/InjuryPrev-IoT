from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """
    MLP (fully-connected) per classificazione binaria con output logits (1 neurone).

    Usare con BCEWithLogitsLoss (niente sigmoid nel forward).
    """

    def __init__(
            self,
            input_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64,32,10),
            dropout: float = 0.2,
            batch_norm: bool = False,
    ):
        super().__init__()

        if int(input_dim) <= 0:
            raise ValueError("input_dim deve essere > 0")
        if any(int(h) <= 0 for h in hidden_dims):
            raise ValueError("hidden_dims deve contenere soli valori > 0")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout deve essere in [0, 1)")

        layers: list[nn.Module] = []
        prev = int(input_dim)

        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h

        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_dims(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return ()
        return tuple(int(p) for p in parts)
    return tuple(int(v) for v in value)
