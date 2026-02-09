from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from webApp.model.FCNN import FCNN


@dataclass
class ModelService:
    model: torch.nn.Module
    device: torch.device
    order: List[str]
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    threshold: float = 0.3
    model_version: str = "v1"

    @classmethod
    def load(cls, *, weights_path: str, model_version: str = "v1") -> "ModelService":
        # questo modello serve a caricare il modello migliorare del training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights_path, map_location=device) # carico il modello ssalvato in outputs che è un dizionario tipo cosi :
        """
        {
            "model_state_dict": ...,
            "input_dim": 32,
            "hidden_dims": (128,64),
            "dropout": 0.2,
            "batch_norm": False,
            "feature_names": [...],
            "mean": tensor(...),
            "std": tensor(...),
            "threshold": 0.3
            }
        """
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            order = list(ckpt["feature_names"])
            input_dim = int(ckpt["input_dim"])
            hidden_dims = tuple(int(x) for x in ckpt["hidden_dims"])
            dropout = float(ckpt["dropout"])
            batch_norm = bool(ckpt["batch_norm"])
            threshold = float(ckpt.get("threshold", 0.5))

            model = FCNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                batch_norm=batch_norm,
            ).to(device) #crea il modello stesso del modello migliore del training

            model.load_state_dict(ckpt["model_state_dict"]) #carica i pesi migliori
            model.eval()

            mean = ckpt.get("mean", None)
            std = ckpt.get("std", None)
            if mean is not None:
                mean = mean.to(device).float()
            if std is not None:
                std = std.to(device).float()

            return cls(
                model=model,
                device=device,
                order=order,
                mean=mean,
                std=std,
                threshold=threshold,
                model_version=model_version,
            )

    def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
        x = torch.tensor([float(features[name]) for name in self.order],
                         dtype=torch.float32, device=self.device)  # (32,)

        # questo si fa perchè lo facciamo anche nel main.py ed è una standdarizione delle feature
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        return x.unsqueeze(0)  # (1,32) # vettore di input alla rete con le feature rollate

    @torch.no_grad()
    def predict_proba(self, features: Dict[str, float]) -> float:
        x = self._vectorize(features)
        logits = self.model(x)              # (1,1)
        print(logits)
        prob = torch.sigmoid(logits).item() # float in [0,1]
        print(prob)
        return float(prob)

    def predict_label(self, prob: float) -> int:
        return int(prob >= self.threshold)
