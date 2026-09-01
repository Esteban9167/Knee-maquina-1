"""DenseNet121 + Attention + cabeza 12 labels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from .config import FEATURE_DIM, PLANOS
from .preprocess import crear_25d, normalizar_volumen, redimensionar_bloque, serie_a_volumen


class AttentionCortes(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        pesos = torch.softmax(self.score(x), dim=0)
        return (pesos * x).sum(dim=0), pesos.squeeze(-1)


class AttentionSeries(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        pesos = torch.softmax(self.score(x), dim=0)
        return (pesos * x).sum(dim=0), pesos.squeeze(-1)


class CabezaMultiplanar(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM, n_labels: int = 12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_labels),
        )

    def forward(self, sag, cor, axi):
        return self.mlp(torch.cat([sag, cor, axi], dim=-1))


class KneePipeline(nn.Module):
    """Un estudio → 12 logits. Backbone ImageNet congelado (validar pipeline)."""

    def __init__(self, n_labels: int = 12, freeze_backbone: bool = True):
        super().__init__()
        densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.extractor = densenet.features
        if freeze_backbone:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False
        self.attn_cortes = AttentionCortes()
        self.attn_series = AttentionSeries()
        self.cabeza = CabezaMultiplanar(n_labels=n_labels)

    def extraer_features(self, batch: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(batch)
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)

    def procesar_serie(self, serie: dict):
        device = next(self.parameters()).device
        bloques = crear_25d(normalizar_volumen(serie_a_volumen(serie)))
        if len(bloques) == 0:
            z = torch.zeros(FEATURE_DIM, device=device)
            return z, torch.zeros(0, device=device)
        batch = torch.stack([redimensionar_bloque(b) for b in bloques]).to(device)
        with torch.no_grad():
            feats = self.extraer_features(batch)
        return self.attn_cortes(feats)

    def procesar_plano(self, series_plano: list):
        device = next(self.parameters()).device
        if not series_plano:
            z = torch.zeros(FEATURE_DIM, device=device)
            return z, torch.zeros(0, device=device)
        feats = [self.procesar_serie(s)[0] for s in series_plano]
        return self.attn_series(torch.stack(feats, dim=0))

    def features_estudio(self, estudio: dict):
        outs = {}
        pesos = {}
        for plano in PLANOS:
            f, w = self.procesar_plano(estudio[plano])
            outs[plano] = f
            pesos[plano] = w
        return outs, pesos

    def forward(self, estudio: dict) -> torch.Tensor:
        f, _ = self.features_estudio(estudio)
        return self.cabeza(f["Sagittal"], f["Coronal"], f["Axial"])
