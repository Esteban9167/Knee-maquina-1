"""Volumen MRI → bloques 2.5D 224×224 normalizados ImageNet."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import FEATURE_DIM, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .dicom import leer_corte


def serie_a_volumen(serie: dict) -> np.ndarray:
    imagenes = [leer_corte(c["path"]) for c in serie["dicoms"]]
    if not imagenes:
        return np.zeros((0, 1, 1), dtype=np.float32)
    h, w = imagenes[0].shape[-2], imagenes[0].shape[-1]
    alineadas = []
    for img in imagenes:
        if img.shape != (h, w):
            t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            img = t.squeeze().numpy()
        alineadas.append(img)
    return np.stack(alineadas, axis=0).astype(np.float32)


def normalizar_volumen(volumen: np.ndarray) -> np.ndarray:
    if volumen.size == 0:
        return volumen
    p1, p99 = np.percentile(volumen, 1), np.percentile(volumen, 99)
    volumen = np.clip(volumen, p1, p99)
    return (volumen - p1) / (p99 - p1 + 1e-8)


def crear_25d(volumen: np.ndarray) -> np.ndarray:
    n = len(volumen)
    if n < 3:
        return np.zeros((0, 3, *volumen.shape[1:]), dtype=np.float32)
    bloques = [
        np.stack([volumen[i - 1], volumen[i], volumen[i + 1]], axis=0)
        for i in range(1, n - 1)
    ]
    return np.stack(bloques, axis=0)


def redimensionar_bloque(bloque: np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = torch.tensor(bloque, dtype=torch.float32).unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x.squeeze(0) - mean) / std


# reexport for callers that only import preprocess
__all__ = [
    "FEATURE_DIM",
    "IMAGE_SIZE",
    "serie_a_volumen",
    "normalizar_volumen",
    "crear_25d",
    "redimensionar_bloque",
]
