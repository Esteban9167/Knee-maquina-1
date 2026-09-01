"""Rutas, constantes y dispositivo."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

IMAGE_SIZE = 224
FEATURE_DIM = 1024
PLANOS = ("Sagittal", "Coronal", "Axial")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

KAGGLE_INPUT = Path("/kaggle/input")
LOCAL_COMP = Path("/Volumes/ADATA HD680/KAGGLE_RSNA/dataset_extraido")
LOCAL_LABELS = Path("/Volumes/ADATA HD680/KAGGLE_RSNA/train_cursor.csv")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def competencia() -> Path:
    if KAGGLE_INPUT.exists():
        for p in sorted(KAGGLE_INPUT.iterdir()):
            if p.is_dir() and (p / "train.csv").exists() and (p / "train_series.csv").exists():
                return p
        raise FileNotFoundError("Añade la competencia RSNA Knee en Kaggle.")
    load_dotenv()
    extraido = Path(os.environ.get("BASE_PATH", LOCAL_COMP.parent)) / "dataset_extraido"
    if extraido.exists() and (extraido / "train.csv").exists():
        return extraido
    if LOCAL_COMP.exists():
        return LOCAL_COMP
    raise FileNotFoundError("No está dataset_extraido ni /kaggle/input.")


def etiquetas_csv(comp: Path) -> Path:
    if KAGGLE_INPUT.exists():
        for p in KAGGLE_INPUT.iterdir():
            if not p.is_dir():
                continue
            cursor = p / "train_cursor.csv"
            if cursor.exists():
                return cursor
            if "cursor" in p.name.lower():
                csvs = list(p.glob("*.csv"))
                if csvs:
                    return csvs[0]
    load_dotenv()
    env_csv = Path(os.environ.get("BASE_PATH", LOCAL_LABELS.parent)) / os.environ.get(
        "TRAIN_CSV", "train_cursor.csv"
    )
    if env_csv.exists():
        return env_csv
    if LOCAL_LABELS.exists():
        return LOCAL_LABELS
    return comp / "train.csv"


def imagenes_train(comp: Path) -> Path:
    return comp / "train_series"
