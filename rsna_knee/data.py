"""Carga de tablas Kaggle / disco."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import competencia, etiquetas_csv, imagenes_train


def cargar_tablas():
    comp = competencia()
    train_images = imagenes_train(comp)
    if not train_images.exists():
        raise FileNotFoundError(train_images)

    train_df = pd.read_csv(etiquetas_csv(comp))
    series_df = pd.read_csv(comp / "train_series.csv")
    train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)
    series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].astype(str)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    labels = [c for c in train_df.columns if c not in ["StudyInstanceUID", "Report"]]
    if len(labels) != 12:
        raise ValueError(f"Se esperaban 12 etiquetas, hay {labels}")

    return train_df, series_df, labels, train_images, comp
