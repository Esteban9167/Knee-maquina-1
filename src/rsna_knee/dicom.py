"""Lectura y orden anatómico de DICOM. Conserva todas las series (0/0 y 1/1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

from .config import PLANOS


def posicion_corte(ds) -> float:
    try:
        orientacion = np.asarray(ds.ImageOrientationPatient, dtype=np.float32)
        posicion = np.asarray(ds.ImagePositionPatient, dtype=np.float32)
        normal = np.cross(orientacion[:3], orientacion[3:])
        return float(np.dot(posicion, normal))
    except Exception:
        if hasattr(ds, "SliceLocation"):
            try:
                return float(ds.SliceLocation)
            except Exception:
                pass
        return float(getattr(ds, "InstanceNumber", 0))


def _a_2d(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(np.asarray(img))
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] <= 4:
            return img[..., 0]
        return img[img.shape[0] // 2]
    raise ValueError(f"Forma MRI inesperada: {img.shape}")


def _pixel_array_robusto(ds) -> np.ndarray:
    try:
        return np.asarray(ds.pixel_array)
    except Exception:
        pass
    rows, cols = int(ds.Rows), int(ds.Columns)
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    raw = bytes(ds.PixelData)
    n_pix = rows * cols * samples
    arr = np.frombuffer(raw, dtype=np.uint8, count=min(len(raw), n_pix))
    if arr.size < n_pix:
        arr = np.pad(arr, (0, n_pix - arr.size))
    arr = arr[:n_pix]
    if samples == 1:
        return arr.reshape(rows, cols)
    return arr.reshape(rows, cols, samples)[..., 0]


def leer_corte(path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    img = _a_2d(_pixel_array_robusto(ds)).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    return img


def cargar_serie(row, train_images: Path) -> dict:
    study_uid = str(row["StudyInstanceUID"])
    series_uid = str(row["SeriesInstanceUID"])
    serie_dir = train_images / study_uid / series_uid
    cortes = []
    if serie_dir.exists():
        for archivo in serie_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(archivo, stop_before_pixels=True)
                cortes.append({"path": archivo, "position": posicion_corte(ds)})
            except Exception as e:
                print("Error header:", archivo.name, e)
    cortes = sorted(cortes, key=lambda x: x["position"])
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "Anatomical_Plane": row["Anatomical_Plane"],
        "Fluid_Sensitive": int(row["Fluid_Sensitive"]),
        "Fat_Suppression": int(row["Fat_Suppression"]),
        "dicoms": cortes,
    }


def cargar_estudio(study_uid: str, series_df, train_images: Path) -> dict:
    study_uid = str(study_uid)
    filas = series_df[series_df["StudyInstanceUID"] == study_uid]
    estudio = {plano: [] for plano in PLANOS}
    for _, row in filas.iterrows():
        plano = row["Anatomical_Plane"]
        if plano in estudio:
            estudio[plano].append(cargar_serie(row, train_images))
    return estudio
