# Kaggle

Sube **código**, no el dataset ni `.venv` ni `.env`.

## Qué subir

- `Knee.ipynb` (mejor sin outputs)
- carpeta `src/rsna_knee/`
- `requirements.txt` (Kaggle ya trae pandas/torch; sirve de referencia)

No subas `.env`, `.venv/` ni archivos `.dcm`.

## Add data

1. Competencia **RSNA Knee** (trae `train.csv`, `train_series.csv`, `train_series/`).
2. Dataset `train_cursor` (el CSV de 12 etiquetas).
3. GPU **on**.

## Rutas

El paquete busca primero `/kaggle/input`:

- competencia: carpeta con `train.csv` + `train_series.csv`
- etiquetas: `train_cursor.csv` en un dataset cuyo nombre contenga `cursor`

No hace falta `.env` en Kaggle.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from rsna_knee import cargar_tablas, cargar_estudio, KneePipeline, device

train_df, series_df, labels, train_images, comp = cargar_tablas()
pipe = KneePipeline(n_labels=12).to(device())
```
