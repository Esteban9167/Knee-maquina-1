# RSNA Knee — producción

Pipeline **2.5D + DenseNet121 + Attention → 12 labels**. El notebook `Knee.ipynb` arma y valida el modelo; el paquete en `src/` es el mismo código para reutilizar o subir a Kaggle.

No incluye exploración, Grad-CAM ni entrenamiento masivo.

## Carpetas

```text
produccion/
  Knee.ipynb              notebook
  requirements.txt
  pyproject.toml
  config/
    env.example           plantilla de rutas locales (cópiala a .env)
  src/
    rsna_knee/            paquete (DICOM, preprocess, modelo)
  docs/
    local.md
    kaggle.md
```

`.env`, `.venv/` y los DICOM **no** van a git.

## Local

Ver [`docs/local.md`](docs/local.md).

```bash
cp config/env.example .env
python -m pip install -r requirements.txt
```

Disco `ADATA HD680` montado. Abre `Knee.ipynb` con el kernel del `.venv`.

## Kaggle

Ver [`docs/kaggle.md`](docs/kaggle.md). Sube notebook + `src/rsna_knee/`. Add data: competencia RSNA + `train_cursor`. GPU on.

## Uso del paquete

```python
from rsna_knee import cargar_tablas, cargar_estudio, KneePipeline, device

train_df, series_df, labels, train_images, comp = cargar_tablas()
pipe = KneePipeline(n_labels=12).to(device())
estudio = cargar_estudio(train_df.iloc[0]["StudyInstanceUID"], series_df, train_images)
logits = pipe(estudio)
```
