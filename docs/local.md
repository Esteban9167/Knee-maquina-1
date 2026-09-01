# Local

El dataset no va en git: vive en el disco `ADATA HD680`.

```bash
cd produccion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# opcional: instalar el paquete en modo editable
python -m pip install -e .
cp config/env.example .env   # ajusta BASE_PATH si el disco montó en otra ruta
```

Kernel del notebook: el `.venv` de esta carpeta (no el `python3` del sistema).

Estructura esperada en `BASE_PATH`:

```text
KAGGLE_RSNA/
  dataset_extraido/
    train.csv
    train_series.csv
    train_series/          # StudyUID / SeriesUID / *.dcm
  train_cursor.csv
  train_series.csv
```

Abre `Knee.ipynb` con el disco montado. El paquete detecta las mismas rutas:

```python
from rsna_knee import cargar_tablas, cargar_estudio, KneePipeline, device

train_df, series_df, labels, train_images, comp = cargar_tablas()
pipe = KneePipeline(n_labels=12).to(device())
estudio = cargar_estudio(train_df.iloc[0]["StudyInstanceUID"], series_df, train_images)
logits = pipe(estudio)
```
