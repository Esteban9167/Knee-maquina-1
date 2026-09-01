# RSNA Knee — producción

Paquete listo para correr el pipeline de `chamba` (2.5D + DenseNet121 + Attention → 12 labels).

No incluye exploración, Grad-CAM ni entrenamiento masivo.

## Contenido

```text
produccion/
  rsna_knee/     código
  chamba.ipynb   notebook mínimo
  requirements.txt
  .env.example
```

## Local

```bash
cd produccion
python -m pip install -r requirements.txt
cp .env.example .env   # ajusta BASE_PATH
```

Abre `chamba.ipynb` con el venv. Disco `ADATA HD680` montado.

## Kaggle

Sube esta carpeta (o el notebook + `rsna_knee/`). Add data: competencia RSNA + dataset `train_cursor`. GPU on.

## Uso

```python
from rsna_knee import cargar_tablas, cargar_estudio, KneePipeline, device

train_df, series_df, labels, train_images, comp = cargar_tablas()
pipe = KneePipeline(n_labels=12).to(device())
estudio = cargar_estudio(train_df.iloc[0]["StudyInstanceUID"], series_df, train_images)
logits = pipe(estudio)
```
