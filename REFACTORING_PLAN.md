# Refactoring Plan

Questo documento illustra come migliorare la modularità di `app/server.py` spostando le funzioni di elaborazione pure in un nuovo modulo `app/core_logic.py`.

## Funzioni da spostare

Le seguenti funzioni non dipendono direttamente da Flask e possono essere riutilizzate in altri contesti (es. Celery, test unitari). Verranno quindi trasferite in `app/core_logic.py`:

- `release_vram`
- `normalize_image`
- `ensure_yolo_parser_is_loaded`
- `ensure_sam_predictor_is_loaded`
- `ensure_pipeline_is_loaded`
- `ensure_face_analyzer_is_loaded`
- `ensure_face_swapper_is_loaded`
- `ensure_face_restorer_is_loaded`
- `process_generate_all_parts`
- `process_create_scene`
- `process_detail_and_upscale`
- `process_final_swap`

Le route Flask e i task Celery continueranno a risiedere in `server.py`, ma richiameranno la logica importata da `core_logic.py`.

## Esempio di funzioni con docstring

Di seguito sono riportate due funzioni con le nuove docstring e i commenti esplicativi. Queste versioni saranno collocate in `app/core_logic.py`.

```python
from PIL import Image
import io
import cv2
import numpy as np

# ... altre importazioni utili ...

def process_final_swap(target_bytes: bytes, source_bytes: bytes,
                       source_idx: int, target_idx: int,
                       progress_cb=None) -> Image.Image:
    """Esegue il face swap tra due immagini.

    Args:
        target_bytes (bytes): Immagine di destinazione.
        source_bytes (bytes): Immagine sorgente da cui prelevare il volto.
        source_idx (int): Indice del volto sorgente da usare.
        target_idx (int): Indice del volto da sostituire nella destinazione.
        progress_cb (Callable[[int], None] | None): callback per l'avanzamento.

    Returns:
        Image.Image: Immagine risultante con il volto sostituito.
    """
    ensure_face_analyzer_is_loaded()
    ensure_face_swapper_is_loaded()
    ensure_face_restorer_is_loaded()
    target_pil = normalize_image(Image.open(io.BytesIO(target_bytes)).convert("RGB"))
    source_pil = normalize_image(Image.open(io.BytesIO(source_bytes)).convert("RGB"))
    target_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
    source_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
    target_faces = face_analyzer.get(target_cv)
    source_faces = face_analyzer.get(source_cv)
    if not target_faces or not source_faces:
        raise ValueError("Volti non trovati")
    if source_idx >= len(source_faces) or target_idx >= len(target_faces):
        raise IndexError("Indice del volto non valido")
    result_img = face_swapper.get(target_cv, target_faces[target_idx],
                                  source_faces[source_idx], paste_back=True)
    if progress_cb:
        progress_cb(50)
    if face_restorer:
        _, _, result_img = face_restorer.enhance(
            result_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.8,
        )
    if progress_cb:
        progress_cb(100)
    return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))


def detect_faces() -> Response:
    """Rileva i volti nell'immagine inviata e applica un padding proporzionale."""
    # ... codice iniziale di caricamento immagine ...
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = [int(c) for c in f.bbox]
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        pad_top_percent = 0.10
        pad_bottom_percent = 0.15
        pad_sides_percent = 0.08
        # pt = padding sopra espresso in percentuale dell'altezza
        # pb = padding sotto espresso in percentuale dell'altezza
        # ps = padding laterale espresso in percentuale della larghezza
        pt = int(bbox_height * pad_top_percent)
        pb = int(bbox_height * pad_bottom_percent)
        ps = int(bbox_width * pad_sides_percent)
        x1_padded = max(0, x1 - ps)
        y1_padded = max(0, y1 - pt)
        x2_padded = min(img_w, x2 + ps)
        y2_padded = min(img_h, y2 + pb)
        # ... resto della funzione ...
```

## Utilizzo dal server

`app/server.py` importerà il nuovo modulo nel seguente modo:

```python
from .core_logic import (
    process_create_scene,
    process_detail_and_upscale,
    process_generate_all_parts,
    process_final_swap,
    ensure_pipeline_is_loaded,
    ensure_face_analyzer_is_loaded,
    # ...altre funzioni...
)
```

Le route esistenti continueranno a chiamare queste funzioni come ora, ma il codice di elaborazione sarà mantenuto separato da `server.py`.
