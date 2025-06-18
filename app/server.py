import sys
import time
import os
import io
import logging
import requests
import traceback
import gzip
import json
import uuid
import subprocess
import gc
import base64
import torch
import shutil
import tempfile
import re
from threading import Lock
import numpy as np
import cv2
from celery import Celery
from PIL import Image, ImageDraw, ImageOps

try:
    from PIL import UnidentifiedImageError
except Exception:  # pragma: no cover

    class UnidentifiedImageError(Exception):
        pass


import insightface.app
import insightface.model_zoo
from gfpgan import GFPGANer
from rembg import remove
try:
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )
except Exception:  # pragma: no cover - handled in tests with stubs
    StableDiffusionXLPipeline = None
    StableDiffusionXLControlNetInpaintPipeline = None
    ControlNetModel = None
    DPMSolverMultistepScheduler = None
from controlnet_aux import CannyDetector

# from PIL import Image, ImageDraw, ImageOps # Rimosso duplicato, ImageDraw già importato sopra
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from ultralytics import YOLO
from flask import (
    Flask,
    request,
    send_file,
    jsonify,
    render_template,
    current_app,
    url_for,
    session,
)
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_wtf import CSRFProtect
from app.meme_studio import meme_bp, GEMINI_MODEL_NAME
from app.auth import auth_bp, login_required
from app.model_manager import model_bp
from .forms import (
    SearchForm,
)  # Assumendo che forms.py sia allo stesso livello di server.py
from .user_model import init_db  # Assumendo che user_model.py sia allo stesso livello
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Costanti e Configurazioni Globali ---

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    logger.error("La libreria 'segment_anything' non è installata.")
    sam_model_registry, SamPredictor = None, None

try:
    import imageio_ffmpeg

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    logger.warning("imageio_ffmpeg non è installato.")
    ffmpeg_path = None

# === CONFIGURAZIONE GLOBALE ===
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"
DEFAULT_MODEL_NAME = "sdxl-yamers-realistic5-v5Rundiffusion"
ACTIVE_MODEL_FILE = os.path.join("models", "active_model.txt")
loaded_model_name = None
CFG_DETAIL_STEPS = 25
MAX_IMAGE_DIMENSION = 1280
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8MB limit
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# --- Inizializzazione Celery ---
celery = Celery(
    __name__,
    broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0"),
)

# === GESTIONE MODELLI ===
(
    face_analyzer,
    face_swapper,
    face_restorer,
    yolo_parser,
    sam_predictor,
    pipe,
    canny_detector,
) = (None,) * 7


# LOCKS per rendere i modelli thread-safe
face_analyzer_lock = Lock()
yolo_parser_lock = Lock()


def release_vram():
    logger.info("Rilascio della memoria cache della GPU...")
    gc.collect()
    torch.cuda.empty_cache()


def init_celery(
    app_instance,
):  # Rinominato per evitare conflitto con variabile 'app' globale in create_app
    celery.conf.update(
        broker_url=app_instance.config["CELERY_BROKER_URL"],
        result_backend=app_instance.config["CELERY_RESULT_BACKEND"],
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app_instance.app_context():  # Usa app_instance
                return super().__call__(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def validate_upload(file):
    ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF"]
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        # Assicurati che current_app sia disponibile o usa un default
        limit_source = (
            current_app
            if current_app
            else {"config": {"MAX_CONTENT_LENGTH": MAX_UPLOAD_SIZE}}
        )
        limit = limit_source.config.get("MAX_CONTENT_LENGTH", MAX_UPLOAD_SIZE)

        if file_size > limit:
            return None, f"File troppo grande (massimo {limit // 1024 // 1024}MB)"
        with Image.open(file) as img:
            if img.format not in ALLOWED_FORMATS:
                return (
                    None,
                    f"Formato immagine non supportato ({img.format}). Sono permessi: {', '.join(ALLOWED_FORMATS)}",
                )
        file.seek(0)  # Importante riavvolgere per letture successive
        filename = secure_filename(file.filename) if file.filename else "image.png"
        return filename, None
    except UnidentifiedImageError:
        return None, "Il file fornito non è un'immagine valida."
    except Exception as e:
        logger.error(f"Errore imprevisto durante la validazione dell'upload: {e}")
        return None, "Errore interno durante la validazione del file."


# --- FUNZIONI DI CARICAMENTO MODELLI ---
def ensure_yolo_parser_is_loaded():
    global yolo_parser
    if yolo_parser is None:
        logger.info("Caricamento YOLO Human Parser...")
        yolo_parser = YOLO(os.path.join("models", "yolo-human-parse-v2.pt"))


def ensure_sam_predictor_is_loaded():
    global sam_predictor
    if sam_predictor is None and sam_model_registry:
        model_type = "vit_l"
        model_filename = "sam_vit_l_0b3195.pth"
        model_path = os.path.abspath(os.path.join("models", model_filename))
        if os.path.exists(model_path):
            logger.info(
                "Caricamento Segment Anything (SAM) - Modello: %s...", model_type
            )
            sam_model = sam_model_registry[model_type](checkpoint=model_path)
            sam_model.to(device="cuda" if torch.cuda.is_available() else "cpu")
            sam_predictor = SamPredictor(sam_model)
        else:
            logger.error("Modello SAM '%s' non trovato.", model_filename)


def _get_active_model():
    if os.path.isfile(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    return name
        except Exception:
            pass
    return DEFAULT_MODEL_NAME


def ensure_pipeline_is_loaded(model_name=None):
    global pipe, canny_detector, loaded_model_name
    model_name = model_name or _get_active_model()
    if pipe is not None and loaded_model_name == model_name:
        return True

    release_vram()
    logger.info("Caricamento pipeline SDXL '%s'...", model_name)
    model_path = os.path.join("models", "checkpoints", model_name)
    if not os.path.isdir(model_path):
        logger.error("Directory del modello SDXL non trovata: %s", model_path)
        return False
    canny_detector = CannyDetector()
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    loaded_model_name = model_name
    return True


def ensure_face_analyzer_is_loaded():
    global face_analyzer
    if face_analyzer is None:
        logger.info("Caricamento FaceAnalysis...")
        face_analyzer = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))


def ensure_face_swapper_is_loaded():
    global face_swapper
    if face_swapper is None:
        model_path = os.path.join("models", "inswapper_128.onnx")
        if not os.path.exists(model_path):
            logger.error("Modello Face Swapper non trovato: %s", model_path)
            return
        face_swapper = insightface.model_zoo.get_model(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


def ensure_face_restorer_is_loaded():
    global face_restorer
    if face_restorer is None and os.path.exists(
        os.path.join("models", "GFPGANv1.4.pth")
    ):
        face_restorer = GFPGANer(
            model_path=os.path.join("models", "GFPGANv1.4.pth"),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )


# --- FUNZIONI HELPER E DI PROCESSO ---
def normalize_image(
    img: Image.Image, max_dim: int = MAX_IMAGE_DIMENSION
) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height:
            new_width, new_height = max_dim, int(height * (max_dim / width))
        else:
            new_height, new_width = max_dim, int(width * (max_dim / height))
        new_width -= new_width % 8
        new_height -= new_height % 8
        logger.info("Immagine ridimensionata a %dx%d.", new_width, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


# ... (le altre funzioni di processo come make_mask, process_generate_all_parts, etc. rimangono invariate) ...
# (Per brevità, non le ripeto qui, ma assicurati che siano presenti nel tuo file)


def process_generate_all_parts(image_bytes, prompts, progress_cb=None, model_name=None):
    ensure_pipeline_is_loaded(model_name)
    ensure_yolo_parser_is_loaded()
    ensure_sam_predictor_is_loaded()
    current_image = normalize_image(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    valid_prompts = [p for p in prompts.values() if p]
    total = len(valid_prompts) or 1
    completed = 0
    for part_name, prompt_text in prompts.items():
        if not prompt_text:
            continue
        mask = make_mask(current_image, (part_name,))
        if mask:
            w, h = current_image.size
            canny_map = canny_detector(
                current_image, low_threshold=50, high_threshold=150
            )
            if canny_map.size != current_image.size:
                canny_map = canny_map.resize(
                    current_image.size, Image.Resampling.LANCZOS
                )
            canny_array = np.array(canny_map)
            mask_resized = mask.resize((w, h), Image.Resampling.LANCZOS)
            mask_array = np.array(mask_resized.convert("L"))
            canny_array[mask_array > 128] = 0
            control_image = Image.fromarray(canny_array)
            if DEBUG_MODE and "current_app" in globals() and current_app:
                # Usa current_app.root_path se disponibile, altrimenti una directory relativa
                temp_dir_base = current_app.root_path if current_app else "."
                control_image.save(
                    os.path.join(
                        temp_dir_base, "temp", f"debug_control_{part_name}.png"
                    )
                )
            current_image = pipe(
                prompt=prompt_text,
                image=current_image,
                mask_image=mask_resized,
                control_image=control_image,
                width=w,
                height=h,
                controlnet_conditioning_scale=0.8,
                num_inference_steps=CFG_DETAIL_STEPS,
                strength=1.0,
                guidance_scale=10,
            ).images[0]
        completed += 1
        if progress_cb:
            progress_cb(int(completed / total * 100))
    return current_image


def process_create_scene(subject_bytes, prompt, progress_cb=None, model_name=None):
    ensure_pipeline_is_loaded(model_name)

    def progress_callback(
        pipe_ref, step, timestep, callback_kwargs
    ):  # Modificato per corrispondere alla firma attesa
        if progress_cb:
            progress_cb(int((step + 1) / CFG_DETAIL_STEPS * 100))
        return callback_kwargs

    subject = normalize_image(Image.open(io.BytesIO(subject_bytes)).convert("RGB"))
    mask = remove(subject, only_mask=True, post_process_mask=True)
    mask = ImageOps.invert(mask.convert("L"))
    canny_map = canny_detector(subject, low_threshold=50, high_threshold=150)
    if canny_map.size != subject.size:
        canny_map = canny_map.resize(subject.size, Image.Resampling.LANCZOS)
    result = pipe(
        prompt=prompt,
        image=subject,
        mask_image=mask,
        control_image=canny_map,
        width=subject.width,
        height=subject.height,
        callback_on_step_end=progress_callback,  # Modificato nome callback
        controlnet_conditioning_scale=0.8,
        num_inference_steps=CFG_DETAIL_STEPS,
        strength=1.0,
        guidance_scale=10,
    ).images[0]
    if progress_cb:
        progress_cb(100)
    return result


def process_detail_and_upscale(
    scene_image_bytes,
    enable_hires,
    tile_denoising_strength,
    progress_cb=None,
    model_name=None,
):
    global pipe, canny_detector  # Riferimenti a variabili globali
    try:
        ensure_pipeline_is_loaded(model_name)
        scene = normalize_image(
            Image.open(io.BytesIO(scene_image_bytes)).convert("RGB")
        )
        if enable_hires:
            logger.info("Avvio Hi-Res fix (RealESRGAN)...")
            if progress_cb:
                progress_cb(10)
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            upsampler = RealESRGANer(
                scale=2,
                model_path=os.path.join("models", "RealESRGAN_x2plus.pth"),
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
            # Assicurati che 'scene' sia un array NumPy per upsampler.enhance
            scene_np = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2BGR)
            output, _ = upsampler.enhance(scene_np, outscale=2)
            scene = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            if progress_cb:
                progress_cb(50)

        logger.info("Avvio detailing con ControlNet...")
        canny_map = canny_detector(scene, low_threshold=50, high_threshold=150)
        if canny_map.size != scene.size:
            canny_map = canny_map.resize(scene.size, Image.Resampling.LANCZOS)

        def detailing_progress_callback(
            pipe_ref, step, timestep, callback_kwargs
        ):  # Modificato per corrispondere
            if progress_cb:
                progress = 50 + int(
                    (step + 1) / CFG_DETAIL_STEPS * 45
                )  # Assumendo CFG_DETAIL_STEPS
                progress_cb(progress)
            return callback_kwargs

        result = pipe(
            prompt="",
            image=scene,
            mask_image=Image.new("L", scene.size, 255),
            control_image=canny_map,
            width=scene.width,
            height=scene.height,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=CFG_DETAIL_STEPS,
            strength=float(tile_denoising_strength),
            guidance_scale=5,  # Convertito a float
            callback_on_step_end=detailing_progress_callback,  # Modificato nome callback
        ).images[0]

        if progress_cb:
            progress_cb(100)
        return result
    finally:
        pipe, canny_detector = None, None  # Resetta le variabili globali
        release_vram()


def process_final_swap(
    target_bytes, source_bytes, source_idx, target_idx, progress_cb=None
):
    """Esegue il face swap tra due immagini.

    Args:
        target_bytes (bytes): Immagine di destinazione in formato bytes.
        source_bytes (bytes): Immagine sorgente da cui prelevare il volto.
        source_idx (int): Indice del volto sorgente da usare.
        target_idx (int): Indice del volto da sostituire nella destinazione.
        progress_cb (Callable[[int], None] | None): Callback opzionale per
            notificare l'avanzamento in percentuale.

    Returns:
        Image.Image: L'immagine risultante con il volto sostituito e
        opzionalmente restaurato.
    """

    ensure_face_analyzer_is_loaded()
    ensure_face_swapper_is_loaded()
    ensure_face_restorer_is_loaded()
    target_pil = normalize_image(Image.open(io.BytesIO(target_bytes)).convert("RGB"))
    source_pil = normalize_image(Image.open(io.BytesIO(source_bytes)).convert("RGB"))
    target_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
    source_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)

    # Usa face_analyzer globale
    target_faces = face_analyzer.get(target_cv)
    source_faces = face_analyzer.get(source_cv)

    if not target_faces or not source_faces:
        raise ValueError("Volti non trovati nell'immagine sorgente o di destinazione.")
    if source_idx >= len(source_faces) or target_idx >= len(target_faces):
        raise IndexError("Indice del volto non valido.")

    result_img = face_swapper.get(
        target_cv, target_faces[target_idx], source_faces[source_idx], paste_back=True
    )
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


# --- TASKS CELERY --- (invariate, ma assicurati che usino le funzioni di processo corrette)
@celery.task(bind=True)
def create_scene_task(self, subject_bytes, prompt, model_name=None):
    try:
        self.update_state(
            state="PROGRESS",
            meta={"progress": 5, "status": "Avvio generazione scena..."},
        )

        def update_progress(p):
            progress = 5 + int(p * 0.90)
            self.update_state(state="PROGRESS", meta={"progress": progress})

        final_image = process_create_scene(
            subject_bytes, prompt, update_progress, model_name
        )
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        release_vram()
        self.update_state(state="SUCCESS", meta={"progress": 100})
        return {"progress": 100, "data": img_str}
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise e


@celery.task(bind=True)
def detail_and_upscale_task(
    self, scene_image_bytes, enable_hires, tile_denoising_strength, model_name=None
):
    try:
        self.update_state(
            state="PROGRESS",
            meta={"progress": 5, "status": "Avvio dettaglio/upscale..."},
        )

        def update_progress(p):
            progress = 5 + int(p * 0.90)
            self.update_state(state="PROGRESS", meta={"progress": progress})

        final_image = process_detail_and_upscale(
            scene_image_bytes,
            enable_hires,
            tile_denoising_strength,
            update_progress,
            model_name,
        )
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        self.update_state(state="SUCCESS", meta={"progress": 100})
        return {"progress": 100, "data": img_str}
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise e


@celery.task(bind=True)
def generate_all_parts_task(self, prompts_json_str, image_bytes, model_name=None):
    try:
        self.update_state(
            state="PROGRESS",
            meta={"progress": 5, "status": "Avvio generazione parti..."},
        )
        prompts = json.loads(prompts_json_str)

        def update_progress(p):
            progress = 5 + int(p * 0.90)
            self.update_state(state="PROGRESS", meta={"progress": progress})

        final_image = process_generate_all_parts(
            image_bytes, prompts, update_progress, model_name
        )
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        release_vram()
        self.update_state(state="SUCCESS", meta={"progress": 100})
        return {"progress": 100, "data": img_str}
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise e


@celery.task(bind=True)
def final_swap_task(self, target_bytes, source_bytes, s_idx, t_idx):
    try:
        self.update_state(
            state="PROGRESS",
            meta={"progress": 5, "status": "Avvio face swap..."},
        )

        def update_progress(p):
            progress = 5 + int(p * 0.90)
            self.update_state(state="PROGRESS", meta={"progress": progress})

        final_image = process_final_swap(
            target_bytes, source_bytes, s_idx, t_idx, update_progress
        )
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        release_vram()
        self.update_state(state="SUCCESS", meta={"progress": 100})
        return {"progress": 100, "data": img_str}
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise e


@celery.task(bind=True)
def download_and_install_model_task(self, civitai_url):
    """
    Task per scaricare e installare un modello da Civitai, mostrando la velocità di download.
    """
    try:
        self.update_state(
            state="PROGRESS",
            meta={"progress": 5, "status": "Preparazione download modello..."},
        )

        def update(p, status):
            self.update_state(state="PROGRESS", meta={"progress": p, "status": status})

        logger.info(f"Inizio download per URL: {civitai_url}")
        
        match = re.search(r"/models/(\d+)", civitai_url)
        if not match:
            raise ValueError("URL Civitai non valido. Assicurati di usare il link della pagina del modello.")
        
        model_id = match.group(1)
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        info = response.json()

        ver = info.get("modelVersions", [{}])[0]
        file_info = next((f for f in ver.get("files", []) if f.get("name", "").endswith(".safetensors")), None)

        if not file_info or not file_info.get("downloadUrl"):
            raise ValueError("Nessun file .safetensors scaricabile trovato per questo modello su Civitai.")

        download_url = file_info["downloadUrl"]
        original_filename = file_info["name"]
        
        base_name = os.path.splitext(original_filename)[0]
        safe_model_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", base_name).strip("-").lower() or f"model-{model_id}"
        
        logger.info(f"Trovato modello '{safe_model_name}'.")

        output_dir = os.path.join("models", "checkpoints", safe_model_name)
        downloads_dir = os.path.join("models", "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        cached_path = os.path.join(downloads_dir, f"{safe_model_name}.safetensors")

        if os.path.isdir(output_dir):
            logger.info("Modello già convertito. Niente da fare.")
            self.update_state(
                state="SUCCESS",
                meta={"progress": 100, "status": "Model already installed."},
            )
            return {
                "progress": 100,
                "status": "Model already installed.",
                "model_name": safe_model_name,
            }

        if not os.path.isfile(cached_path):
            update(5, "Starting download...")
            api_key = os.getenv("CIVITAI_API_KEY")
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                logger.info("Richiesta di download autenticata con chiave API.")

            with requests.get(download_url, stream=True, timeout=30, headers=headers) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                start_time = time.time()
                last_update_time = start_time
                downloaded_since_last_update = 0
                with open(cached_path, "wb") as f:
                    downloaded_size = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        chunk_size = len(chunk)
                        downloaded_size += chunk_size
                        downloaded_since_last_update += chunk_size
                        current_time = time.time()
                        if current_time - last_update_time >= 1:
                            progress = 5 + int((downloaded_size / total_size) * 90) if total_size > 0 else 5
                            elapsed_time = current_time - last_update_time
                            speed = downloaded_since_last_update / elapsed_time / 1024
                            update(progress, f"Downloading... {speed:.1f} KB/s")
                            last_update_time = current_time
                            downloaded_since_last_update = 0
        else:
            update(5, "File già presente, salto download...")

        logger.info("Download disponibile. Inizio conversione...")
        update(98, "Converting model...")

        command = [sys.executable, "convert_safetensor.py", "--input", cached_path, "--output", safe_model_name]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_message = f"Conversione fallita: {result.stderr}"
            logger.error(f"Errore dallo script di conversione:\n{error_message}")
            raise Exception(error_message)

        logger.info("Conversione completata con successo.")
        release_vram()
        self.update_state(state="SUCCESS", meta={"progress": 100, "status": "Completed!"})
        return {"progress": 100, "status": "Completed!", "model_name": safe_model_name}

    except Exception as e:
        error_str = str(e)
        logger.exception("Errore critico durante l'installazione del modello")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_str})
        raise Exception(error_str)


# --- FACTORY DELL'APPLICAZIONE FLASK ---
def create_app():
    app_instance = Flask(__name__)  # Rinominato per chiarezza
    load_dotenv()
    with app_instance.app_context():  # Usa app_instance
        init_db()

    app_instance.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
    app_instance.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    app_instance.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
    app_instance.config.update(
        CELERY_BROKER_URL=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
        CELERY_RESULT_BACKEND=os.getenv(
            "CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0"
        ),
    )
    init_celery(app_instance)  # Passa app_instance
    CORS(app_instance, resources={r"/*": {"origins": "*"}})  # Usa app_instance
    CSRFProtect(app_instance)  # Usa app_instance
    app_instance.register_blueprint(meme_bp)  # Usa app_instance
    app_instance.register_blueprint(auth_bp)  # Usa app_instance
    app_instance.register_blueprint(model_bp)

    # --- DEFINIZIONE ROUTE ---
    @app_instance.route("/")  # Usa app_instance
    def home():
        return render_template("index.html", username=session.get("user_id"))

    @app_instance.route("/explore")  # Usa app_instance
    def explore():
        form = SearchForm()
        return render_template(
            "esplora.html", form=form, username=session.get("user_id")
        )

    @app_instance.route("/gallery")  # Usa app_instance
    @login_required
    def gallery_page():
        form = SearchForm()
        return render_template(
            "galleria.html", form=form, username=session.get("user_id")
        )

    @app_instance.route("/api/stickers")  # Usa app_instance
    def get_stickers_api():
        # Usa current_app se sei dentro un contesto di richiesta, altrimenti app_instance.static_folder
        static_folder_path = (
            current_app.static_folder if current_app else app_instance.static_folder
        )
        sticker_dir = os.path.join(static_folder_path, "stickers")
        if not os.path.isdir(sticker_dir):
            logger.warning("La cartella '%s' non è stata trovata.", sticker_dir)
            return jsonify([])
        sticker_data = []
        for root, dirs, files in os.walk(sticker_dir):
            if root == sticker_dir:
                continue
            category_name = os.path.basename(root)
            sticker_paths = [
                os.path.join(
                    "static", os.path.relpath(root, static_folder_path), file
                ).replace("\\", "/")
                for file in sorted(files)
                if file.lower().endswith((".png", ".webm", ".tgs"))
            ]
            if sticker_paths:
                sticker_data.append(
                    {"category": category_name, "stickers": sticker_paths}
                )
        return jsonify(sticker_data)

    # ... (Altre route API come /api/approved_memes, /api/meme, /lottie_json, /prepare_subject, etc.
    # dovrebbero usare 'current_app' o 'app_instance' in modo simile per accedere a config o static_folder)

    @app_instance.route("/api/approved_memes")
    def get_approved_memes():
        static_folder_path = (
            current_app.static_folder if current_app else app_instance.static_folder
        )
        gallery_dir = os.path.join(static_folder_path, "gallery")
        if not os.path.isdir(gallery_dir):
            return jsonify([])
        # ... (resto della logica) ...
        items = []  # Inizializza items
        for root, dirs, files in os.walk(gallery_dir):
            meta_path = os.path.join(root, "meta.json")
            meta = {}
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                except (json.JSONDecodeError, IOError):
                    meta = {}

            for fname in sorted(files):
                if not fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".webp", ".gif")
                ):
                    continue
                info = meta.get(fname, {})
                if not info.get("shared"):
                    continue
                rel_path = os.path.relpath(
                    os.path.join(root, fname), static_folder_path
                )
                items.append(
                    {
                        "title": info.get("title", os.path.splitext(fname)[0]),
                        "url": url_for(
                            "static", filename=rel_path.replace(os.sep, "/")
                        ),
                        "caption": info.get("caption", ""),
                        "tags": info.get("tags", []),
                        "ts": info.get("ts"),
                        "shared": True,
                    }
                )
        items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
        return jsonify(items)

    @app_instance.route("/api/memes")
    def api_memes():
        return get_approved_memes()

    @app_instance.route("/api/meme", methods=["POST"])
    @login_required
    def api_add_meme():
        if "image" not in request.files:
            return jsonify({"error": "Immagine mancante"}), 400
        session_user = session.get("user_id")
        user = request.form.get("user", session_user or "guest")
        if user != session_user:
            return jsonify({"error": "Forbidden"}), 403

        file = request.files["image"]
        filename, err = validate_upload(file)
        if err:
            return jsonify({"error": err}), 400

        fname = uuid.uuid4().hex + os.path.splitext(filename)[1]
        static_folder_path = (
            current_app.static_folder if current_app else app_instance.static_folder
        )
        user_dir = os.path.join(static_folder_path, "gallery", user)
        os.makedirs(user_dir, exist_ok=True)

        # Salva il file dopo aver riavvolto lo stream, dato che validate_upload lo ha letto
        file.seek(0)
        file.save(os.path.join(user_dir, fname))

        meta_path = os.path.join(
            user_dir, "gallery.json"
        )  # Dovrebbe essere user_dir non static_folder_path
        try:
            data = (
                json.load(open(meta_path, "r", encoding="utf-8"))
                if os.path.isfile(meta_path)
                else []
            )
        except (json.JSONDecodeError, IOError):
            data = []

        data.append(
            {
                "file": fname,
                "title": os.path.splitext(filename)[0],
                "shared": request.form.get("shared", "false").lower() == "true",
            }
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return jsonify({"url": url_for("static", filename=f"gallery/{user}/{fname}")})

    @app_instance.route("/lottie_json/<path:sticker_path>")
    def get_lottie_json(sticker_path):
        try:
            static_folder_path = (
                current_app.static_folder if current_app else app_instance.static_folder
            )
            base_dir = os.path.abspath(static_folder_path)
            # Il percorso dello sticker dovrebbe essere relativo alla directory 'static'
            # Esempio: sticker_path = "stickers/categoria/file.tgs"
            # full_path dovrebbe essere costruito partendo da base_dir
            full_path = os.path.abspath(
                os.path.join(base_dir, sticker_path.replace("static/", "", 1))
            )  # Rimuovi 'static/' se presente

            if not full_path.startswith(base_dir):
                return jsonify({"error": "Accesso non autorizzato al percorso."}), 403
            if not os.path.isfile(full_path):
                return jsonify({"error": "File non trovato."}), 404
            with gzip.open(full_path, "rt", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            logger.exception("Errore durante la lettura del file Lottie JSON")
            return jsonify({"error": f"Errore interno del server: {e}"}), 500

    @app_instance.route("/prepare_subject", methods=["POST"])
    def prepare_subject():
        try:
            if "subject_image" not in request.files:
                return jsonify(error="Immagine soggetto mancante."), 400
            file = request.files["subject_image"]
            _, err = validate_upload(file)  # validate_upload ora fa file.seek(0)
            if err:
                return jsonify(error=err), 400
            # file.seek(0) # Non più necessario qui se validate_upload lo fa
            subject = normalize_image(
                Image.open(io.BytesIO(file.read())).convert("RGBA")
            )
            processed = remove(subject)
            buf = io.BytesIO()
            processed.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app_instance.route("/enhance_prompt", methods=["POST"])
    def enhance_prompt():
        api_key = (
            current_app.config.get("GEMINI_API_KEY")
            if current_app
            else app_instance.config.get("GEMINI_API_KEY")
        )
        if not api_key:
            return jsonify(error="Gemini API key not configured"), 400
        # ... (resto della logica, assicurati che GEMINI_MODEL_NAME sia definito) ...
        try:
            data = request.get_json()
            if not data or "image_data" not in data:
                return jsonify(error="image_data missing"), 400
            system_prompt = "Sei un esperto prompt engineer. Migliora il seguente prompt in italiano basandoti sull'immagine fornita. Restituisci solo il prompt ottimizzato."
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": system_prompt
                                + "\nUtente: "
                                + data.get("prompt_text", "")
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": data["image_data"],
                                }
                            },
                        ]
                    }
                ]
            }
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("candidates"):
                return jsonify(
                    enhanced_prompt=result["candidates"][0]["content"]["parts"][0][
                        "text"
                    ].strip('"')
                )
            return jsonify(error="No prompt generated"), 500
        except requests.Timeout:
            return (
                jsonify(error="La richiesta a Gemini ha impiegato troppo tempo."),
                504,
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=f"Gemini error: {e}"), 500

    @app_instance.route("/enhance_part_prompt", methods=["POST"])
    def enhance_part_prompt():
        api_key = (
            current_app.config.get("GEMINI_API_KEY")
            if current_app
            else app_instance.config.get("GEMINI_API_KEY")
        )
        if not api_key:
            return jsonify(error="Gemini API key not configured"), 400
        # ... (resto della logica) ...
        try:
            data = request.get_json()
            if not data or "image_data" not in data:
                return jsonify(error="image_data missing"), 400
            system_prompt = f"Migliora il prompt per la parte '{data.get('part_name', 'subject')}'. Rispondi in italiano con un testo adatto a Stable Diffusion. Restituisci solo il prompt ottimizzato."
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": system_prompt
                                + "\nUtente: "
                                + data.get("prompt_text", "")
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": data["image_data"],
                                }
                            },
                        ]
                    }
                ]
            }
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("candidates"):
                return jsonify(
                    enhanced_prompt=result["candidates"][0]["content"]["parts"][0][
                        "text"
                    ].strip('"')
                )
            return jsonify(error="No prompt generated"), 500
        except requests.Timeout:
            return (
                jsonify(error="La richiesta a Gemini ha impiegato troppo tempo."),
                504,
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=f"Gemini error: {e}"), 500

    @app_instance.route("/analyze_parts", methods=["POST"])
    def analyze_parts():
        global yolo_parser
        with yolo_parser_lock:
            try:
                if "image" not in request.files:
                    return jsonify(error="Missing image"), 400
                file = request.files["image"]
                _, err = validate_upload(file)  # validate_upload ora fa file.seek(0)
                if err:
                    return jsonify(error=err), 400
                # file.seek(0) # Non più necessario
                ensure_yolo_parser_is_loaded()
                image_pil = normalize_image(
                    Image.open(io.BytesIO(file.read())).convert("RGB")
                )
                results = yolo_parser(image_pil)[0]
                return jsonify(
                    parts=sorted(
                        list(
                            set([results.names[int(cls)] for cls in results.boxes.cls])
                        )
                    )
                )
            except Exception as e:
                traceback.print_exc()
                return jsonify(error=str(e)), 500
            finally:
                yolo_parser = None
                release_vram()

    # --- MODIFICHE PER DEBUG IN detect_faces ---
    @app_instance.route("/detect_faces", methods=["POST"])
    def detect_faces():
        """Rileva i volti nell'immagine inviata e applica un padding proporzionale.

        Returns:
            Response: JSON con la lista di bounding box rilevate e paddate.
        """
        global face_analyzer
        with face_analyzer_lock:
            try:
                if "image" not in request.files:
                    return jsonify({"error": "Immagine mancante."}), 400
                file = request.files["image"]

                _, err = validate_upload(file)
                if err:
                    return jsonify({"error": err}), 400

                image_bytes_content = file.read()
                image_pil = normalize_image(
                    Image.open(io.BytesIO(image_bytes_content)).convert("RGB")
                )

                # Ottieni le dimensioni di image_pil QUI, dopo che è stata normalizzata
                img_w, img_h = image_pil.size  # <<--- DEFINIZIONE DI img_w e img_h

                temp_dir_base = (
                    current_app.root_path
                    if current_app
                    else os.path.abspath(os.path.dirname(__file__))
                )
                temp_debug_dir = os.path.join(temp_dir_base, "temp")
                os.makedirs(temp_debug_dir, exist_ok=True)

                debug_input_path = os.path.join(
                    temp_debug_dir, "debug_detection_input.png"
                )
                image_pil.save(debug_input_path)
                logger.info(
                    f"Immagine di input per il rilevamento salvata in: {debug_input_path}"
                )

                ensure_face_analyzer_is_loaded()
                # Passa l'array numpy a insightface come prima
                faces = face_analyzer.get(
                    cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                )

                image_pil_with_boxes = image_pil.copy()
                draw = ImageDraw.Draw(image_pil_with_boxes)
                processed_faces_for_json = []

                for i, f in enumerate(faces):
                    x1, y1, x2, y2 = [int(c) for c in f.bbox]

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    pad_top_percent = 0.10
                    pad_bottom_percent = 0.15
                    pad_sides_percent = 0.08

                    # Calcola la quantità di padding in pixel.
                    # pt = padding sopra la bbox in percentuale dell'altezza.
                    # pb = padding sotto la bbox in percentuale dell'altezza.
                    # ps = padding laterale in percentuale della larghezza.
                    pt = int(bbox_height * pad_top_percent)
                    pb = int(bbox_height * pad_bottom_percent)
                    ps = int(bbox_width * pad_sides_percent)

                    # Usa img_w e img_h definite sopra per i limiti
                    x1_padded = max(0, x1 - ps)
                    y1_padded = max(0, y1 - pt)
                    x2_padded = min(img_w, x2 + ps)  # Ora img_w è definita
                    y2_padded = min(img_h, y2 + pb)  # Ora img_h è definita

                    padded_bbox = [x1_padded, y1_padded, x2_padded, y2_padded]
                    processed_faces_for_json.append({"id": i, "bbox": padded_bbox})
                    draw.rectangle(padded_bbox, outline="lime", width=3)
                    draw.text(
                        (padded_bbox[0], padded_bbox[1] - 12),
                        f"Face {i} (Padded)",
                        fill="lime",
                    )

                debug_output_path = os.path.join(
                    temp_debug_dir, "server_detection_output_padded.png"
                )
                image_pil_with_boxes.save(debug_output_path)
                logger.info(
                    f"Immagine con BBOX PADDED DAL SERVER salvata in: {debug_output_path}"
                )

                logger.info(
                    f"BBOX (con padding) inviate al frontend: {processed_faces_for_json}"
                )
                return jsonify({"faces": processed_faces_for_json})

            except Exception as e:
                traceback.print_exc()
                # Restituisci un errore più specifico se possibile, altrimenti generico
                error_message = f"Errore rilevamento volti: {type(e).__name__} - {e}"
                logger.error(error_message)  # Logga l'errore completo
                return (
                    jsonify({"error": error_message}),
                    500,
                )  # Invia l'errore al client
            finally:
                face_analyzer = None
                release_vram()

    # --- FINE MODIFICHE PER DEBUG ---

    @app_instance.route("/async/create_scene", methods=["POST"])
    @login_required
    def async_create_scene():
        if "subject_data" not in request.files or "prompt" not in request.form:
            return jsonify(error="Dati mancanti"), 400
        subject_bytes = request.files["subject_data"].read()
        prompt_text = request.form["prompt"]
        model_name = request.form.get("model_name")
        task = create_scene_task.apply_async(
            args=[subject_bytes, prompt_text, model_name]
        )
        return jsonify(task_id=task.id), 202

    @app_instance.route("/async/detail_and_upscale", methods=["POST"])
    def async_detail_and_upscale():
        if "scene_image" not in request.files:
            return jsonify(error="Immagine mancante"), 400
        file = request.files["scene_image"]
        _, err = validate_upload(file)  # validate_upload ora fa file.seek(0)
        if err:
            return jsonify(error=err), 400
        # file.seek(0) # Non più necessario
        scene_image_bytes = file.read()
        enable_hires = request.form.get("enable_hires", "true").lower() == "true"
        tile_denoising_strength = float(
            request.form.get("tile_denoising_strength", 0.3)
        )
        model_name = request.form.get("model_name")
        task = detail_and_upscale_task.apply_async(
            args=[scene_image_bytes, enable_hires, tile_denoising_strength, model_name]
        )
        return jsonify(task_id=task.id), 202

    @app_instance.route("/async/generate_all_parts", methods=["POST"])
    def async_generate_all_parts():
        if "image" not in request.files or "prompts" not in request.form:
            return jsonify(error="Dati mancanti"), 400
        file = request.files["image"]
        _, err = validate_upload(file)  # validate_upload ora fa file.seek(0)
        if err:
            return jsonify(error=err), 400
        # file.seek(0) # Non più necessario
        image_bytes_content = file.read()
        prompts_json_str = request.form.get("prompts")
        model_name = request.form.get("model_name")
        task = generate_all_parts_task.apply_async(
            args=[prompts_json_str, image_bytes_content, model_name]
        )
        return jsonify(task_id=task.id), 202

    @app_instance.route("/async/final_swap", methods=["POST"])
    def async_final_swap():
        if (
            "target_image_high_res" not in request.files
            or "source_face_image" not in request.files
        ):
            return jsonify(error="Immagini mancanti."), 400

        target_file = request.files["target_image_high_res"]
        _, err_t = validate_upload(target_file)
        if err_t:
            return jsonify(error=f"Errore immagine target: {err_t}"), 400
        target_bytes_content = target_file.read()

        source_file = request.files["source_face_image"]
        _, err_s = validate_upload(source_file)
        if err_s:
            return jsonify(error=f"Errore immagine sorgente: {err_s}"), 400
        source_bytes_content = source_file.read()

        s_idx = int(request.form.get("source_face_index", 0))
        t_idx = int(request.form.get("target_face_index", 0))

        task = final_swap_task.apply_async(
            args=[target_bytes_content, source_bytes_content, s_idx, t_idx]
        )
        return jsonify(task_id=task.id), 202

    @app_instance.route("/api/models/download", methods=["POST"])
    def api_download_model():
        data = request.get_json() or {}
        civitai_url = data.get("url")
        if not civitai_url:
            return jsonify(error="url required"), 400
        task = download_and_install_model_task.apply_async(args=[civitai_url])
        return jsonify(task_id=task.id), 202

    @app_instance.route("/save_result_video", methods=["POST"])  # Usa app_instance
    def save_result_video():
        if not request.get_data():
            return jsonify(error="Missing video data"), 400
        try:
            static_folder_path = (
                current_app.static_folder if current_app else app_instance.static_folder
            )
            temp_dir = os.path.join(static_folder_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            input_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.webm")
            with open(input_path, "wb") as f:
                f.write(request.get_data())
            ext = "gif" if request.args.get("fmt", "mp4").lower() == "gif" else "mp4"
            output_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{ext}")
            if ffmpeg_path:
                cmd = [ffmpeg_path, "-y", "-i", input_path]
                if ext == "mp4":
                    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
                cmd.append(output_path)
                subprocess.run(cmd, check=True, capture_output=True)
                os.remove(input_path)
            else:
                shutil.move(input_path, output_path)
            rel_path = os.path.relpath(output_path, static_folder_path).replace(
                os.sep, "/"
            )
            return jsonify(url=url_for("static", filename=rel_path))
        except subprocess.CalledProcessError as e:
            logger.error(
                "Errore conversione video FFMPEG: %s", e.stderr.decode("utf-8")
            )
            return jsonify(error=f"Errore FFMPEG: {e.stderr.decode('utf-8')}"), 500
        except Exception as e:
            logger.exception("Errore conversione video")
            return jsonify(error=str(e)), 500

    @app_instance.route("/task_status/<task_id>")  # Usa app_instance
    def task_status(task_id):
        task = celery.AsyncResult(task_id)
        response = {"state": task.state, "progress": 0}
        if task.state == "PROGRESS":
            response["progress"] = task.info.get("progress", 0)
        elif task.state == "SUCCESS":
            response["progress"] = 100
            response["result"] = task.result
        elif task.state != "PENDING":
            response["error"] = str(task.info)
        return jsonify(response)

    @app_instance.after_request  # Usa app_instance
    def add_pna_header(response):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    return app_instance  # Restituisci app_instance


# Istanza per compatibilità con worker Celery e run.py
flask_app = create_app()
