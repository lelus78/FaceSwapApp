from __future__ import annotations

import os
import cv2
import numpy as np
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
import shutil
import torch
from threading import Lock
from celery import Celery
from PIL import Image, ImageDraw, ImageOps
try:  # UnidentifiedImageError may be missing in the stubbed PIL during tests
    from PIL import UnidentifiedImageError
except Exception:  # pragma: no cover - fallback for minimal stubs
    class UnidentifiedImageError(Exception):
        pass
import insightface.app 
import insightface.model_zoo 
from gfpgan import GFPGANer
from rembg import remove
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from controlnet_aux import CannyDetector
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
from app.meme_studio import meme_bp, GEMINI_MODEL_NAME # Assicurati che GEMINI_MODEL_NAME sia definito qui o importato
from app.auth import auth_bp, login_required
from .forms import SearchForm 
from .user_model import init_db 
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

DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"
DEFAULT_MODEL_NAME = "sdxl-yamers-realistic5-v5Rundiffusion"
ACTIVE_MODEL_FILE = os.path.join("models", "active_model.txt")
loaded_model_name = None
CFG_DETAIL_STEPS = 18
MAX_IMAGE_DIMENSION = 1280
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8MB limit
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

celery = Celery(__name__, broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
                backend=os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0"))

(face_analyzer, face_swapper, face_restorer, yolo_parser, sam_predictor, pipe, canny_detector) = (None, ) * 7
face_analyzer_lock = Lock() 
yolo_parser_lock = Lock()   

def release_vram():
    logger.info("Rilascio della memoria cache della GPU...")
    gc.collect()
    torch.cuda.empty_cache()

def init_celery(app_instance):
    celery.conf.update(
        broker_url=app_instance.config["CELERY_BROKER_URL"],
        result_backend=app_instance.config["CELERY_RESULT_BACKEND"],
    )
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app_instance.app_context():
                return super().__call__(*args, **kwargs)
    celery.Task = ContextTask
    return celery

def validate_upload(file):
    ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF"]
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        limit_source = current_app if current_app else {"config": {"MAX_CONTENT_LENGTH": MAX_UPLOAD_SIZE}}
        limit = limit_source.config.get("MAX_CONTENT_LENGTH", MAX_UPLOAD_SIZE)
        if file_size > limit: return None, f"File troppo grande (massimo {limit // 1024 // 1024}MB)"
        with Image.open(file) as img:
            if img.format not in ALLOWED_FORMATS: return None, f"Formato immagine non supportato ({img.format})."
        file.seek(0)
        filename = secure_filename(file.filename) if file.filename else "image.png"
        return filename, None
    except UnidentifiedImageError: return None, "Il file fornito non è un'immagine valida."
    except Exception as e:
        logger.error(f"Errore imprevisto validazione upload: {e}")
        return None, "Errore interno validazione file."

def _get_active_model() -> str:
    if os.path.isfile(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    return name
        except Exception:
            pass
    return DEFAULT_MODEL_NAME

def ensure_yolo_parser_is_loaded(): # ... (invariata)
    global yolo_parser
    if yolo_parser is None:
        logger.info("Caricamento YOLO Human Parser...")
        yolo_parser = YOLO(os.path.join("models", "yolo-human-parse-v2.pt"))
def ensure_sam_predictor_is_loaded(): # ... (invariata)
    global sam_predictor
    if sam_predictor is None and sam_model_registry:
        model_type = "vit_l"; model_filename = "sam_vit_l_0b3195.pth"
        model_path = os.path.abspath(os.path.join("models", model_filename))
        if os.path.exists(model_path):
            logger.info("Caricamento SAM: %s...", model_type)
            sam_model = sam_model_registry[model_type](checkpoint=model_path)
            sam_model.to(device="cuda" if torch.cuda.is_available() else "cpu")
            sam_predictor = SamPredictor(sam_model)
        else: logger.error("Modello SAM '%s' non trovato.", model_filename)
def ensure_pipeline_is_loaded(model_name: str | None = None):
    global pipe, canny_detector, loaded_model_name
    model_name = model_name or _get_active_model()
    if pipe is not None and loaded_model_name == model_name:
        return True
    release_vram()
    logger.info("Caricamento pipeline SDXL '%s'...", model_name)
    model_path = os.path.join("models", "checkpoints", model_name)
    if not os.path.isdir(model_path):
        logger.error("Dir SDXL non trovata: %s", model_path)
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
def ensure_face_analyzer_is_loaded(): # ... (invariata)
    global face_analyzer
    if face_analyzer is None:
        logger.info("Caricamento FaceAnalysis...")
        face_analyzer = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
def ensure_face_swapper_is_loaded(): # ... (invariata)
    global face_swapper
    if face_swapper is None:
        model_path = os.path.join("models", "inswapper_128.onnx")
        if not os.path.exists(model_path): logger.error("Modello Swapper non trovato: %s", model_path); return
        face_swapper = insightface.model_zoo.get_model(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
def ensure_face_restorer_is_loaded(): # ... (invariata)
    global face_restorer
    if face_restorer is None and os.path.exists(os.path.join("models", "GFPGANv1.4.pth")):
        face_restorer = GFPGANer(model_path=os.path.join("models", "GFPGANv1.4.pth"), upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)

def normalize_image(img, max_dim: int = MAX_IMAGE_DIMENSION):
    img = ImageOps.exif_transpose(img)
    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height: new_width, new_height = max_dim, int(height * (max_dim / width))
        else: new_height, new_width = max_dim, int(width * (max_dim / height))
        new_width -= new_width % 8; new_height -= new_height % 8
        logger.info("Immagine ridimensionata a %dx%d.", new_width, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

# ... (Le funzioni process_... come process_generate_all_parts, process_create_scene, ecc. rimangono, le ometto per brevità)
# Assicurati che tutte le tue funzioni process_... siano presenti nel file.
# ... (Task Celery come create_scene_task, etc. rimangono invariati, li ometto per brevità)
# ...
def process_final_swap(target_bytes, source_bytes, source_idx, target_idx, progress_cb=None): # ... (come prima)
    ensure_face_analyzer_is_loaded(); ensure_face_swapper_is_loaded(); ensure_face_restorer_is_loaded()
    target_pil = normalize_image(Image.open(io.BytesIO(target_bytes)).convert("RGB"))
    source_pil = normalize_image(Image.open(io.BytesIO(source_bytes)).convert("RGB"))
    target_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
    source_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
    target_faces = face_analyzer.get(target_cv); source_faces = face_analyzer.get(source_cv)
    if not target_faces or not source_faces: raise ValueError("Volti non trovati.")
    if source_idx >= len(source_faces) or target_idx >= len(target_faces): raise IndexError("Indice volto non valido.")
    result_img = face_swapper.get(target_cv, target_faces[target_idx], source_faces[source_idx], paste_back=True)
    if progress_cb: progress_cb(50)
    if face_restorer: _, _, result_img = face_restorer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.8)
    if progress_cb: progress_cb(100)
    return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

# Task Celery (assicurati che i tuoi task siano definiti correttamente)
@celery.task(bind=True)
def create_scene_task(self, subject_bytes, prompt, model_name=None):
    def update_progress(p):
        self.update_state(state='PROGRESS', meta={'progress': p})
    final_image = process_create_scene(subject_bytes, prompt, update_progress, model_name=model_name)
    buffered = io.BytesIO(); final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    release_vram(); return {'progress': 100, 'data': img_str}

@celery.task(bind=True)
def detail_and_upscale_task(self, scene_image_bytes, enable_hires, tile_denoising_strength, model_name=None):
    def update_progress(p):
        self.update_state(state='PROGRESS', meta={'progress': p})
    final_image = process_detail_and_upscale(scene_image_bytes, enable_hires, tile_denoising_strength, update_progress, model_name=model_name)
    buffered = io.BytesIO(); final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {'progress': 100, 'data': img_str}

@celery.task(bind=True)
def generate_all_parts_task(self, prompts_json_str, image_bytes, model_name=None):
    prompts = json.loads(prompts_json_str)
    def update_progress(p):
        self.update_state(state="PROGRESS", meta={"progress": p})
    final_image = process_generate_all_parts(image_bytes, prompts, update_progress, model_name=model_name)
    buffered = io.BytesIO(); final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    release_vram(); return {"progress": 100, "data": img_str}

@celery.task(bind=True)
def final_swap_task(self, target_bytes, source_bytes, s_idx, t_idx): # ... (come prima)
    def update_progress(p): self.update_state(state="PROGRESS", meta={"progress": p})
    final_image = process_final_swap(target_bytes, source_bytes, s_idx, t_idx, update_progress)
    buffered = io.BytesIO(); final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    release_vram(); return {"progress": 100, "data": img_str}


def create_app(): # ... (invariata, assicurati che tutte le route siano definite)
    app_instance = Flask(__name__) 
    load_dotenv()
    with app_instance.app_context(): init_db()
    app_instance.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
    app_instance.config["WTF_CSRF_ENABLED"] = False
    app_instance.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    app_instance.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
    app_instance.config["WTF_CSRF_ENABLED"] = False
    app_instance.config.update(CELERY_BROKER_URL=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"), CELERY_RESULT_BACKEND=os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0"))
    init_celery(app_instance) 
    CORS(app_instance, resources={r"/*": {"origins": "*"}}) 
    CSRFProtect(app_instance) 
    app_instance.register_blueprint(meme_bp) 
    app_instance.register_blueprint(auth_bp) 

    # ... (Tutte le tue route @app_instance.route("/") etc. devono essere qui)
    @app_instance.route("/")
    def home():
        return render_template("index.html", username=session.get('user_id'))

    @app_instance.route("/explore")
    def explore():
        form = SearchForm()
        return render_template("esplora.html", form=form, username=session.get('user_id'))

    # ... (altre route)

    @app_instance.route("/detect_faces", methods=["POST"])
    def detect_faces():
        global face_analyzer
        with face_analyzer_lock:
            try:
                if "image" not in request.files: return jsonify({"error": "Immagine mancante."}), 400
                file = request.files["image"]
                
                _, err = validate_upload(file) # validate_upload ora fa file.seek(0)
                if err: return jsonify({"error": err}), 400
                
                image_bytes_content = file.read() # Leggi i byte dopo la validazione
                image_pil = normalize_image(Image.open(io.BytesIO(image_bytes_content)).convert("RGB"))
                
                # DEBUG: Salva l'immagine di input
                # Assicurati che current_app sia disponibile o usa un percorso alternativo per 'temp'
                temp_dir_base = current_app.root_path if current_app else os.path.abspath(os.path.dirname(__file__))
                temp_debug_dir = os.path.join(temp_dir_base, "temp")
                os.makedirs(temp_debug_dir, exist_ok=True)

                debug_input_path = os.path.join(temp_debug_dir, "debug_detection_input.png")
                image_pil.save(debug_input_path)
                logger.info(f"Immagine di input per il rilevamento salvata in: {debug_input_path}")

                ensure_face_analyzer_is_loaded()
                faces = face_analyzer.get(cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR))
                
                image_pil_with_boxes = image_pil.copy()
                draw = ImageDraw.Draw(image_pil_with_boxes)
                processed_faces_for_json = []

                img_w, img_h = image_pil.size # Dimensioni dell'immagine normalizzata

                for i, f in enumerate(faces):
                    x1, y1, x2, y2 = [int(c) for c in f.bbox]
                    
                    # --- INIZIO LOGICA PADDING ---
                    # Calcola il padding come percentuale della dimensione della bbox rilevata
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Percentuali di padding (puoi regolarle)
                    pad_top_percent = 0.10  # 10% sopra
                    pad_bottom_percent = 0.15 # 15% sotto (per il mento)
                    pad_sides_percent = 0.08 # 8% ai lati

                    # Calcola i pixel di padding
                    pt = int(bbox_height * pad_top_percent)
                    pb = int(bbox_height * pad_bottom_percent)
                    ps = int(bbox_width * pad_sides_percent)

                    # Applica padding, assicurandoti di non andare fuori dai limiti dell'immagine (img_w, img_h)
                    x1_padded = max(0, x1 - ps)
                    y1_padded = max(0, y1 - pt)
                    x2_padded = min(img_w, x2 + ps)
                    y2_padded = min(img_h, y2 + pb)
                    
                    padded_bbox = [x1_padded, y1_padded, x2_padded, y2_padded]
                    # --- FINE LOGICA PADDING ---

                    processed_faces_for_json.append({"id": i, "bbox": padded_bbox}) # Usa la bbox con padding
                    
                    # Disegna la bbox con padding per il debug output
                    draw.rectangle(padded_bbox, outline="lime", width=3) 
                    draw.text((padded_bbox[0], padded_bbox[1] - 12), f"Face {i} (Padded)", fill="lime")

                debug_output_path = os.path.join(temp_debug_dir, "server_detection_output_padded.png") # Nome file diverso
                image_pil_with_boxes.save(debug_output_path)
                logger.info(f"Immagine con BBOX PADDED DAL SERVER salvata in: {debug_output_path}")
                
                logger.info(f"BBOX (con padding) inviate al frontend: {processed_faces_for_json}") # Log delle BBOX inviate
                return jsonify({"faces": processed_faces_for_json})
            
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"Errore rilevamento volti: {e}"}), 500
            finally:
                face_analyzer = None
                release_vram()

    @app_instance.route("/swap_face", methods=["POST"])
    def swap_face():
        if "target_image_high_res" not in request.files or "source_face_image" not in request.files:
            return "", 400
        return jsonify(status="ok")

    @app_instance.route("/generate_with_mask", methods=["POST"])
    def generate_with_mask():
        if "image" not in request.files or "mask" not in request.files:
            return "", 400
        return jsonify(status="ok")

    # ... (tutte le altre tue route come /async/create_scene etc. devono essere qui) ...
    @app_instance.route("/async/create_scene", methods=["POST"]) # ... (come prima)
    @login_required
    def async_create_scene():
        if "subject_data" not in request.files or "prompt" not in request.form: return jsonify(error="Dati mancanti"), 400
        subject_bytes = request.files["subject_data"].read()
        prompt_text = request.form["prompt"]
        model_name = request.form.get("model_name")
        task = create_scene_task.apply_async(args=[subject_bytes, prompt_text, model_name])
        return jsonify(task_id=task.id), 202
    
    @app_instance.route("/async/detail_and_upscale", methods=["POST"]) # ... (come prima)
    def async_detail_and_upscale():
        if "scene_image" not in request.files: return jsonify(error="Immagine mancante"), 400
        file = request.files["scene_image"]; _, err = validate_upload(file);
        if err: return jsonify(error=err), 400
        scene_image_bytes = file.read()
        enable_hires = request.form.get("enable_hires", "true").lower() == "true"
        tile_denoising_strength = float(request.form.get("tile_denoising_strength", 0.3))
        model_name = request.form.get("model_name")
        task = detail_and_upscale_task.apply_async(args=[scene_image_bytes, enable_hires, tile_denoising_strength, model_name])
        return jsonify(task_id=task.id), 202

    @app_instance.route("/async/generate_all_parts", methods=["POST"]) # ... (come prima)
    def async_generate_all_parts():
        if "image" not in request.files or "prompts" not in request.form : return jsonify(error="Dati mancanti"), 400
        file = request.files["image"]; _, err = validate_upload(file);
        if err: return jsonify(error=err), 400
        image_bytes_content = file.read()
        prompts_json_str = request.form.get("prompts")
        model_name = request.form.get("model_name")
        task = generate_all_parts_task.apply_async(args=[prompts_json_str, image_bytes_content, model_name])
        return jsonify(task_id=task.id), 202

    @app_instance.route("/async/final_swap", methods=["POST"]) # ... (come prima)
    def async_final_swap():
        if "target_image_high_res" not in request.files or "source_face_image" not in request.files: return jsonify(error="Immagini mancanti."), 400
        target_file = request.files["target_image_high_res"]; _, err_t = validate_upload(target_file)
        if err_t: return jsonify(error=f"Errore img target: {err_t}"),400
        target_bytes_content = target_file.read()
        source_file = request.files["source_face_image"]; _, err_s = validate_upload(source_file)
        if err_s: return jsonify(error=f"Errore img sorgente: {err_s}"),400
        source_bytes_content = source_file.read()
        s_idx = int(request.form.get("source_face_index", 0)); t_idx = int(request.form.get("target_face_index", 0))
        task = final_swap_task.apply_async(args=[target_bytes_content, source_bytes_content, s_idx, t_idx])
        return jsonify(task_id=task.id), 202
    
    @app_instance.route("/save_result_video", methods=["POST"]) # ... (come prima)
    def save_result_video():
        if not request.get_data():
            return jsonify(error="Dati video mancanti"), 400
        try:
            static_folder_path = current_app.static_folder if current_app else app_instance.static_folder
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
            rel_path = os.path.relpath(output_path, static_folder_path).replace(os.sep, "/")
            return jsonify(url=url_for("static", filename=rel_path))
        except subprocess.CalledProcessError as e:
            logger.error("Errore FFMPEG: %s", e.stderr.decode("utf-8"))
            return jsonify(error=f"Errore FFMPEG: {e.stderr.decode('utf-8')}"), 500
        except Exception as e:
            logger.exception("Errore conversione video")
            return jsonify(error=str(e)), 500

    @app_instance.route("/task_status/<task_id>") # ... (come prima)
    def task_status(task_id):
        task = celery.AsyncResult(task_id)
        response = {"state": task.state, "progress": 0}
        if task.state == "PROGRESS": response["progress"] = task.info.get("progress", 0)
        elif task.state == "SUCCESS": response["progress"] = 100; response["result"] = task.result
        elif task.state != "PENDING": response["error"] = str(task.info)
        return jsonify(response)

    @app_instance.after_request 
    def add_pna_header(response):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    return app_instance

flask_app = create_app()
app = flask_app
