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
import torch

from insightface.app import FaceAnalysis
import insightface.model_zoo
from gfpgan import GFPGANer
from rembg import remove
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from controlnet_aux import CannyDetector
from PIL import Image, ImageDraw
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from ultralytics import YOLO
from flask import Flask, request, send_file, jsonify, render_template, current_app, url_for
from flask_cors import CORS
from app.meme_studio import meme_bp
from dotenv import load_dotenv

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print(" [ERRORE] La libreria 'segment_anything' non è installata.")
    sam_model_registry, SamPredictor = None, None

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    print(" [ATTENZIONE] imageio_ffmpeg non è installato.")
    ffmpeg_path = None

# === CONFIGURAZIONE GLOBALE ===
DEBUG_MODE = True 
CFG_MODEL_NAME = "sdxl-yamers-realistic5-v5Rundiffusion"
CFG_DETAIL_STEPS = 18
MAX_IMAGE_DIMENSION = 1280

# === GESTIONE MODELLI ===
face_analyzer, face_swapper, face_restorer, yolo_parser, sam_predictor, pipe, canny_detector = (None,) * 7

def release_vram():
    print(" [VRAM] Rilascio della memoria cache della GPU...")
    gc.collect()
    torch.cuda.empty_cache()

# --- FUNZIONI DI CARICAMENTO MODELLI ---
def ensure_yolo_parser_is_loaded():
    global yolo_parser
    if yolo_parser is None:
        print(" [VRAM] Caricamento YOLO Human Parser...")
        yolo_parser = YOLO(os.path.join('models', 'yolo-human-parse-v2.pt'))

def ensure_sam_predictor_is_loaded():
    global sam_predictor
    if sam_predictor is None and sam_model_registry:
        model_type = "vit_l"
        model_filename = "sam_vit_l_0b3195.pth"
        model_path = os.path.abspath(os.path.join('models', model_filename))
        if os.path.exists(model_path):
            print(f" [VRAM] Caricamento Segment Anything (SAM) - Modello: {model_type}...")
            sam_model = sam_model_registry[model_type](checkpoint=model_path)
            sam_model.to(device="cuda" if torch.cuda.is_available() else "cpu")
            sam_predictor = SamPredictor(sam_model)
        else:
            print(f" [ERRORE] Modello SAM '{model_filename}' non trovato.")

def ensure_pipeline_is_loaded():
    global pipe, canny_detector
    if pipe is None:
        print(f" [VRAM] Caricamento pipeline SDXL '{CFG_MODEL_NAME}'...")
        model_path = os.path.join('models', 'checkpoints', CFG_MODEL_NAME)
        if not os.path.isdir(model_path): return False
        canny_detector = CannyDetector()
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
    return True
    
def ensure_face_analyzer_is_loaded():
    global face_analyzer
    if face_analyzer is None:
        print(" [VRAM] Caricamento FaceAnalysis...")
        face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

def ensure_face_swapper_is_loaded():
    global face_swapper
    if face_swapper is None:
        model_path = os.path.join('models', 'inswapper_128.onnx')
        face_swapper = insightface.model_zoo.get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def ensure_face_restorer_is_loaded():
    global face_restorer
    if face_restorer is None and os.path.exists(os.path.join('models', 'GFPGANv1.4.pth')):
        face_restorer = GFPGANer(model_path=os.path.join('models', 'GFPGANv1.4.pth'), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

# --- FUNZIONI HELPER ---
def normalize_image(img: Image.Image, max_dim=MAX_IMAGE_DIMENSION) -> Image.Image:
    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height: new_width, new_height = max_dim, int(height * (max_dim / width))
        else: new_height, new_width = max_dim, int(width * (max_dim / height))
        new_width -= new_width % 8; new_height -= new_height % 8
        print(f" [OTTIMIZZAZIONE] Immagine ridimensionata a {new_width}x{new_height}.")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

def make_mask(pil_img, parts_to_mask, conf_threshold=0.20):
    if sam_predictor is None or yolo_parser is None: return None
    res = yolo_parser(pil_img.convert("RGB"))[0]
    
    if DEBUG_MODE:
        temp_dir = os.path.join(current_app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        debug_session_id = uuid.uuid4().hex[:8]
        yolo_debug_img = pil_img.copy()
        draw = ImageDraw.Draw(yolo_debug_img)
        for box in res.boxes:
            class_name = res.names[int(box.cls.item())]
            coords = box.xyxy[0].cpu().numpy()
            draw.rectangle(coords, outline="red", width=3)
            draw.text((coords[0], coords[1] - 10), f"{class_name} ({box.conf.item():.2f})", fill="white")
        yolo_debug_img.save(os.path.join(temp_dir, f"{debug_session_id}_yolo_detections.png"))

    idx_map = {v: k for k, v in res.names.items()}
    target_idx = [idx_map[p] for p in parts_to_mask if p in idx_map]
    if not target_idx: return None
        
    final_mask_np = None

    if getattr(res, "masks", None) is not None and getattr(res.masks, "data", None) is not None:
        # Prefer YOLO segmentation masks when available. Use polygon coordinates
        # to avoid letterboxing misalignment issues.
        mask_polys = res.masks.xy  # already scaled to original image size
        combined = np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
        for poly, cls_idx, conf in zip(mask_polys, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            conf_val = conf.item() if hasattr(conf, "item") else float(conf)
            if int(cls_idx) in target_idx and conf_val > conf_threshold:
                poly_int = np.round(np.array(poly)).astype(np.int32)
                cv2.fillPoly(combined, [poly_int.reshape(-1, 1, 2)], 255)
        if combined.sum() > 0:
            final_mask_np = combined
    if final_mask_np is None:
        detected_boxes = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls.item()) in target_idx and b.conf.item() > conf_threshold]
        if not detected_boxes:
            return None

        box = detected_boxes[0]
        center_point = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
        point_labels = np.array([1])

        sam_predictor.set_image(np.array(pil_img.convert("RGB")))

        masks, scores, _ = sam_predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=True,
        )
        if masks is None:
            return None

        idx = int(np.array(scores).argmax()) if not isinstance(scores, torch.Tensor) else int(torch.argmax(scores).item())

        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        if masks_np.ndim == 4:
            best_mask = masks_np[0, idx]
        else:
            best_mask = masks_np[idx]

        final_mask_np = best_mask.astype(np.uint8) * 255

    if final_mask_np is None:
        return None
    
    closed_mask_np = cv2.morphologyEx(final_mask_np, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dilated_mask_np = cv2.dilate(closed_mask_np, np.ones((10, 10), np.uint8), iterations=1)
    
    final_mask = Image.fromarray(dilated_mask_np)

    if DEBUG_MODE:
        temp_dir = os.path.join(current_app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        debug_session_id = uuid.uuid4().hex[:8]
        final_mask.save(os.path.join(temp_dir, f"{debug_session_id}_final_mask.png"))
    
    return final_mask

def create_app():
    app = Flask(__name__)
    load_dotenv()
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.register_blueprint(meme_bp)

    @app.route('/')
    def home(): return render_template('index.html')

    @app.route('/api/stickers')
    def get_stickers_api():
        sticker_dir = os.path.join(app.static_folder, 'stickers')
        if not os.path.isdir(sticker_dir):
            print(f" [ATTENZIONE] La cartella '{sticker_dir}' non è stata trovata.")
            return jsonify([])
        sticker_data = []
        for root, dirs, files in os.walk(sticker_dir):
            if root == sticker_dir: continue
            category_name = os.path.basename(root)
            sticker_paths = [os.path.join('static', os.path.relpath(root, app.static_folder), file).replace("\\", "/") for file in sorted(files) if file.lower().endswith(('.png', '.webm', '.tgs'))]
            if sticker_paths: sticker_data.append({"category": category_name, "stickers": sticker_paths})
        return jsonify(sticker_data)

    @app.route('/lottie_json/<path:sticker_path>')
    def get_lottie_json(sticker_path):
        try:
            file_path = os.path.join(app.static_folder, sticker_path)
            if not os.path.exists(file_path): return jsonify({"error": "File non trovato"}), 404
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/prepare_subject', methods=['POST'])
    def prepare_subject():
        try:
            if 'subject_image' not in request.files:
                return jsonify(error="Immagine soggetto mancante."), 400
            subject = normalize_image(Image.open(io.BytesIO(request.files['subject_image'].read())).convert("RGBA"))
            processed = remove(subject)
            buf = io.BytesIO()
            processed.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500

    @app.route('/create_scene', methods=['POST'])
    def create_scene():
        global pipe, canny_detector
        try:
            if 'subject_data' not in request.files or 'prompt' not in request.form:
                return jsonify(error="Dati mancanti"), 400
            ensure_pipeline_is_loaded()
            subject = normalize_image(Image.open(io.BytesIO(request.files['subject_data'].read())).convert("RGB"))
            mask = remove(subject, only_mask=True, post_process_mask=True)
            canny_map = canny_detector(subject, low_threshold=50, high_threshold=150)
            if canny_map.size != subject.size:
                canny_map = canny_map.resize(subject.size, Image.Resampling.LANCZOS)
            result = pipe(
                prompt=request.form['prompt'],
                image=subject,
                mask_image=mask,
                control_image=canny_map,
                width=subject.width,
                height=subject.height,
                controlnet_conditioning_scale=0.8,
                num_inference_steps=CFG_DETAIL_STEPS,
                strength=1.0,
                guidance_scale=10
            ).images[0]
            buf = io.BytesIO()
            result.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500
        finally:
            pipe, canny_detector = None, None; release_vram()

    @app.route('/detail_and_upscale', methods=['POST'])
    def detail_and_upscale():
        global pipe, canny_detector
        try:
            if 'scene_image' not in request.files:
                return jsonify(error="Immagine mancante"), 400
            enable_hires = request.form.get('enable_hires', 'true').lower() == 'true'
            denoise = float(request.form.get('tile_denoising_strength', 0.3))
            ensure_pipeline_is_loaded()
            scene = normalize_image(Image.open(io.BytesIO(request.files['scene_image'].read())).convert("RGB"))
            if enable_hires:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                upsampler = RealESRGANer(
                    scale=2,
                    model_path=os.path.join('models', 'RealESRGAN_x2plus.pth'),
                    model=model,
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=True
                )
                img_cv = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2BGR)
                output, _ = upsampler.enhance(img_cv, outscale=2)
                scene = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            canny_map = canny_detector(scene, low_threshold=50, high_threshold=150)
            if canny_map.size != scene.size:
                canny_map = canny_map.resize(scene.size, Image.Resampling.LANCZOS)
            result = pipe(
                prompt="",
                image=scene,
                mask_image=None,
                control_image=canny_map,
                width=scene.width,
                height=scene.height,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=CFG_DETAIL_STEPS,
                strength=denoise,
                guidance_scale=5
            ).images[0]
            buf = io.BytesIO()
            result.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500
        finally:
            pipe, canny_detector = None, None; release_vram()

    @app.route('/analyze_parts', methods=['POST'])
    def analyze_parts():
        global yolo_parser
        try:
            if 'image' not in request.files: return jsonify(error="Missing image"), 400
            ensure_yolo_parser_is_loaded()
            image_pil = normalize_image(Image.open(io.BytesIO(request.files['image'].read())).convert("RGB"))
            results = yolo_parser(image_pil)[0]
            return jsonify(parts=sorted(list(set([results.names[int(cls)] for cls in results.boxes.cls]))))
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500
        finally:
            yolo_parser = None; release_vram()

    @app.route('/generate_all_parts', methods=['POST'])
    def generate_all_parts():
        global yolo_parser, sam_predictor, pipe, canny_detector
        try:
            if 'image' not in request.files: return jsonify(error="Dati mancanti"), 400
            
            ensure_pipeline_is_loaded(); ensure_yolo_parser_is_loaded(); ensure_sam_predictor_is_loaded()
            
            prompts = json.loads(request.form.get('prompts'))
            current_image = normalize_image(Image.open(io.BytesIO(request.files['image'].read())).convert("RGB"))
            
            for part_name, prompt_text in prompts.items():
                if not prompt_text: continue
                mask = make_mask(current_image, (part_name,))
                if mask:
                    w, h = current_image.size
                    canny_map = canny_detector(current_image, low_threshold=50, high_threshold=150)
                    if canny_map.size != current_image.size:
                        canny_map = canny_map.resize(current_image.size, Image.Resampling.LANCZOS)
                    canny_array = np.array(canny_map)
                    mask_resized = mask.resize((w, h), Image.Resampling.LANCZOS)
                    mask_array = np.array(mask_resized.convert("L"))
                    canny_array[mask_array > 128] = 0
                    control_image = Image.fromarray(canny_array)
                    if DEBUG_MODE:
                        control_image.save(os.path.join(current_app.root_path, 'temp', f"debug_control_{part_name}.png"))
                    current_image = pipe(prompt=prompt_text, image=current_image, mask_image=mask_resized, control_image=control_image, width=w, height=h, controlnet_conditioning_scale=0.8, num_inference_steps=CFG_DETAIL_STEPS, strength=1.0, guidance_scale=10).images[0]
                else:
                    print(f" [ATTENZIONE] Maschera per '{part_name}' non generata, step saltato.")
            buf = io.BytesIO()
            current_image.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500
        finally:
            yolo_parser, sam_predictor, pipe, canny_detector = (None,) * 4; release_vram()

    @app.route('/detect_faces', methods=['POST'])
    def detect_faces():
        global face_analyzer
        try:
            if 'image' not in request.files: return jsonify({"error": "Immagine mancante."}), 400
            ensure_face_analyzer_is_loaded()
            image_pil = normalize_image(Image.open(io.BytesIO(request.files['image'].read())).convert("RGB"))
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            faces = face_analyzer.get(image_np)
            return jsonify({"faces": [{"id": i, "bbox": [int(c) for c in f.bbox]} for i, f in enumerate(faces)]})
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore rilevamento volti: {e}"}), 500
        finally:
            face_analyzer = None; release_vram()
            
    @app.route('/final_swap', methods=['POST'])
    def final_swap():
        global face_analyzer, face_swapper, face_restorer
        try:
            if 'target_image_high_res' not in request.files or 'source_face_image' not in request.files: return jsonify(error="Immagini mancanti."), 400
            ensure_face_analyzer_is_loaded(); ensure_face_swapper_is_loaded(); ensure_face_restorer_is_loaded()
            target_pil = normalize_image(Image.open(io.BytesIO(request.files['target_image_high_res'].read())).convert("RGB"))
            source_pil = normalize_image(Image.open(io.BytesIO(request.files['source_face_image'].read())).convert("RGB"))
            target_img_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
            source_img_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
            target_faces = face_analyzer.get(target_img_cv)
            source_faces = face_analyzer.get(source_img_cv)
            if not source_faces or not target_faces: return jsonify(error="Volti non trovati."), 400
            source_face_index = int(request.form.get('source_face_index', 0))
            target_face_index = int(request.form.get('target_face_index', 0))
            result_img = face_swapper.get(target_img_cv, target_faces[target_face_index], source_faces[source_face_index], paste_back=True)
            if face_restorer:
                _, _, result_img = face_restorer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.8)
            _, buf = cv2.imencode('.png', result_img)
            return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify(error=str(e)), 500
        finally:
            face_analyzer, face_swapper, face_restorer = (None,) * 3; release_vram()

    @app.after_request
    def add_pna_header(response):
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
        return response
    
    return app
