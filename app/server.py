# ===================================================================================
# === AI FACE SWAP STUDIO PRO 2.0 - SERVER.PY - VERSIONE FINALE E CORRETTA ===
# ===================================================================================
import os
import cv2
import numpy as np
import io
import logging
import requests
import traceback
import gzip
import json

# Import delle librerie AI
import torch
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from rembg import remove
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from controlnet_aux import CannyDetector
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from flask import Flask, request, send_file, jsonify, render_template, current_app
from flask_cors import CORS

# Importa il Blueprint dal nuovo file e le librerie necessarie
from app.meme_studio import meme_bp
from dotenv import load_dotenv

# ===================================================================================
# === CONFIGURAZIONE GLOBALE ===
# ===================================================================================
CFG_MODEL_NAME = "sdxl-yamers-realistic5-v5Rundiffusion"
CFG_SAMPLER = "DPM++"
CFG_SCENE_STEPS = 35
CFG_SCENE_GUIDANCE = 12
CFG_UPSCALE_FACTOR = 1.5
CFG_DETAIL_STEPS = 20
CFG_OVERLAP = 128
# ===================================================================================

# --- INIZIALIZZAZIONE MODELLI GLOBALI ---
print(" [+] Inizializzazione modelli AI...");
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
model_path_swapper = os.path.join('models', 'inswapper_128.onnx'); face_swapper = insightface.model_zoo.get_model(model_path_swapper, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
gfpgan_model_path = os.path.join('models', 'GFPGANv1.4.pth'); face_restorer = GFPGANer(model_path=gfpgan_model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None) if os.path.exists(gfpgan_model_path) else None
scene_upsampler = None
try:
    esrgan_model_path = os.path.join('models', 'RealESRGAN_x2plus.pth')
    if os.path.exists(esrgan_model_path):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scene_upsampler = RealESRGANer(scale=2, model_path=esrgan_model_path, model=model, dni_weight=None, tile=400, tile_pad=10, pre_pad=0, half=True, gpu_id=0)
        print(" [+] Modello Real-ESRGAN x2 per upscaling caricato con successo.")
    else: print(" [ATTENZIONE] Modello RealESRGAN_x2plus.pth non trovato.")
except Exception as e: print(f" [ERRORE] Impossibile caricare Real-ESRGAN: {e}.")
pipe = None; canny_detector = None; current_model_type = None
print(" [+] Modelli base pronti. Le pipeline SDXL verranno caricate al primo utilizzo.")


# --- FUNZIONI HELPER ---
def ensure_pipeline_is_loaded():
    global pipe, canny_detector
    if pipe is not None: return True
    print(f" [INFO] Caricamento della pipeline...")
    try:
        model_path = os.path.join('models', 'checkpoints', CFG_MODEL_NAME)
        print(f" [INFO] Caricamento del checkpoint personalizzato da: {model_path}")
        if not os.path.isdir(model_path): print(f"[ERRORE GRAVE] La cartella del modello non esiste: {model_path}"); return False
        canny_detector = CannyDetector()
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True)
        if CFG_SAMPLER == "DPM++":
            print(" [INFO] Utilizzo dello scheduler DPMSolverMultistepScheduler (DPM++).")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing(); pipe.enable_model_cpu_offload()
        print(f" [INFO] Pipeline con checkpoint '{CFG_MODEL_NAME}' e scheduler '{CFG_SAMPLER}' caricata.")
        return True
    except Exception as e:
        print(f" [ERRORE] Impossibile caricare la pipeline: {e}"); traceback.print_exc(); return False

def normalize_image(img, max_dim=1024):
    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height: new_width = max_dim; new_height = int(height * (max_dim / width))
        else: new_height = max_dim; new_width = int(width * (max_dim / height))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

# --- APPLICATION FACTORY ---
def create_app():
    # CORREZIONE: Usiamo il costruttore semplice. Flask cercherà 'static' e 'templates'
    # automaticamente dentro la cartella 'app/', che è la nostra struttura.
    app = Flask(__name__)
    
    # Carichiamo la chiave API e la salviamo nella config dell'app
    load_dotenv()
    app.config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Imposta il limite a 16 Megabyte

    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    app.register_blueprint(meme_bp)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/lottie_json/<path:sticker_path>')
    def get_lottie_json(sticker_path):
        try:
            # CORREZIONE: Usa direttamente app.static_folder che ora punta al posto giusto
            file_path = os.path.join(app.static_folder, sticker_path)
            
            safe_path = os.path.abspath(file_path)
            if not safe_path.startswith(os.path.abspath(app.static_folder)):
                 return jsonify({"error": "Forbidden"}), 403
            if not os.path.exists(safe_path):
                return jsonify({"error": "File not found"}), 404
            
            with gzip.open(safe_path, 'rt', encoding='utf-8') as f:
                json_content = json.load(f)
            return jsonify(json_content)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route('/api/stickers')
    def get_stickers_api():
        # CORREZIONE: Il percorso ora punta alla cartella stickers dentro la cartella static di default
        sticker_dir = os.path.join(app.static_folder, 'stickers')
        sticker_data = []
        if not os.path.isdir(sticker_dir):
            return jsonify({"error": "La cartella degli sticker non esiste"}), 404
        
        for root, dirs, files in os.walk(sticker_dir):
            category_name = os.path.basename(root)
            if root == sticker_dir:
                category_name = "Generale"
            sticker_paths = []
            for file in sorted(files):
                if file.lower().endswith(('.png', '.webm', '.tgs')):
                    relative_dir = os.path.relpath(root, app.static_folder)
                    rel_path = os.path.join(relative_dir, file).replace("\\", "/")
                    sticker_paths.append(os.path.join('static', rel_path).replace("\\", "/"))
            if sticker_paths:
                sticker_data.append({ "category": category_name, "stickers": sticker_paths })
        return jsonify(sticker_data)

    @app.route('/detect_faces', methods=['POST'])
    def detect_faces():
        if 'image' not in request.files: return jsonify({"error": "Immagine mancante."}), 400
        try:
            image_bytes = request.files['image'].read()
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            faces = face_analyzer.get(image_np)
            face_data = []
            for i, face in enumerate(faces):
                bbox = [int(coord) for coord in face.bbox]
                face_data.append({"id": i, "bbox": bbox})
            return jsonify({"faces": face_data})
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore durante il rilevamento dei volti: {e}"}), 500

    @app.route('/prepare_subject', methods=['POST'])
    def prepare_subject():
        if 'subject_image' not in request.files: return jsonify({"error": "Immagine del soggetto mancante."}), 400
        try:
            image = Image.open(io.BytesIO(request.files['subject_image'].read())).convert("RGB")
            normalized_image = normalize_image(image, max_dim=1024)
            img_byte_arr = io.BytesIO()
            normalized_image.save(img_byte_arr, format='PNG')
            output_bytes = remove(img_byte_arr.getvalue())
            return send_file(io.BytesIO(output_bytes), mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore durante la preparazione: {e}"}), 500

    @app.route('/create_scene', methods=['POST'])
    def create_scene():
        if 'subject_data' not in request.files: return jsonify({"error": "Dati del soggetto mancanti."}), 400
        try:
            subject_img = Image.open(request.files['subject_data'].stream).convert("RGBA")
            prompt = request.form['prompt']
            if not ensure_pipeline_is_loaded(): return jsonify({"error": "Pipeline AI non caricata."}), 500
            width, height = subject_img.size
            dummy_image = Image.new("RGB", (width, height), "black")
            full_mask = Image.new("RGB", (width, height), "white")
            generated_bg = pipe(prompt=prompt, image=dummy_image, mask_image=full_mask, control_image=dummy_image, controlnet_conditioning_scale=0.0, num_inference_steps=CFG_SCENE_STEPS, strength=1.0, guidance_scale=CFG_SCENE_GUIDANCE, width=width, height=height).images[0]
            background_img = generated_bg.convert("RGBA")
            if background_img.size != subject_img.size:
                background_img = background_img.resize(subject_img.size, Image.Resampling.LANCZOS)
            final_scene = Image.alpha_composite(background_img, subject_img)
            buf = io.BytesIO(); final_scene.convert("RGB").save(buf, format='PNG'); buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore durante la creazione della scena: {e}"}), 500

    @app.route('/detail_and_upscale', methods=['POST'])
    def detail_and_upscale():
        if 'scene_image' not in request.files: return jsonify({"error": "Immagine della scena mancante."}), 400
        try:
            scene_image_pil = Image.open(io.BytesIO(request.files['scene_image'].read())).convert("RGB")
            enable_hires = request.form.get('enable_hires') == 'true'
            upscaled_image = scene_image_pil
            if enable_hires and scene_upsampler:
                scene_image_cv = cv2.cvtColor(np.array(scene_image_pil), cv2.COLOR_RGB2BGR)
                output, _ = scene_upsampler.enhance(scene_image_cv, outscale=CFG_UPSCALE_FACTOR)
                upscaled_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            detailing_prompt = "best quality, ultra-detailed, high-resolution, 8k, professional photography, sharp focus"
            tile_denoising_strength = float(request.form.get('tile_denoising_strength', 0.4))
            if not ensure_pipeline_is_loaded(): return jsonify({"error": "Pipeline AI (ControlNet) non caricata."}), 500
            tile_size = 768; overlap = CFG_OVERLAP; num_inference_steps = CFG_DETAIL_STEPS; guidance_scale = 7.5
            width, height = upscaled_image.size; final_image = Image.new("RGB", (width, height))
            control_image = canny_detector(upscaled_image)
            def get_tile_coords(dimension, tile_dim, overlap_dim):
                coords = [0]; step = tile_dim - overlap_dim
                while coords[-1] + step < dimension: coords.append(coords[-1] + step)
                last_coord = dimension - tile_dim
                if len(coords) == 1 or coords[-1] != last_coord:
                    if last_coord > 0: coords.append(last_coord)
                return sorted(list(set(coords))) if dimension > tile_dim else [0]
            x_coords = get_tile_coords(width, tile_size, overlap); y_coords = get_tile_coords(height, tile_size, overlap)
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    box = (x, y, x + tile_size, y + tile_size); tile = upscaled_image.crop(box)
                    mask_tile = Image.new("RGB", tile.size, "white"); control_image_tile = control_image.crop(box)
                    detailed_tile = pipe(prompt=detailing_prompt, image=tile, mask_image=mask_tile, control_image=control_image_tile, strength=tile_denoising_strength, controlnet_conditioning_scale=1.0, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=tile_size, height=tile_size).images[0]
                    final_image.paste(detailed_tile, (x, y))
            buf = io.BytesIO(); final_image.save(buf, format='PNG'); buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore durante l'upscaling: {e}"}), 500

    @app.route('/final_swap', methods=['POST'])
    def final_swap():
        if 'target_image_high_res' not in request.files or 'source_face_image' not in request.files: return jsonify({"error": "Immagini mancanti."}), 400
        try:
            source_face_index = int(request.form.get('source_face_index', -1))
            target_face_index = int(request.form.get('target_face_index', -1))
            if source_face_index == -1 or target_face_index == -1: return jsonify({"error": "Indici dei volti non validi."}), 400
            target_img = cv2.imdecode(np.frombuffer(request.files['target_image_high_res'].read(), np.uint8), cv2.IMREAD_COLOR)
            source_img = cv2.imdecode(np.frombuffer(request.files['source_face_image'].read(), np.uint8), cv2.IMREAD_COLOR)
            source_faces = face_analyzer.get(source_img)
            target_faces = face_analyzer.get(target_img)
            if not source_faces or source_face_index >= len(source_faces): return jsonify({"error": "Volto sorgente non trovato o indice non valido."}), 400
            if not target_faces or target_face_index >= len(target_faces): return jsonify({"error": "Volto di destinazione non trovato o indice non valido."}), 400
            selected_source_face = source_faces[source_face_index]
            selected_target_face = target_faces[target_face_index]
            result_img = face_swapper.get(target_img, selected_target_face, selected_source_face, paste_back=True)
            if face_restorer: _, _, result_img = face_restorer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.8)
            _, buf = cv2.imencode('.png', result_img)
            return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')
        except Exception as e:
            traceback.print_exc(); return jsonify({"error": f"Errore durante il face swap finale: {e}"}), 500

    @app.route('/enhance_prompt', methods=['POST'])
    def enhance_prompt():
        # CORREZIONE: Leggiamo la chiave dalla config dell'app
        api_key = current_app.config.get('GEMINI_API_KEY')
        
        if not api_key:
            return jsonify({"error": "Chiave API di Gemini non trovata o non configurata nel file .env"}), 400
            
        try:
            data = request.get_json()
            base64_image, user_prompt = data.get('image_data'), data.get('prompt_text')
            
            if not all([base64_image, user_prompt]):
                return jsonify({"error": "Dati mancanti"}), 400
                
            google_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            system_prompt = (f"You are an expert prompt engineer for AI image generators. Look at the people in the attached image. Your task is to create a detailed, photorealistic background scene for them based on the user's idea: '{user_prompt}'.\n**Crucially, your generated prompt must describe ONLY the background, the environment, and the lighting. DO NOT mention or describe people, figures, or subjects in your prompt.** Your prompt must create an empty stage for the people in the image to be placed into. **The entire response must be less than 75 tokens long.** Respond ONLY with the new, enhanced prompt. Do not add quotation marks.")
            
            payload = {"contents": [{"parts": [{"text": system_prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}]}]}
            response = requests.post(google_api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("candidates"):
                enhanced_prompt = result["candidates"][0]["content"]["parts"][0]["text"].strip().replace('"', '')
                return jsonify({"enhanced_prompt": enhanced_prompt})
            else:
                error_info = result.get("promptFeedback", {})
                return jsonify({"error": f"Gemini non ha restituito un prompt valido. Causa: {error_info}"}), 500
                
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Errore durante il miglioramento del prompt: {e}"}), 500

    return app