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
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from ultralytics import YOLO
from flask import Flask, request, send_file, jsonify, render_template, current_app, url_for
from flask_cors import CORS
from app.meme_studio import meme_bp
from dotenv import load_dotenv

# --- MODIFICA CHIAVE: IMPORT DAL NOSTRO FILE LOCALE ---
from app.bisenet_model import BiSeNet

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
load_dotenv()

# === GESTIONE DEI MODELLI ===
face_analyzer = None
face_swapper = None
face_restorer = None
inpainting_pipe = None
yolo_person_detector = None
sam_predictor = None
face_parser = None 

def ensure_face_analyzer_is_loaded():
    global face_analyzer
    if face_analyzer is None:
        print("[INFO] Caricamento FaceAnalysis...")
        face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] FaceAnalysis caricato.")

def ensure_face_swapper_is_loaded():
    global face_swapper
    if face_swapper is None:
        print("[INFO] Caricamento FaceSwapper (inswapper_128.onnx)...")
        face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=False)
        print("[INFO] FaceSwapper caricato.")

def ensure_face_restorer_is_loaded():
    global face_restorer
    if face_restorer is None:
        print("[INFO] Caricamento GFPGAN (Face Restorer)...")
        face_restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None)
        print("[INFO] GFPGAN caricato.")

def ensure_inpainting_pipe_is_loaded():
    global inpainting_pipe
    if inpainting_pipe is None:
        print("[INFO] Caricamento Stable Diffusion Inpainting Pipeline...")
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        inpainting_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        print("[INFO] Inpainting Pipeline caricato.")

def ensure_face_parser_is_loaded():
    global face_parser
    if face_parser is None:
        print("[INFO] Caricamento del modello di Face Parsing...")
        try:
            model_path = 'models/face_parsing/79999_iter.pth'
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Modello di Face Parsing non trovato in {model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            face_parser = BiSeNet(n_classes=19)
            face_parser.load_state_dict(torch.load(model_path, map_location=device))
            face_parser.to(device)
            face_parser.eval()
            print(f"[INFO] Modello di Face Parsing caricato su {device}.")
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare il modello di Face Parsing: {e}")
            face_parser = "ERROR"

# === UTILITY FUNCTIONS ===
def normalize_image(img):
    max_size = 1280
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    return img

def generate_hair_mask_from_parser(pil_image, blur_amount=5):
    if face_parser is None or face_parser == "ERROR":
        raise RuntimeError("Face parser non disponibile o non caricato correttamente.")

    w, h = pil_image.size
    img_resized = pil_image.resize((512, 512), Image.BILINEAR)
    img_np = np.array(img_resized)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
        img_tensor = (img_tensor - 127.5) / 128.0 # Normalize
        
        out = face_parser(img_tensor)[0]

    # La classe per i capelli (hair) nel modello è 17
    parsing_map = out.squeeze(0).cpu().numpy().argmax(0)
    hair_mask_np = (parsing_map == 17).astype(np.uint8) * 255

    hair_mask_pil = Image.fromarray(hair_mask_np).resize((w, h), Image.NEAREST)

    if blur_amount > 0:
        hair_mask_pil = hair_mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_amount))

    return hair_mask_pil.convert('L')


# === CONFIGURAZIONE FLASK APP ===
app = Flask(__name__, static_folder='static', template_folder='.')
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
app.register_blueprint(meme_bp)
CORS(app)

# === ENDPOINTS API ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/swap_face', methods=['POST'])
def swap_face():
    print("\n[FACE SWAP] Richiesta ricevuta...")
    try:
        if 'target_image_high_res' not in request.files or 'source_face_image' not in request.files:
            return jsonify(error="Immagini mancanti."), 400
            
        ensure_face_analyzer_is_loaded()
        ensure_face_swapper_is_loaded()
        ensure_face_restorer_is_loaded()

        target_pil = normalize_image(Image.open(io.BytesIO(request.files['target_image_high_res'].read())).convert("RGB"))
        source_pil = normalize_image(Image.open(io.BytesIO(request.files['source_face_image'].read())).convert("RGB"))

        target_img_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
        source_img_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)

        target_faces = face_analyzer.get(target_img_cv)
        source_faces = face_analyzer.get(source_img_cv)

        if not source_faces or not target_faces:
            return jsonify(error="Volti non trovati in una delle immagini."), 400

        result_img = face_swapper.get(target_img_cv, target_faces[0], source_faces[0], paste_back=True)

        if face_restorer:
            _, _, result_img = face_restorer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.8)

        _, buf = cv2.imencode('.png', result_img)
        return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"Errore interno del server: {str(e)}"), 500

@app.route('/generate_with_mask', methods=['POST'])
def generate_with_mask():
    print("\n[INPAINTING] Richiesta ricevuta...")
    try:
        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify(error="Richiesta incompleta: mancano immagine, maschera o prompt."), 400
        
        part_name = request.form.get('part_name', 'generic')
        prompt = request.form['prompt']
        image_pil = normalize_image(Image.open(io.BytesIO(request.files['image'].read())).convert("RGB"))
        
        mask_pil = None
        
        if part_name.lower() == 'hair':
            print("[INFO] Rilevata parte 'hair'. Uso il metodo di Face Parsing locale.")
            ensure_face_parser_is_loaded()
            if face_parser is None or face_parser == "ERROR":
                return jsonify(error="Il modello di Face Parsing non è disponibile. Controlla i log."), 500
            mask_pil = generate_hair_mask_from_parser(image_pil)
            print("[INFO] Maschera dei capelli generata con successo.")
        else:
            print(f"[INFO] Parte '{part_name}' gestita con rembg per una maschera generica del soggetto.")
            mask_pil = remove(image_pil, only_mask=True, post_process_mask=True)

        ensure_inpainting_pipe_is_loaded()

        image_for_pipe = image_pil.resize((1024, 1024))
        mask_for_pipe = mask_pil.resize((1024, 1024)).convert("RGB")
        
        canny_detector = CannyDetector()
        canny_image = canny_detector(image_for_pipe, 100, 200, output_type="pil")

        result_image = inpainting_pipe(
            prompt=prompt,
            negative_prompt="testo, scritte, watermark, brutte mani, sfigurato, deforme",
            image=image_for_pipe,
            mask_image=mask_for_pipe,
            control_image=canny_image,
            num_inference_steps=25,
            controlnet_conditioning_scale=0.8,
            guidance_scale=7.5
        ).images[0]

        buf = io.BytesIO()
        result_image.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"Errore durante l'inpainting: {str(e)}"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE)
