import os
import io
import json
import time
import gzip
import traceback
import threading
import gc
import uuid

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis
from app.meme_studio import meme_bp

# Import for Stable Diffusion Inpainting
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler


# Ensure a temp directory exists for debug masks
DEBUG_MASK_DIR = 'temp/masks'
os.makedirs(DEBUG_MASK_DIR, exist_ok=True)

UNLOAD_DELAY = 600
_last_used = {}
_lock = threading.Lock()

face_analyzer = None
yolo_parser = None
stable_diffusion_model = None

def _touch(name):
    _last_used[name] = time.time()

def scale_segments(img1_shape, segments, img0_shape, normalize=False):
    gain_w = img0_shape[1] / img1_shape[1]
    gain_h = img0_shape[0] / img1_shape[0]

    scaled = []
    for s in segments:
        s = np.array(s, dtype=np.float32)
        s[:, 0] *= gain_w
        s[:, 1] *= gain_h
        scaled.append(s)
    return scaled

def get_face_analyzer():
    global face_analyzer
    with _lock:
        if face_analyzer is None:
            face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        _touch('face_analyzer')
    return face_analyzer

def get_yolo_parser():
    global yolo_parser
    with _lock:
        if yolo_parser is None:
            yolo_parser = YOLO(os.path.join('models', 'yolo-human-parse-v2.pt'))
        _touch('yolo_parser')
    return yolo_parser

def get_stable_diffusion_inpainting_model():
    """
    Function to load the Stable Diffusion Inpainting model.
    Now supports SDXL Inpainting.
    """
    global stable_diffusion_model
    with _lock:
        if stable_diffusion_model is None:
            print("Loading Stable Diffusion Inpainting Model (SDXL variant)...")
            try:
                # Per SDXL Inpainting, usa "stabilityai/stable-diffusion-xl-inpainting-0.1"
                # o "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                # Richiede molta più VRAM (es. 12GB+ GPU).
                model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                
                stable_diffusion_model = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None
                )
                
                stable_diffusion_model.scheduler = DPMSolverMultistepScheduler.from_config(stable_diffusion_model.scheduler.config)

                if torch.cuda.is_available():
                    stable_diffusion_model.to("cuda")
                    print(f"{model_id} loaded to CUDA.")
                else:
                    print(f"{model_id} loaded to CPU (CUDA not available).")
                
            except Exception as e:
                print(f"Error loading Stable Diffusion Inpainting Model ({model_id}): {e}")
                traceback.print_exc()
                class DummySDModel:
                    def __call__(self, prompt, image, mask_image, **kwargs):
                        print(f"FALLBACK: Dummy Inpainting (Error loading real model): Prompt='{prompt}', Mask applied. Returning red rectangle.")
                        img_array = np.array(image.convert("RGB"))
                        mask_array = np.array(mask_image.convert("L")) > 0
                        img_array[mask_array] = [255, 0, 0]
                        return [Image.fromarray(img_array)]
                stable_diffusion_model = DummySDModel()
        _touch('stable_diffusion_model')
    return stable_diffusion_model

def get_square_bbox(coords, image_width, image_height, padding_ratio=0.1):
    """
    Calculates a square bounding box around given coordinates with padding,
    ensuring it stays within image boundaries and its size is divisible by 8.
    Returns (x_min, y_min, x_max, y_max)
    """
    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]

    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Add padding, ensure it's an integer
    pad_x = int(bbox_width * padding_ratio)
    pad_y = int(bbox_height * padding_ratio)

    x_min_padded = max(0, x_min - pad_x)
    y_min_padded = max(0, y_min - pad_y)
    x_max_padded = min(image_width, x_max + pad_x)
    y_max_padded = min(image_height, y_max + pad_y)

    padded_width = x_max_padded - x_min_padded
    padded_height = y_max_padded - y_min_padded

    # Make it square
    side = max(padded_width, padded_height)

    # Ensure side is divisible by 8 and has a minimum size
    side = max(8, (side // 8) * 8) 
    if side == 0: side = 8 # Ensure minimum size

    # Adjust to keep centered and within bounds
    center_x = (x_min_padded + x_max_padded) / 2
    center_y = (y_min_padded + y_max_padded) / 2

    # Calculate final coordinates, cast to int
    x_min_final = int(center_x - side / 2)
    y_min_final = int(center_y - side / 2)
    x_max_final = int(center_x + side / 2)
    y_max_final = int(center_y + side / 2)

    # Re-adjust to fit within image boundaries after making square
    # Ensure final coordinates are integers
    x_min_final = max(0, x_min_final)
    y_min_final = max(0, y_min_final)
    x_max_final = min(image_width, x_min_final + side)
    y_max_final = min(image_height, y_min_final + side)

    # Final adjustment to ensure actual width/height match 'side'
    # This might shift the bbox slightly if it was clipped
    final_width_actual = x_max_final - x_min_final
    final_height_actual = y_max_final - y_min_final

    if final_width_actual < side:
        x_min_final = max(0, x_max_final - side)
    if final_height_actual < side:
        y_min_final = max(0, y_max_final - side)

    # Ensure final coordinates are integers again after last adjustments
    return (int(x_min_final), int(y_min_final), int(x_min_final + side), int(y_min_final + side))


def inpaint_image(original_image: Image.Image, mask_image: Image.Image, prompt: str, part_name: str, debug_save_mask=False):
    """
    Performs inpainting using the loaded Stable Diffusion model on a cropped region.
    """
    try:
        sd_model = get_stable_diffusion_inpainting_model()
        
        # Get original dimensions
        original_width, original_height = original_image.size

        # Find the bounding box for the specific part
        res = get_yolo_parser()(original_image.convert("RGB"), retina_masks=True)[0]
        if res.masks is None:
            print(f"No masks detected for image during inpainting of {part_name}.")
            return original_image # Return original if no masks found in YOLO

        polys = scale_segments(res.masks.data.shape[-2:], res.masks.xy, res.orig_shape)
        idx_map = {v: k for k, v in res.names.items()}
        target_cls_id = idx_map.get(part_name)

        target_poly_coords = None
        for i, poly in enumerate(polys):
            if res.boxes.cls[i].item() == target_cls_id:
                target_poly_coords = poly
                break
        
        if target_poly_coords is None:
            print(f"Could not find polygon for part '{part_name}'. Skipping inpainting for this part.")
            return original_image # Return original if part not found

        # Calculate square bounding box for the cropped region
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = get_square_bbox(
            target_poly_coords, original_width, original_height, padding_ratio=0.3
        )
        
        # Ensure the bbox dimensions are valid
        if bbox_x_max <= bbox_x_min or bbox_y_max <= bbox_y_min:
            print(f"Invalid bounding box calculated for {part_name}. Skipping inpainting.")
            return original_image


        # Crop the original image and the mask to the bounding box
        cropped_img = original_image.crop((int(bbox_x_min), int(bbox_y_min), int(bbox_x_max), int(bbox_y_max))).convert("RGB")
        cropped_mask = mask_image.crop((int(bbox_x_min), int(bbox_y_min), int(bbox_x_max), int(bbox_y_max))).convert("L")

        target_sd_size = 1024 # Fixed for SDXL as per your choice
        print(f"Using target_sd_size: {target_sd_size} for the inpainting model.")

        # Pad cropped image/mask to a square of target_sd_size
        padded_cropped_img = Image.new("RGB", (target_sd_size, target_sd_size), (0,0,0))
        padded_cropped_mask = Image.new("L", (target_sd_size, target_sd_size), 0)

        img_aspect = cropped_img.width / cropped_img.height
        if img_aspect > 1: # Wider than tall
            resize_width = target_sd_size
            resize_height = int(target_sd_size / img_aspect)
        else: # Taller than wide or square
            resize_height = target_sd_size
            resize_width = int(target_sd_size * img_aspect)

        resize_width = max(1, resize_width)
        resize_height = max(1, resize_height)

        resized_img_for_sd = cropped_img.resize((resize_width, resize_height), Image.LANCZOS)
        resized_mask_for_sd = cropped_mask.resize((resize_width, resize_height), Image.LANCZOS)

        x_paste = (target_sd_size - resize_width) // 2
        y_paste = (target_sd_size - resize_height) // 2

        padded_cropped_img.paste(resized_img_for_sd, (x_paste, y_paste))
        padded_cropped_mask.paste(resized_mask_for_sd, (x_paste, y_paste))


        if debug_save_mask:
            cropped_mask_filename = os.path.join(DEBUG_MASK_DIR, f"cropped_mask_{part_name}_{uuid.uuid4().hex}.png")
            padded_cropped_mask.save(cropped_mask_filename)
            print(f"Debug padded cropped mask saved to: {cropped_mask_filename}")

            cropped_img_filename = os.path.join(DEBUG_MASK_DIR, f"cropped_img_{part_name}_{uuid.uuid4().hex}.png")
            padded_cropped_img.save(cropped_img_filename)
            print(f"Debug padded cropped image saved to: {cropped_img_filename}")


        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        
        result_images_cropped = sd_model(
            prompt=prompt,
            image=padded_cropped_img,
            mask_image=padded_cropped_mask,
            width=target_sd_size,
            height=target_sd_size,
            strength=0.9,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images
        
        if result_images_cropped and len(result_images_cropped) > 0:
            generated_cropped_image_padded = result_images_cropped[0]
            
            target_bbox_width = int(bbox_x_max - bbox_x_min)
            target_bbox_height = int(bbox_y_max - bbox_y_min)

            generated_cropped_image = generated_cropped_image_padded.crop(
                (x_paste, y_paste, x_paste + resize_width, y_paste + resize_height)
            ).resize((target_bbox_width, target_bbox_height), Image.LANCZOS)

            final_image = original_image.copy()
            final_image.paste(generated_cropped_image, (int(bbox_x_min), int(bbox_y_min)))
            
            return final_image
        else:
            print("Stable Diffusion returned no images for the cropped region.")
            return original_image
    except Exception:
        traceback.print_exc()
        print("Error during inpainting process for a specific part.")
        return original_image

def make_mask(pil_img, parts_to_mask=('hair',)):
    """
    Generates a mask image for specified human body parts.
    
    Args:
        pil_img (PIL.Image): The input PIL image.
        parts_to_mask (tuple): A tuple of strings, e.g., ('hair', 'outfit').

    Returns:
        PIL.Image: A grayscale mask image (255 for masked, 0 for unmasked),
                   or None if no parts are found or an error occurs.
        List[Tuple]: A list of polygons (list of points) for the masked parts.
    """
    try:
        img_for_yolo = pil_img.convert("RGB")
        res = get_yolo_parser()(img_for_yolo, retina_masks=True)[0]

        if res.masks is None:
            return None, []

        polys = scale_segments(res.masks.data.shape[-2:], res.masks.xy, res.orig_shape)

        idx_map = {v: k for k, v in res.names.items()}
        target_idx = [idx_map[p] for p in parts_to_mask if p in idx_map]

        if not target_idx:
            print(f"No specified parts ({parts_to_mask}) found in YOLO model's names.")
            return None, []

        mask_img = Image.new("L", pil_img.size, 0)
        draw = ImageDraw.Draw(mask_img)

        masked_polygons = []
        for i, poly in enumerate(polys):
            if res.boxes.cls[i].item() in target_idx:
                int_poly = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(int_poly, fill=255)
                masked_polygons.append(int_poly)

        kernel = np.ones((60, 60), np.uint8) 
        mask_np = cv2.morphologyEx(np.array(mask_img), cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(mask_np), masked_polygons
    except Exception:
        traceback.print_exc()
        return None, []

def _cleaner():
    while True:
        time.sleep(30)
        with _lock:
            now = time.time()
            for n in list(_last_used):
                if now - _last_used[n] > UNLOAD_DELAY:
                    if n == 'face_analyzer':
                        global face_analyzer
                        face_analyzer = None
                    elif n == 'yolo_parser':
                        global yolo_parser
                        yolo_parser = None
                    elif n == 'stable_diffusion_model':
                        global stable_diffusion_model
                        if stable_diffusion_model is not None and hasattr(stable_diffusion_model, 'to'):
                            try:
                                stable_diffusion_model.to("cpu")
                                del stable_diffusion_model
                                stable_diffusion_model = None
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print("Stable Diffusion Model unloaded from GPU.")
                            except Exception as e:
                                print(f"Error unloading Stable Diffusion Model: {e}")
                                stable_diffusion_model = None
                        else:
                            stable_diffusion_model = None
                    
                    _last_used.pop(n, None)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

def create_app():
    load_dotenv()
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.register_blueprint(meme_bp)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/detect_faces', methods=['POST'])
    def detect_faces():
        try:
            if 'image' not in request.files:
                return jsonify(error="Missing image"), 400
            img_b = request.files['image'].read()
            arr = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
            analyzer = get_face_analyzer()
            faces = analyzer.get(arr)
            return jsonify(faces=[dict(bbox=list(map(int, f.bbox))) for f in faces])
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route('/analyze_parts', methods=['POST'])
    def analyze_parts():
        try:
            if 'image' not in request.files:
                return jsonify(error="Missing image"), 400
            img_b = request.files['image'].read()
            parser = get_yolo_parser()
            # Perform inference to get results
            results = parser(Image.open(io.BytesIO(img_b)), retina_masks=True)[0]
            
            # Extract detected class names
            detected_names = []
            if results.boxes is not None:
                # Get unique class IDs that were actually detected
                detected_class_ids = torch.unique(results.boxes.cls).cpu().tolist()
                # Map class IDs back to names
                for class_id in detected_class_ids:
                    if class_id in results.names:
                        detected_names.append(results.names[class_id])
            
            # Sort the names for consistent ordering in the frontend
            detected_names.sort()

            return jsonify(parts=detected_names)
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route('/generate_with_mask', methods=['POST'])
    def generate_with_mask():
        """
        Generates a mask for a SINGLE specified part and performs inpainting with a given prompt.
        """
        try:
            if 'image' not in request.files:
                return jsonify(error="Missing image"), 400
            img_b = request.files['image'].read()
            
            part_name = request.form.get('part_name')
            prompt = request.form.get('prompt')

            if not part_name or not prompt:
                return jsonify(error="Missing 'part_name' or 'prompt' for single mask generation."), 400
            
            pil_img = Image.open(io.BytesIO(img_b)).convert("RGB")
            mask_img, _ = make_mask(pil_img, parts_to_mask=(part_name,)) # Only need mask here
            
            if mask_img is None:
                return jsonify(error=f"Mask generation failed for part: {part_name}. Part not detected or internal error."), 500

            result_img = inpaint_image(pil_img, mask_img, prompt, part_name, debug_save_mask=True) # Pass part_name
            
            if result_img is None:
                return jsonify(error="Inpainting failed for this part. Check server logs."), 500

            buf = io.BytesIO()
            result_img.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route('/generate_all_parts', methods=['POST'])
    def generate_all_parts():
        """
        Generates modifications on an image for multiple specified parts using masks and prompts.
        """
        try:
            if 'image' not in request.files:
                return jsonify(error="Missing image"), 400
            
            img_b = request.files['image'].read()
            prompts_json = request.form.get('prompts')
            
            if not prompts_json:
                return jsonify(error="Missing 'prompts' data."), 400

            prompts = json.loads(prompts_json)
            
            if not isinstance(prompts, dict) or not prompts:
                return jsonify(error="Invalid or empty 'prompts' data."), 400

            current_image = Image.open(io.BytesIO(img_b)).convert("RGB")

            for part_name, prompt_text in prompts.items():
                print(f"Processing part: '{part_name}' with prompt: '{prompt_text}'")
                # Make mask for the *specific* part only for this iteration
                mask_img, _ = make_mask(current_image, parts_to_mask=(part_name,))
                
                if mask_img:
                    # Pass the specific part_name to inpaint_image
                    inpainted_part_image = inpaint_image(current_image, mask_img, prompt_text, part_name, debug_save_mask=True)
                    if inpainted_part_image:
                        current_image = inpainted_part_image
                    else:
                        print(f"Inpainting failed for part: {part_name}. Continuing with previous image.")
                else:
                    print(f"No mask generated for part: {part_name}. Skipping inpainting for this part.")

            buf = io.BytesIO()
            current_image.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')

        except json.JSONDecodeError:
            return jsonify(error="Invalid JSON for 'prompts'."), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route('/api/stickers')
    def stickers():
        base = os.path.join(app.static_folder, 'stickers')
        if not os.path.exists(base):
            return jsonify([]), 200

        data = []
        for root, _, files in os.walk(base):
            if root == base:
                continue
            cat = os.path.basename(root)
            items = [
                os.path.join('static', os.path.relpath(root, app.static_folder), f).replace("\\", "/")
                for f in files if f.lower().endswith(('.png', '.webm', '.tgs'))
            ]
            if items:
                data.append(dict(category=cat, stickers=sorted(items)))
        return jsonify(data)

    @app.route('/lottie_json/<path:p>')
    def lottie_json(p):
        try:
            fp = os.path.abspath(os.path.join(app.static_folder, p))
            if not fp.startswith(app.static_folder):
                return jsonify(error="Forbidden"), 403
            with gzip.open(fp, 'rt', encoding='utf-8') as g:
                return jsonify(json.load(g))
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500
            
    @app.route('/enhance_prompt', methods=['POST'])
    def enhance_prompt():
        try:
            data = request.get_json()
            user_prompt = data.get('prompt_text')

            if not user_prompt:
                return jsonify(error="Missing prompt_text"), 400

            enhanced_prompt = f"A high-quality, photorealistic image of {user_prompt}, intricate details, cinematic lighting, ultra HD."
            return jsonify(enhanced_prompt=enhanced_prompt)
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route('/enhance_part_prompt', methods=['POST'])
    def enhance_part_prompt():
        try:
            data = request.get_json()
            part_name = data.get('part_name')
            user_prompt = data.get('prompt_text')

            if not part_name or not user_prompt:
                return jsonify(error="Missing part_name or prompt_text"), 400

            enhanced_prompt = f"Highly detailed, realistic {part_name} in a {user_prompt} style, studio lighting, professional photography."
            return jsonify(enhanced_prompt=enhanced_prompt)
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500

    threading.Thread(target=_cleaner, daemon=True).start()
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8765, debug=True)