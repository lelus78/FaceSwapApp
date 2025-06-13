# ===================================================================================
# === MEME STUDIO - MODULO PER LA GESTIONE DI TESTO E STICKER ===
# ===================================================================================
import os
import requests
import traceback
from flask import Blueprint, request, jsonify, current_app
# ===================================================================================
# --- CONFIGURAZIONE ---
# In un'applicazione più grande, anche questa potrebbe essere in un file di config separato.
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# --- CREAZIONE DEL BLUEPRINT ---
# Creiamo un "gruppo" di rotte chiamato 'meme_studio'
# Aggiungiamo un prefisso '/meme' a tutte le rotte in questo file (es. /meme/generate_caption)
meme_bp = Blueprint('meme_api', __name__, url_prefix='/meme')


# --- ENDPOINT DEL MEME STUDIO ---

@meme_bp.route('/generate_caption', methods=['POST'])
def generate_caption_proxy():
    print("\n[MEME STUDIO] Richiesta di generazione didascalia...")
    try:
        # 1. Percorso robusto per trovare api_key.txt nella cartella principale dell'app
        api_key_path = os.path.join(current_app.root_path, 'api_key.txt')
        api_key = ""
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
        
        if not api_key:
            return jsonify({"error": "Chiave API di Gemini non trovata sul server."}), 400

        # ... il resto della funzione rimane invariato ...
        data = request.get_json()
        base64_image = data.get('image_data')
        tone = data.get('tone', 'scherzoso')

        if not base64_image: 
            return jsonify({"error": "Dati immagine mancanti per la didascalia."}), 400
        
        google_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
        
        tone_instructions = {
            "scherzoso": "divertente, sagace o ironico",
            "sarcastico": "estremamente sarcastico e pungente",
            "epico": "con un tono da trailer cinematografico",
            "assurdo": "completamente nonsense e surreale"
        }
        instruction = tone_instructions.get(tone, "scherzoso")

        system_prompt = (
            "Your only task is to generate a single, short caption for the following image. "
            "The caption must be in Italian and must be witty and impactful. "
            f"The desired tone is: **{instruction}**. "
            "Your entire response must consist ONLY of the caption text itself. "
            "Do not add any extra words, introductory phrases like 'Ecco la didascalia:', or quotation marks. "
            "Just the caption."
        )

        payload = {"contents": [{"parts": [{"text": system_prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image}}]}]}
        
        response = requests.post(google_api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result.get("candidates"):
            raw_caption = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            clean_caption = raw_caption.strip('"')
            return jsonify({"caption": clean_caption})
        else:
            error_info = result.get("promptFeedback", {})
            return jsonify({"error": f"Gemini non ha restituito una didascalia valida. Causa: {error_info}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Errore durante la generazione della didascalia: {e}"}), 500

# Qui in futuro potremmo aggiungere altre rotte, come:
# @meme_bp.route('/get_stickers', methods=['GET'])
# def get_stickers():
#     # ... logica per leggere i file degli sticker ...
#     pass