# app/meme_studio.py - VERSIONE CORRETTA
import requests
import traceback
# Importiamo 'current_app' da Flask per accedere alla config
from flask import Blueprint, request, jsonify, current_app

# --- CONFIGURAZIONE ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# --- CREAZIONE DEL BLUEPRINT ---
meme_bp = Blueprint('meme_api', __name__, url_prefix='/meme')

# --- ENDPOINT DEL MEME STUDIO ---
@meme_bp.route('/generate_caption', methods=['POST'])
def generate_caption_proxy():
    print("\n[MEME STUDIO] Richiesta di generazione didascalia...")

    # Leggiamo la chiave dalla config dell'app corrente
    api_key = current_app.config.get('GEMINI_API_KEY')

    if not api_key:
        return jsonify({"error": "Chiave API di Gemini non trovata o non configurata nel file .env"}), 400

    try:
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

        payload = {"contents": [{"parts": [{"text": system_prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}]}]}

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
