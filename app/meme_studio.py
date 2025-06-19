import logging
import requests
import traceback
import os
from flask import Blueprint, request, jsonify

# --- CREAZIONE DEL BLUEPRINT ---
logging.basicConfig(level=logging.INFO)
meme_bp = Blueprint("meme_api", __name__, url_prefix="/meme")


@meme_bp.route('/generate_caption', methods=['POST'])
def generate_caption_proxy():
    logging.info("[MEME STUDIO] Richiesta di generazione didascalia (Ibrido: Gemini -> Ollama)...")

    # Carica tutte le configurazioni dall'ambiente
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL")

    data = request.get_json()
    base64_image = data.get('image_data')
    tone = data.get('tone', 'scherzoso')

    if not base64_image:
        return jsonify({"error": "Dati immagine mancanti"}), 400

    fallback_message = None  # Messaggio di avviso da inviare al frontend
    clean_caption = ""

    # --- TENTATIVO 1: USARE GOOGLE GEMINI ---
    try:
        if not gemini_api_key or not gemini_model:
            raise ValueError("Configurazione Gemini non trovata, procedo con il fallback.")

        logging.info(f"Tentativo con API Google Gemini (Modello: {gemini_model})...")
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"
        
        prompt_text = f"Agisci come un comico esperto di cultura internet. Crea una didascalia geniale e inaspettata per l'immagine. Tono richiesto: {tone}. Rispondi solo con la didascalia."
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        }
        
        response = requests.post(gemini_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        clean_caption = result["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"')
        logging.info("Successo con Gemini!")

    except Exception as e:
        logging.warning(f"Chiamata a Gemini fallita ({e}). Eseguo il fallback su Ollama locale.")
        fallback_message = "Generazione eseguita con un modello locale a causa di un errore del servizio principale."

        # --- TENTATIVO 2 (FALLBACK): USARE OLLAMA ---
        try:
            logging.info(f"Tentativo con Ollama (Modello: {ollama_fallback_model})...")
            
            prompt_templates = {
                "scherzoso": "Sei un creatore di meme amichevole e spiritoso. Il tuo obiettivo è creare una battuta o un gioco di parole leggero e simpatico per l'immagine. Deve far sorridere. La risposta deve essere SOLO la didascalia, breve e diretta.",
                "sarcastico": "Incarna il sarcasmo puro. Scrivi una didascalia tagliente, secca e ironica che prenda in giro il soggetto dell'immagine o una situazione comune. La risposta DEVE essere solo la battuta sarcastica, senza alcuna spiegazione.",
                "epico": "Sei il narratore di un trailer cinematografico. Descrivi la scena con un tono grandioso, drammatico e solenne, come se fosse il momento clou di un film fantasy. Usa parole potenti. La tua risposta deve essere SOLO la frase epica.",
                "assurdo": "Pensa come un surrealista. Ignora il contesto logico dell'immagine e crea una didascalia completamente nonsense, bizzarra e inaspettata. Deve essere strana e far dire 'Cosa?!'. Rispondi SOLO con la frase assurda."
            }
            system_prompt = prompt_templates.get(tone, prompt_templates["scherzoso"])

            ollama_payload = {
                "model": ollama_fallback_model, "prompt": system_prompt, "images": [base64_image],
                "stream": False, "keep_alive": "30s"
            }
            
            ollama_response = requests.post(f"{ollama_base_url}/api/generate", json=ollama_payload, timeout=60)
            ollama_response.raise_for_status()
            
            result = ollama_response.json()
            clean_caption = result.get("response", "").strip().strip('"')
            logging.info("Successo con Ollama (fallback).")
        
        except Exception as ollama_e:
            logging.error(f"ERRORE: Anche il fallback su Ollama è fallito: {ollama_e}")
            return jsonify({"error": "Sia il servizio principale che quello di backup non sono disponibili."}), 503

    # Risposta finale al frontend
    return jsonify({
        "caption": clean_caption,
        "fallback_message": fallback_message
    })


@meme_bp.route('/generate_tags', methods=['POST'])
def generate_tags_proxy():
    logging.info("[MEME STUDIO] Richiesta di generazione tag (Ibrido: Gemini -> Ollama)...")

    # Carica tutte le configurazioni
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL")

    data = request.get_json()
    base64_image = data.get('image_data')

    if not base64_image:
        return jsonify({"error": "Dati immagine mancanti"}), 400

    fallback_message = None # Questo non verrà usato nel frontend per i tag, ma manteniamo la logica
    tags = []

    # --- TENTATIVO 1: USARE GOOGLE GEMINI ---
    try:
        if not gemini_api_key or not gemini_model:
            raise ValueError("Configurazione Gemini non trovata, procedo con il fallback.")

        logging.info(f"Tentativo con API Google Gemini per i tag (Modello: {gemini_model})...")
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"
        
        prompt_text = "Genera da 3 a 5 brevi tag pertinenti per l'immagine. Rispondi solo con i tag separati da virgola, senza nient'altro. Esempio: reazione, gatto, computer, divertente"
        
        payload = {"contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}]}
        
        response = requests.post(gemini_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        raw_tags = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        tags = [t.strip().lstrip("#") for t in raw_tags.split(",") if t.strip()]
        logging.info("Successo con Gemini per i tag!")

    except Exception as e:
        logging.warning(f"Chiamata a Gemini per i tag fallita ({e}). Eseguo il fallback su Ollama locale.")

        # --- TENTATIVO 2 (FALLBACK): USARE OLLAMA ---
        try:
            logging.info(f"Tentativo con Ollama per i tag (Modello: {ollama_fallback_model})...")
            system_prompt = "Analizza l'immagine e genera da 3 a 5 tag brevi, pertinenti per un meme. La tua risposta DEVE essere solo una lista di parole separate da virgola. Esempio: reazione, divertente, gatto. NON includere frasi o spiegazioni."
            
            ollama_payload = {
                "model": ollama_fallback_model, "prompt": system_prompt, "images": [base64_image],
                "stream": False, "keep_alive": "30s"
            }
            
            ollama_response = requests.post(f"{ollama_base_url}/api/generate", json=ollama_payload, timeout=60)
            ollama_response.raise_for_status()
            
            result = ollama_response.json()
            raw_tags = result.get("response", "").strip()
            tags = [t.strip().lstrip("#") for t in raw_tags.split(",") if t.strip()]
            logging.info("Successo con Ollama per i tag (fallback).")
        
        except Exception as ollama_e:
            logging.error(f"ERRORE: Anche il fallback su Ollama per i tag è fallito: {ollama_e}")
            return jsonify({"error": "Servizi AI non disponibili per la generazione dei tag."}), 503

    # Risposta finale al frontend
    return jsonify({"tags": tags[:5]})