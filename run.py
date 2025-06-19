# run.py - VERSIONE AGGIORNATA CON CONTROLLO MODELLO OLLAMA

import os
import requests
import logging
from dotenv import load_dotenv
from app.server import create_app
from app.user_model import init_db

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione base del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_ollama_model_is_available():
    """
    Controlla e/o forza l'aggiornamento del modello Ollama in base alla configurazione.
    """
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    # Leggiamo il nuovo flag, di default è 'false' se non presente
    force_pull = os.getenv("OLLAMA_FORCE_PULL", "false").lower() == "true"

    if not ollama_base_url or not ollama_model:
        logger.warning("Variabili OLLAMA non impostate nel .env. Salto il controllo del modello.")
        return

    logger.info("--- INIZIO CONTROLLO OLLAMA ---")
    logger.info(f"Modello richiesto: '{ollama_model}'. Aggiornamento forzato: {force_pull}")

    try:
        # Controlliamo comunque i modelli esistenti per un log informativo
        tags_url = f"{ollama_base_url}/api/tags"
        response = requests.get(tags_url, timeout=30)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_found = any(m.get("name").startswith(ollama_model) for m in models)

        if model_found:
            logger.info(f"Il modello '{ollama_model}' è già presente localmente.")
        else:
            logger.warning(f"Il modello '{ollama_model}' non è presente localmente.")

        # Decidiamo se procedere con il pull
        if not model_found or force_pull:
            if force_pull:
                logger.info(f"L'aggiornamento forzato è attivo. Avvio il comando 'pull' per '{ollama_model}'...")
            else: # Eseguito solo se model_found è False
                logger.info(f"Avvio il download per il modello mancante '{ollama_model}'...")

            logger.info("Questa operazione potrebbe richiedere tempo...")
            pull_url = f"{ollama_base_url}/api/pull"
            pull_payload = {"name": ollama_model, "stream": False}
            
            pull_response = requests.post(pull_url, json=pull_payload, timeout=3600)
            pull_response.raise_for_status()

            logger.info(f"Risposta dal comando 'pull': {pull_response.json().get('status')}")
            logger.info("Ollama gestisce il download/aggiornamento in background.")
        
        else:
            logger.info("Il modello è già presente e l'aggiornamento forzato è disattivato. Salto il pull.")

    except requests.exceptions.RequestException as e:
        logger.error(f"ERRORE DI CONNESSIONE: Impossibile comunicare con Ollama. Dettagli: {e}")
    except Exception as e:
        logger.error(f"ERRORE IMPREVISTO durante la comunicazione con Ollama: {e}")
    
    logger.info("--- FINE CONTROLLO OLLAMA ---")
    
    
    
    
    
class PrivateNetworkAccessMiddleware:
    """Aggiunge l'header PNA richiesto da Chrome per le richieste di rete privata."""
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def custom_start_response(status, headers, exc_info=None):
            headers.append(("Access-Control-Allow-Private-Network", "true"))
            return start_response(status, headers, exc_info)
        return self.app(environ, custom_start_response)


# Questo blocco viene eseguito solo quando lanci "python run.py"
if __name__ == "__main__":
    # 0. Controlla e scarica il modello Ollama se necessario
    ensure_ollama_model_is_available()

    # 1. Inizializza il database (crea le tabelle e migra i dati se necessario)
    init_db()

    # 2. Crea l'unica istanza della nostra applicazione Flask
    app = create_app()

    # 3. Applica il middleware all'applicazione
    app.wsgi_app = PrivateNetworkAccessMiddleware(app.wsgi_app)

    # 4. Avvia il server
    logger.info(
        "[+] Avvio del server Flask su http://0.0.0.0:8765 (DEBUG ATTIVO)")
    app.run(host="0.0.0.0", port=8765, debug=True)