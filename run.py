# run.py

from dotenv import load_dotenv
from app.server import create_app
import logging
from app.user_model import init_db

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione base del logging
logging.basicConfig(level=logging.INFO)


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
    # 1. Inizializza il database (crea le tabelle e migra i dati se necessario)
    init_db()

    # 2. Crea l'unica istanza della nostra applicazione Flask
    app = create_app()

    # 3. Applica il middleware all'applicazione
    app.wsgi_app = PrivateNetworkAccessMiddleware(app.wsgi_app)

    # 4. Avvia il server
    logging.info(
        "[+] Avvio del server Flask su http://0.0.0.0:8765 (DEBUG ATTIVO)")
    app.run(host="0.0.0.0", port=8765, debug=True)