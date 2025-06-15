import os
from dotenv import load_dotenv
from app.server import create_app

load_dotenv()


# --- Blocco Middleware per il fix "Private Network Access" di Chrome ---
class PrivateNetworkAccessMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def custom_start_response(status, headers, exc_info=None):
            headers.append(('Access-Control-Allow-Private-Network', 'true'))
            return start_response(status, headers, exc_info)
        
        return self.app(environ, custom_start_response)
# --- Fine Blocco Middleware ---


# 1. Creiamo l'app Flask usando la nostra factory
flask_app = create_app()

# 2. Applichiamo il nostro middleware per risolvere i problemi di connessione con Chrome
flask_app.wsgi_app = PrivateNetworkAccessMiddleware(flask_app.wsgi_app)


# 3. Avviamo il server
if __name__ == '__main__':
    # Usiamo il server di sviluppo di Flask che fornisce più dettagli in caso di errori
    print(" [+] Avvio del server di sviluppo Flask su http://0.0.0.0:8765 (MODALITÀ DEBUG ATTIVA)")
    flask_app.run(host='0.0.0.0', port=8765, debug=True)