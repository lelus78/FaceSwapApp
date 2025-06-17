from dotenv import load_dotenv
from app.server import create_app
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)


class PrivateNetworkAccessMiddleware:
    """Add the PNA header required by Chrome."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):

        def custom_start_response(status, headers, exc_info=None):
            headers.append(("Access-Control-Allow-Private-Network", "true"))
            return start_response(status, headers, exc_info)

        return self.app(environ, custom_start_response)


flask_app = create_app()
flask_app.wsgi_app = PrivateNetworkAccessMiddleware(flask_app.wsgi_app)

if __name__ == "__main__":
    logging.info(
        "[+] Avvio del server Flask su http://0.0.0.0:8765 (DEBUG ATTIVO)")
    flask_app.run(host="0.0.0.0", port=8765, debug=True)
