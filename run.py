# run.py
from app.server import create_app
from waitress import serve
import os

# Controlla se il file della chiave API esiste, altrimenti lo crea vuoto
if not os.path.exists('api_key.txt'):
    print(" [ATTENZIONE] File 'api_key.txt' non trovato. Ne creo uno vuoto.")
    open('api_key.txt', 'w').close()

app = create_app()

if __name__ == '__main__':
    print(" [+] Avvio del server di produzione Waitress su http://0.0.0.0:8765 con 20 thread.")
    serve(app, host='0.0.0.0', port=8765, threads=20)