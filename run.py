# run.py - MODALITÀ DEBUG
from app.server import create_app
import os

# Manteniamo il controllo sulla chiave API
# if not os.path.exists('api_key.txt'):
#     print(" [ATTENZIONE] File 'api_key.txt' non trovato. Ne creo uno vuoto.")
#     open('api_key.txt', 'w').close()

app = create_app()

if __name__ == '__main__':
    # --- ABBIAMO COMMENTATO WAITRESS ---
    # from waitress import serve
    # print(" [+] Avvio del server di produzione Waitress su http://0.0.0.0:8765 con 20 thread.")
    # serve(app, host='0.0.0.0', port=8765, threads=20)

    # --- E USIAMO IL SERVER DI SVILUPPO FLASK ---
    print(" [+] Avvio del server di sviluppo Flask su http://0.0.0.0:8765 (MODALITÀ DEBUG ATTIVA)")
    app.run(host='0.0.0.0', port=8765, debug=True)