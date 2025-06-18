# run_celery_dev.py
import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Comando per avviare il worker Celery (adattalo se necessario)
CELERY_COMMAND = [
    sys.executable,  # Usa lo stesso python della venv
    "-m", "celery",
    "-A", "app.server.celery",
    "worker",
    "-l", "info",
    "-P", "solo" # Pool 'solo' è consigliato per il debug su Windows
]

# Path da monitorare
PATH_TO_WATCH = "./app"

worker_process = None

def start_worker():
    """Avvia il processo del worker Celery."""
    global worker_process
    print("--- Avvio del worker Celery... ---")
    # Usiamo Popen per avere il controllo sul processo
    worker_process = subprocess.Popen(CELERY_COMMAND)
    print(f"--- Worker avviato con PID: {worker_process.pid} ---")

def restart_worker():
    """Riavvia il processo del worker Celery."""
    global worker_process
    if worker_process:
        print("--- Modifica rilevata. Riavvio del worker Celery... ---")
        worker_process.terminate() # Termina il vecchio processo
        worker_process.wait()      # Attende che il processo sia terminato
    start_worker()

class CodeChangeHandler(FileSystemEventHandler):
    """Handler per gestire gli eventi di modifica dei file."""
    def on_modified(self, event):
        # Reagisce solo alla modifica di file Python
        if not event.is_directory and event.src_path.endswith('.py'):
            print(f"File modificato: {event.src_path}")
            restart_worker()

if __name__ == "__main__":
    # Avvia il worker per la prima volta
    start_worker()

    # Imposta e avvia l'osservatore di watchdog
    event_handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, PATH_TO_WATCH, recursive=True)
    
    print(f"--- Watchdog sta monitorando la cartella '{PATH_TO_WATCH}' per modifiche... ---")
    observer.start()

    try:
        # Tieni lo script principale in esecuzione
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Ferma tutto se l'utente preme Ctrl+C
        print("--- Fermando il worker e l'osservatore... ---")
        if worker_process:
            worker_process.terminate()
            worker_process.wait()
        observer.stop()
    observer.join()
    print("--- Uscita pulita. ---")