# run_celery_dev.py
import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv # <--- AGGIUNGI QUESTO IMPORT

# Carica le variabili d'ambiente dal file .env
# Assicurati che il file .env sia nella directory radice del progetto
# o specifica il percorso se è altrove, es: load_dotenv(dotenv_path='../.env')
project_root = os.path.dirname(os.path.abspath(__file__)) # Assumendo che run_celery_dev.py sia nella root
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    print(f"--- Caricamento variabili d'ambiente da: {dotenv_path} ---")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"--- ATTENZIONE: File .env non trovato in {dotenv_path}. DEBUG_MODE potrebbe non essere attivo per Celery. ---")
    # Puoi decidere se uscire o continuare con le variabili d'ambiente di sistema
    # exit(1) # Opzionale: esci se .env è critico


# Comando per avviare il worker Celery (adattalo se necessario)
CELERY_COMMAND = [
    sys.executable,  # Usa lo stesso python della venv
    "-m", "celery",
    "-A", "app.server.celery", # <--- ASSICURATI CHE SIA CORRETTO
    "worker",
    "-l", "info",
    "-P", "solo" # Pool 'solo' è consigliato per il debug su Windows
]

# Path da monitorare
# Se server.py è in app/, allora ./app è corretto
PATH_TO_WATCH = "./app" # Assumendo che run_celery_dev.py sia nella root del progetto

worker_process = None

def start_worker():
    """Avvia il processo del worker Celery."""
    global worker_process
    print("--- Avvio del worker Celery... ---")
    # Passa le variabili d'ambiente correnti al sottoprocesso
    # Questo è importante perché load_dotenv() modifica os.environ del processo corrente
    env = os.environ.copy()
    worker_process = subprocess.Popen(CELERY_COMMAND, env=env) # <--- PASSA env
    print(f"--- Worker avviato con PID: {worker_process.pid} ---")
    # Log per verificare DEBUG_MODE nel contesto di avvio del worker
    print(f"--- DEBUG_MODE nel contesto di run_celery_dev.py (dopo load_dotenv): {os.getenv('DEBUG_MODE')} ---")

def restart_worker():
    """Riavvia il processo del worker Celery."""
    global worker_process
    if worker_process:
        print("--- Modifica rilevata. Riavvio del worker Celery... ---")
        worker_process.terminate()
        try:
            worker_process.wait(timeout=5) # Attendi fino a 5 secondi
        except subprocess.TimeoutExpired:
            print("--- Timeout terminazione worker. Provo a killare... ---")
            worker_process.kill()
            worker_process.wait()
    start_worker()

class CodeChangeHandler(FileSystemEventHandler):
    """Handler per gestire gli eventi di modifica dei file."""
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            # Ignora modifiche ai file nella venv se PATH_TO_WATCH è troppo generico
            if "venv" in event.src_path or ".git" in event.src_path:
                return
            print(f"File Python modificato: {event.src_path}")
            restart_worker()

if __name__ == "__main__":
    start_worker()

    event_handler = CodeChangeHandler()
    observer = Observer()
    # Assicurati che PATH_TO_WATCH sia corretto rispetto a dove si trova server.py
    # Se run_celery_dev.py è nella root e server.py in app/server.py
    # allora PATH_TO_WATCH = "./app" è corretto.
    actual_path_to_watch = os.path.join(project_root, 'app')
    if not os.path.isdir(actual_path_to_watch):
        print(f"ERRORE: Path da monitorare '{actual_path_to_watch}' non trovato!")
        sys.exit(1)

    print(f"--- Watchdog sta monitorando la cartella '{actual_path_to_watch}' per modifiche... ---")
    observer.schedule(event_handler, actual_path_to_watch, recursive=True)
    observer.start()

    try:
        while True:
            if worker_process and worker_process.poll() is not None:
                print(f"--- Worker Celery terminato inaspettatamente con codice {worker_process.returncode}. Riavvio... ---")
                time.sleep(2) # Attendi un attimo prima di riavviare
                restart_worker()
            time.sleep(1)
    except KeyboardInterrupt:
        print("--- Fermando il worker e l'osservatore... ---")
        if worker_process:
            worker_process.terminate()
            try:
                worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_process.kill()
                worker_process.wait()
        observer.stop()
    observer.join()
    print("--- Uscita pulita. ---")