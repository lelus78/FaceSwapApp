# app/user_model.py (versione con DEBUG)

import json
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash

logging.basicConfig(level=logging.INFO)

# Definiamo il percorso del nostro "database" JSON
USER_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')
logger = logging.getLogger(__name__)


def _load_users():
    """Funzione helper per caricare gli utenti dal file JSON."""
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_users(users):
    """Funzione helper per salvare il dizionario degli utenti nel file JSON."""
    logger.debug("DENTRO _save_users")
    logger.debug("Percorso del file target: %s", os.path.abspath(USER_FILE))
    logger.debug("Dati che sto per salvare: %s", users)
    try:
        with open(USER_FILE, "w") as f:
            json.dump(users, f, indent=4)
        logger.debug("Salvataggio completato con successo")
    except Exception as e:
        logger.exception("ERRORE DURANTE IL SALVATAGGIO DEL FILE: %s", e)


def create_user(username, password):
    """Crea un nuovo utente, esegue l'hashing della password e lo salva."""
    logger.debug("Chiamata a create_user per l'utente: '%s'", username)
    users = _load_users()
    if username in users:
        logger.debug("L'utente '%s' esiste già. Creazione fallita.", username)
        return False

    hashed_password = generate_password_hash(password)
    users[username] = {'password': hashed_password}

    _save_users(users)
    logger.debug("Utente '%s' aggiunto.", username)
    return True


def verify_user(username, password):
    """Verifica se l'username esiste e se la password fornita è corretta."""
    logger.debug("Chiamata a verify_user per l'utente: '%s'", username)
    users = _load_users()
    logger.debug("Utenti caricati per la verifica: %s", list(users.keys()))

    if username not in users:
        logger.debug("Utente non trovato nel dizionario.")
        return False

    hashed_password = users[username].get('password')

    if check_password_hash(hashed_password, password):
        logger.debug("Verifica password riuscita!")
        return True

    logger.debug("Verifica password fallita.")
    return False
