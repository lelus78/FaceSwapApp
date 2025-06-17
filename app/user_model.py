import json
import os

import logging
from werkzeug.security import generate_password_hash, check_password_hash

logging.basicConfig(level=logging.INFO)

# Definiamo il percorso del nostro "database" JSON
USER_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')
logger = logging.getLogger(__name__)



class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password = Column(String, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_from_json()


def migrate_from_json():
    if not os.path.exists(USER_FILE):
        return
    session = SessionLocal()
    try:
        if session.query(User).first():
            return
        with open(USER_FILE, 'r') as f:
            data = json.load(f)
        for name, info in data.items():
            session.add(User(username=name, password=info.get('password', '')))
        session.commit()
    finally:
        session.close()



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
    finally:
        session.close()


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
