# app/user_model.py

import os
import json
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String

# --- IMPORTA LA CONFIGURAZIONE DAL TUO FILE database.py ---
from .database import Base, SessionLocal, engine

logger = logging.getLogger(__name__)

# --- MODELLO UTENTE (ora usa la 'Base' importata) ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)

# --- FUNZIONI DI GESTIONE (ora usano 'SessionLocal' e 'engine' importati) ---

def init_db():
    """Crea le tabelle del database e migra i dati da JSON se necessario."""
    logger.info("Inizializzazione del database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tabelle create.")
    migrate_from_json()

def migrate_from_json():
    """Migra gli utenti dal vecchio file users.json al database."""
    USER_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')
    if not os.path.exists(USER_FILE):
        return

    session = SessionLocal()
    try:
        if session.query(User).first():
            return

        logger.info("Avvio migrazione utenti da users.json...")
        with open(USER_FILE, 'r') as f:
            data = json.load(f)
        
        for name, info in data.items():
            user_hash = info.get('password')
            if user_hash:
                session.add(User(username=name, password_hash=user_hash))
        
        session.commit()
        logger.info("Migrazione completata.")
        os.rename(USER_FILE, USER_FILE + '.migrated')

    except Exception as e:
        logger.error(f"Errore durante la migrazione: {e}")
        session.rollback()
    finally:
        session.close()

def create_user(username, password):
    """Crea un nuovo utente nel database."""
    session = SessionLocal()
    try:
        if session.query(User).filter(User.username == username).first():
            return False
        
        new_user = User(username=username, password_hash=generate_password_hash(password))
        session.add(new_user)
        session.commit()
        return True
    finally:
        session.close()

def verify_user(username, password):
    """Verifica le credenziali di un utente contro il database."""
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            return False
        return check_password_hash(user.password_hash, password)
    finally:
        session.close()