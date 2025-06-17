# app/user_model.py (versione con DEBUG)

import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Definiamo il percorso del nostro "database" JSON
USER_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')

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
    print("\n--- DEBUG: DENTRO _save_users ---")
    print(f"Percorso del file target: {os.path.abspath(USER_FILE)}") # Mostra il percorso completo
    print(f"Dati che sto per salvare: {users}")
    try:
        with open(USER_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        print("--- DEBUG: Salvataggio completato con successo! ---\n")
    except Exception as e:
        print(f"\n!!! DEBUG: ERRORE DURANTE IL SALVATAGGIO DEL FILE: {e} !!!\n")


def create_user(username, password):
    """Crea un nuovo utente, esegue l'hashing della password e lo salva."""
    print(f"\n--- DEBUG: Chiamata a create_user per l'utente: '{username}' ---")
    users = _load_users()
    if username in users:
        print(f"L'utente '{username}' esiste già. Creazione fallita.")
        return False

    hashed_password = generate_password_hash(password)
    users[username] = {'password': hashed_password}
    
    _save_users(users)
    print(f"Utente '{username}' aggiunto. Ritorno True.")
    return True

def verify_user(username, password):
    """Verifica se l'username esiste e se la password fornita è corretta."""
    print(f"\n--- DEBUG: Chiamata a verify_user per l'utente: '{username}' ---")
    users = _load_users()
    print(f"Utenti caricati per la verifica: {list(users.keys())}")
    
    if username not in users:
        print("Utente non trovato nel dizionario.")
        return False

    hashed_password = users[username].get('password')
    
    if check_password_hash(hashed_password, password):
        print("Verifica password riuscita!")
        return True
    
    print("Verifica password fallita.")
    return False