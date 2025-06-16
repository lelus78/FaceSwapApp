import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

_USER_FILE = os.path.join(os.path.dirname(__file__), 'users.json')


def _load_users():
    if os.path.exists(_USER_FILE):
        with open(_USER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_users(data):
    with open(_USER_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)


_users = _load_users()


def create_user(username: str, password: str) -> bool:
    if username in _users:
        return False
    _users[username] = generate_password_hash(password)
    _save_users(_users)
    return True


def verify_user(username: str, password: str) -> bool:
    hash_ = _users.get(username)
    if not hash_:
        return False
    return check_password_hash(hash_, password)
