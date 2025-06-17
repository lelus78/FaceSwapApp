import json
import os
from sqlalchemy import Column, Integer, String
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash

from .database import Base, SessionLocal, engine

USER_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')


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


def create_user(username: str, password: str) -> bool:
    session = SessionLocal()
    hashed = generate_password_hash(password)
    try:
        session.add(User(username=username, password=hashed))
        session.commit()
        return True
    except IntegrityError:
        session.rollback()
        return False
    finally:
        session.close()


def verify_user(username: str, password: str) -> bool:
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=username).first()
        if not user:
            return False
        return check_password_hash(user.password, password)
    finally:
        session.close()
