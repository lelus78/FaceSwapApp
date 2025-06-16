
from functools import wraps
from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    session,
    redirect,
    url_for,
)

from .user_model import create_user, verify_user

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('auth.login'))
        return func(*args, **kwargs)

    return wrapper


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json(silent=True) or {}
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            if verify_user(username, password):
                session['user_id'] = username
                return jsonify(message=f"Welcome {username}"), 200
            return jsonify(error='invalid credentials'), 401
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if verify_user(username, password):
            session['user_id'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error='Credenziali non valide'), 401
    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return render_template('register.html', error='Dati mancanti'), 400
        if not create_user(username, password):
            return render_template('register.html', error='Utente esistente'), 400
        session['user_id'] = username
        return redirect(url_for('home'))
    return render_template('register.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))
