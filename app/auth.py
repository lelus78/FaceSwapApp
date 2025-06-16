from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session

from .user_model import create_user, verify_user


auth_bp = Blueprint('auth', __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)

    return wrapped


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if verify_user(username, password):
            session['user_id'] = username
            return redirect(url_for('home'))
        error = 'Credenziali non valide'
    return render_template('login.html', error=error)


@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Inserisci username e password'
        elif create_user(username, password):
            session['user_id'] = username
            return redirect(url_for('home'))
        else:
            error = 'Utente già esistente'
    return render_template('register.html', error=error)
