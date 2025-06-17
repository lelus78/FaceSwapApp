from functools import wraps
from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for,
)
# Assicurati di importare entrambi i form
from .forms import RegistrationForm, LoginForm
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
    # Usa il LoginForm
    form = LoginForm()
    
    # Usa validate_on_submit per gestire il POST e la validazione
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data.strip()
        
        if verify_user(username, password):
            session['user_id'] = username
            return redirect(url_for('home')) 
        else:
            # Aggiunge un errore generico se le credenziali non sono valide
            form.username.errors.append("Username o password non validi.")

    # Passa sempre il form al template (per le richieste GET o se la validazione fallisce)
    return render_template('login.html', title='Login', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    # Usa il RegistrationForm
    form = RegistrationForm()
    
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data.strip()

        if not create_user(username, password):
            # Aggiunge un errore specifico se l'utente esiste già
            form.username.errors.append('Questo username è già in uso.')
        else:
            # Fa il login e reindirizza
            session['user_id'] = username
            return redirect(url_for('home'))

    # Passa sempre il form al template
    return render_template('register.html', title='Registrazione', form=form)


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))