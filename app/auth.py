from functools import wraps
from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for,
    request,
    jsonify
)
# Assicurati di importare entrambi i form
from .forms import RegistrationForm, LoginForm
from .user_model import create_user, verify_user

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def login_required(func):
    """
    Decorator per verificare che l'utente sia loggato.
    Gestisce sia le chiamate API (restituendo JSON) sia la navigazione standard (reindirizzando).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            # SOLUZIONE MIGLIORATA: Controlla il percorso della richiesta.
            # Se è una chiamata API, restituisci un errore JSON.
            if request.path.startswith('/api/') or request.path.startswith('/async/'):
                return jsonify(error="Autenticazione richiesta. Effettua il login per continuare."), 401
            
            # Altrimenti, reindirizza alla pagina di login.
            return redirect(url_for('auth.login'))
        
        return func(*args, **kwargs)

    return wrapper


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data.strip()
        
        if verify_user(username, password):
            # Rigenera la sessione per sicurezza
            if hasattr(session, "regenerate"):
                session.regenerate()
            else:
                session.clear()
            session['user_id'] = username
            return redirect(url_for('home'))
        else:
            form.username.errors.append("Username o password non validi.")

    return render_template('login.html', title='Login', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data.strip()

        if not create_user(username, password):
            form.username.errors.append('Questo username è già in uso.')
        else:
            if hasattr(session, "regenerate"):
                session.regenerate()
            else:
                session.clear()
            session['user_id'] = username
            return redirect(url_for('home'))

    return render_template('register.html', title='Registrazione', form=form)


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))