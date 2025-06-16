from flask import Blueprint, request, jsonify

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    if not username:
        return jsonify(error='username required'), 400
    return jsonify(message=f"Welcome {username}"), 200
