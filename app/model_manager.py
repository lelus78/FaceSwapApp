import os
from flask import Blueprint, jsonify, request

MODEL_DIR = os.path.join("models", "checkpoints")
ACTIVE_MODEL_FILE = os.path.join("models", "active_model.txt")


def scan_available_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(
        d for d in os.listdir(MODEL_DIR)
        if os.path.isdir(os.path.join(MODEL_DIR, d))
    )


def get_active_model():
    if os.path.isfile(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    return name
        except Exception:
            pass
    return None


model_bp = Blueprint("model_manager", __name__, url_prefix="/api/models")


@model_bp.route("/list", methods=["GET"])
def list_models():
    return jsonify({"models": scan_available_models(), "active": get_active_model()})


@model_bp.route("/activate", methods=["POST"])
def activate_model():
    data = request.get_json() or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name required"}), 400
    if model_name not in scan_available_models():
        return jsonify({"error": "model not found"}), 404
    os.makedirs(os.path.dirname(ACTIVE_MODEL_FILE), exist_ok=True)
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(model_name)
    return jsonify({"active_model": model_name})
