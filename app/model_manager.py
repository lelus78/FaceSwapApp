import os
from flask import Blueprint, jsonify

MODEL_DIR = os.path.join("models", "checkpoints")
ACTIVE_MODEL_FILE = os.path.join("models", "active_model.txt")


def scan_available_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(
        d for d in os.listdir(MODEL_DIR)
        if os.path.isdir(os.path.join(MODEL_DIR, d))
    )


model_bp = Blueprint("model_manager", __name__, url_prefix="/api/models")


@model_bp.route("/list", methods=["GET"])
def list_models():
    return jsonify(scan_available_models())
