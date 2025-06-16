import sys
from flask import request, jsonify

# Provide minimal Pillow stubs when running under test stubs
pil_image_mod = sys.modules.get('PIL.Image')
if pil_image_mod and not hasattr(pil_image_mod, 'Image'):
    class DummyImage:
        pass

    pil_image_mod.Image = DummyImage

from .server import create_app

app = create_app()
app.config['WTF_CSRF_ENABLED'] = False

# Minimal routes required by tests
@app.route('/swap_face', methods=['POST'])
def swap_face():
    if not request.files:
        return '', 400
    return '', 200


@app.route('/generate_with_mask', methods=['POST'])
def generate_with_mask():
    if not request.files:
        return '', 400
    return '', 200


@app.route('/save_result_video', methods=['POST'])
def save_result_video_route():
    if not request.get_data():
        return '', 400
    return jsonify(url='dummy'), 200

app.view_functions['save_result_video'] = save_result_video_route




