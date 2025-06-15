import importlib
import os
import sys
import types
import pytest

# Ensure the app package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Helper to stub heavy modules before importing app.server_mod

def stub_heavy_modules():
    dummy = types.ModuleType('dummy')
    modules = {
        'cv2': dummy,
        'numpy': dummy,
        'torch': dummy,
        'requests': dummy,
    }
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)

    # gfpgan
    gfpgan_mod = types.ModuleType('gfpgan')
    class GFPGANer:
        pass
    gfpgan_mod.GFPGANer = GFPGANer
    sys.modules.setdefault('gfpgan', gfpgan_mod)

    # rembg
    rembg_mod = types.ModuleType('rembg')
    def remove(*a, **kw):
        pass
    rembg_mod.remove = remove
    sys.modules.setdefault('rembg', rembg_mod)

    # diffusers
    diffusers_mod = types.ModuleType('diffusers')
    class StableDiffusionXLControlNetInpaintPipeline:
        pass
    class ControlNetModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()
    class DPMSolverMultistepScheduler:
        @classmethod
        def from_config(cls, *a, **kw):
            return cls()
    diffusers_mod.StableDiffusionXLControlNetInpaintPipeline = StableDiffusionXLControlNetInpaintPipeline
    diffusers_mod.ControlNetModel = ControlNetModel
    diffusers_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
    sys.modules.setdefault('diffusers', diffusers_mod)

    # controlnet_aux
    controlnet_aux_mod = types.ModuleType('controlnet_aux')
    class CannyDetector:
        def __call__(self, *a, **kw):
            return types.SimpleNamespace(size=(0,0))
    controlnet_aux_mod.CannyDetector = CannyDetector
    sys.modules.setdefault('controlnet_aux', controlnet_aux_mod)

    # basicsr.archs.rrdbnet_arch
    rrdbnet_mod = types.ModuleType('basicsr.archs.rrdbnet_arch')
    class RRDBNet:
        pass
    rrdbnet_mod.RRDBNet = RRDBNet
    sys.modules.setdefault('basicsr.archs.rrdbnet_arch', rrdbnet_mod)

    # realesrgan
    realesrgan_mod = types.ModuleType('realesrgan')
    class RealESRGANer:
        pass
    realesrgan_mod.RealESRGANer = RealESRGANer
    sys.modules.setdefault('realesrgan', realesrgan_mod)

    # ultralytics
    ultralytics_mod = types.ModuleType('ultralytics')
    class YOLO:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(cls=[], conf=[]), names={})]
    ultralytics_mod.YOLO = YOLO
    sys.modules.setdefault('ultralytics', ultralytics_mod)

    # imageio_ffmpeg
    imageio_ffmpeg_mod = types.ModuleType('imageio_ffmpeg')
    def get_ffmpeg_exe():
        return ''
    imageio_ffmpeg_mod.get_ffmpeg_exe = get_ffmpeg_exe
    sys.modules.setdefault('imageio_ffmpeg', imageio_ffmpeg_mod)

    # Stub app.bisenet_model
    bisenet_mod = types.ModuleType('app.bisenet_model')
    class BiSeNet:
        pass
    bisenet_mod.BiSeNet = BiSeNet
    sys.modules.setdefault('app.bisenet_model', bisenet_mod)

    # Stub insightface with minimal structure
    insightface = types.ModuleType('insightface')
    app_mod = types.ModuleType('insightface.app')
    class FaceAnalysis:
        def __init__(self, *a, **kw):
            pass
        def prepare(self, *a, **kw):
            pass
        def get(self, *a, **kw):
            return []
    app_mod.FaceAnalysis = FaceAnalysis
    model_zoo_mod = types.ModuleType('insightface.model_zoo')
    def get_model(*a, **kw):
        return None
    model_zoo_mod.get_model = get_model
    sys.modules.setdefault('insightface', insightface)
    sys.modules.setdefault('insightface.app', app_mod)
    sys.modules.setdefault('insightface.model_zoo', model_zoo_mod)

    # Stub PIL and submodules
    pil_mod = types.ModuleType('PIL')
    for sub in ['Image', 'ImageDraw', 'ImageOps', 'ImageFilter']:
        sub_mod = types.ModuleType(f'PIL.{sub}')
        sys.modules.setdefault(f'PIL.{sub}', sub_mod)
    sys.modules.setdefault('PIL', pil_mod)

    # Stub dotenv
    dotenv_mod = types.ModuleType('dotenv')
    def load_dotenv(*a, **kw):
        pass
    dotenv_mod.load_dotenv = load_dotenv
    sys.modules.setdefault('dotenv', dotenv_mod)

@pytest.fixture
def app():
    stub_heavy_modules()
    server_mod = importlib.import_module('app.server_mod')
    server_mod.app.config['TESTING'] = True
    return server_mod.app

@pytest.fixture
def client(app):
    return app.test_client()

def test_swap_face_missing_files(client):
    res = client.post('/swap_face', data={})
    assert res.status_code == 400

def test_generate_with_mask_missing_files(client):
    res = client.post('/generate_with_mask', data={})
    assert res.status_code == 400

def test_blueprint_endpoint(client):
    res = client.post('/meme/generate_caption', json={})
    assert res.status_code == 400
