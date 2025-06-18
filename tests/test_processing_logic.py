import importlib
import io
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

# Ensure the app package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def stub_heavy_modules():
    dummy = types.ModuleType('dummy')
    modules = {
        'torch': dummy,
        'requests': dummy,
    }
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)

    gfpgan_mod = types.ModuleType('gfpgan')
    class GFPGANer:
        pass
    gfpgan_mod.GFPGANer = GFPGANer
    sys.modules.setdefault('gfpgan', gfpgan_mod)

    rembg_mod = types.ModuleType('rembg')
    def remove(*a, **kw):
        pass
    rembg_mod.remove = remove
    sys.modules.setdefault('rembg', rembg_mod)

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

    controlnet_aux_mod = types.ModuleType('controlnet_aux')
    class CannyDetector:
        def __call__(self, *a, **kw):
            return SimpleNamespace(size=(0, 0))
    controlnet_aux_mod.CannyDetector = CannyDetector
    sys.modules.setdefault('controlnet_aux', controlnet_aux_mod)

    rrdbnet_mod = types.ModuleType('basicsr.archs.rrdbnet_arch')
    class RRDBNet:
        pass
    rrdbnet_mod.RRDBNet = RRDBNet
    sys.modules.setdefault('basicsr.archs.rrdbnet_arch', rrdbnet_mod)

    realesrgan_mod = types.ModuleType('realesrgan')
    class RealESRGANer:
        pass
    realesrgan_mod.RealESRGANer = RealESRGANer
    sys.modules.setdefault('realesrgan', realesrgan_mod)

    ultralytics_mod = types.ModuleType('ultralytics')
    class YOLO:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return [SimpleNamespace(boxes=SimpleNamespace(cls=[], conf=[]), names={})]
    ultralytics_mod.YOLO = YOLO
    sys.modules.setdefault('ultralytics', ultralytics_mod)

    imageio_ffmpeg_mod = types.ModuleType('imageio_ffmpeg')
    def get_ffmpeg_exe():
        return ''
    imageio_ffmpeg_mod.get_ffmpeg_exe = get_ffmpeg_exe
    sys.modules.setdefault('imageio_ffmpeg', imageio_ffmpeg_mod)

    bisenet_mod = types.ModuleType('app.bisenet_model')
    class BiSeNet:
        pass
    bisenet_mod.BiSeNet = BiSeNet
    sys.modules.setdefault('app.bisenet_model', bisenet_mod)

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

    dotenv_mod = types.ModuleType('dotenv')
    def load_dotenv(*a, **kw):
        pass
    dotenv_mod.load_dotenv = load_dotenv
    sys.modules.setdefault('dotenv', dotenv_mod)


stub_heavy_modules()
server = importlib.import_module('app.server')


def _image_bytes(color='black'):
    img = Image.new('RGB', (10, 10), color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_process_create_scene_progress_and_output():
    subject_bytes = _image_bytes()
    progress = []
    dummy_output = Image.new('RGB', (10, 10), 'white')

    class DummyPipeline:
        def __call__(self, **kwargs):
            cb = kwargs.get('callback_on_step_end')
            if cb:
                cb(None, 0, 0, {})
            return SimpleNamespace(images=[dummy_output])

    with patch.object(server, 'ensure_pipeline_is_loaded'), \
         patch.object(server, 'pipe', DummyPipeline()), \
         patch.object(server, 'canny_detector', lambda img, **kw: Image.new('RGB', img.size)), \
         patch.object(server, 'remove', lambda *a, **kw: Image.new('L', (10, 10), 0)):
        result = server.process_create_scene(subject_bytes, 'test', progress_cb=progress.append)

    assert isinstance(result, Image.Image)
    assert progress and progress[-1] == 100


def test_process_final_swap_success():
    target_bytes = _image_bytes('red')
    source_bytes = _image_bytes('blue')
    progress = []

    class DummyFace:
        bbox = [0, 0, 5, 5]

    class DummyAnalyzer:
        def get(self, img):
            return [DummyFace(), DummyFace()]

    class DummySwapper:
        def get(self, *a, **kw):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class DummyRestorer:
        def enhance(self, img, **kw):
            return None, None, img

    with patch.object(server, 'ensure_face_analyzer_is_loaded'), \
         patch.object(server, 'ensure_face_swapper_is_loaded'), \
         patch.object(server, 'ensure_face_restorer_is_loaded'):
        server.face_analyzer = DummyAnalyzer()
        server.face_swapper = DummySwapper()
        server.face_restorer = DummyRestorer()
        result = server.process_final_swap(target_bytes, source_bytes, 0, 1, progress.append)

    assert isinstance(result, Image.Image)
    assert progress == [50, 100]


def test_process_final_swap_invalid_index():
    target_bytes = _image_bytes()
    source_bytes = _image_bytes()

    class DummyFace:
        bbox = [0, 0, 5, 5]

    class DummyAnalyzer:
        def get(self, img):
            return [DummyFace()]

    with patch.object(server, 'ensure_face_analyzer_is_loaded'), \
         patch.object(server, 'ensure_face_swapper_is_loaded'), \
         patch.object(server, 'ensure_face_restorer_is_loaded'):
        server.face_analyzer = DummyAnalyzer()
        server.face_swapper = type('S', (), {'get': lambda *a, **kw: np.zeros((10, 10, 3))})()
        server.face_restorer = None
        with pytest.raises(IndexError):
            server.process_final_swap(target_bytes, source_bytes, 0, 1)
