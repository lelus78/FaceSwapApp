try:
    from .wsgi import wsgi_app
except Exception:  # Avoid import errors when optional deps are missing (e.g. tests)
    wsgi_app = None
