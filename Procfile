web: waitress-serve --host=0.0.0.0 --port=8765 app.wsgi:app
worker: celery -A app.server.celery worker --loglevel=info