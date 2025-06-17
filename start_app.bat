@echo off
:: Script per avviare l'ambiente di sviluppo di FaceSwapApp con terminali colorati

:: Imposta il titolo della finestra principale dello script
title Avvio FaceSwapApp

:: Imposta il percorso della cartella del tuo progetto
set "PROJECT_PATH=C:\Users\lelus\Documents\FaceSwapApp"

:: Imposta il percorso per attivare l'ambiente virtuale
set "VENV_ACTIVATE=%PROJECT_PATH%\venv\Scripts\activate"

echo [+] Avvio del server Flask (terminale Acqua) e del worker Celery (terminale Verde)...

:: Colori: 0=Nero, A=Verde Chiaro, B=Acqua. Il primo carattere e' lo sfondo, il secondo il testo.

:: Avvia il primo terminale per il server Flask con uno schema di colori personalizzato
:: Sfondo Nero (0), Testo Acqua (B)
start "Flask Server" cmd /k "color 0B && %VENV_ACTIVATE% && python run.py"

:: Attende 2 secondi per dare tempo al primo terminale di avviarsi in modo pulito
timeout /t 2 /nobreak >nul

@echo off
:: ... (le altre righe restano uguali) ...

:: Avvia il secondo terminale per il worker Celery in modalità SOLO (la più stabile per Windows)
start "Celery Worker" cmd /k "color 0A && %VENV_ACTIVATE% && celery -A app.server.celery worker --loglevel=info -P solo"

:: ...
echo [+] Fatto. Le due finestre colorate sono state avviate.

exit