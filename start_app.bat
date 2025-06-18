@echo off
:: Script per avviare l'ambiente di sviluppo di FaceSwapApp con terminali colorati

:: Imposta il titolo della finestra principale dello script
title Avvio Sviluppo FaceSwapApp

:: Imposta il percorso della cartella del tuo progetto
set "PROJECT_PATH=C:\Users\lelus\Documents\FaceSwapApp"

:: Imposta il percorso per attivare l'ambiente virtuale
set "VENV_ACTIVATE=%PROJECT_PATH%\venv\Scripts\activate"

echo [+] Avvio del server Flask (terminale Acqua) e del gestore Celery con Watchdog (terminale Verde)...

:: Colori: 0=Nero, A=Verde Chiaro, B=Acqua. Il primo carattere e' lo sfondo, il secondo il testo.

:: Avvia il primo terminale per il server Flask con uno schema di colori personalizzato
:: Questo terminale si riavvia da solo quando modifichi il codice (grazie a Flask in debug mode)
start "Flask Server" cmd /k "color 0B && %VENV_ACTIVATE% && python run.py"

:: Attende 2 secondi per dare tempo al primo terminale di avviarsi in modo pulito
timeout /t 2 /nobreak >nul

:: --- RIGA MODIFICATA ---
:: Avvia il secondo terminale che esegue lo script 'run_celery_dev.py'.
:: Questo script si occupera' di avviare, monitorare e riavviare il worker Celery.
start "Celery Worker (+ Watchdog)" cmd /k "color 0A && %VENV_ACTIVATE% && python run_celery_dev.py"


echo [+] Fatto. I due ambienti di sviluppo sono stati avviati.

exit