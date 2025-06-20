@echo off
:: =================================================================
:: == Script per avviare l'ambiente di sviluppo di FaceSwapApp    ==
:: == Lancia 4 terminali: Redis, Flask, Celery e Vite.            ==
:: =================================================================
setlocal

:: Imposta il titolo della finestra principale dello script
title Avvio Sviluppo Completo FaceSwapApp

:: Imposta il percorso della cartella del tuo progetto
set "PROJECT_PATH=C:\Users\lelus\Documents\FaceSwapApp"

:: Imposta il percorso per attivare l'ambiente virtuale Python
set "VENV_ACTIVATE=%PROJECT_PATH%\venv\Scripts\activate"

:: Verifica che la cartella del progetto esista
if not exist "%PROJECT_PATH%" (
    echo [ERRORE] Il percorso del progetto non e' stato trovato:
    echo %PROJECT_PATH%
    echo Modifica lo script e correggi il percorso.
    pause
    exit /b 1
)

echo [+] Avvio dei quattro ambienti di sviluppo...

:: 1. Server Redis su WSL/Ubuntu (Terminale Rosso)
echo [+] Avvio del server Redis su Ubuntu (WSL)...
start "Redis Server (WSL)" wsl.exe bash -c "sudo service redis-server start && echo 'Redis avviato con successo. Questo terminale puo'' essere minimizzato.' && exec bash"

:: Attende 3 secondi per dare tempo a Redis di avviarsi completamente
timeout /t 3 /nobreak >nul

:: 2. Server Backend Flask (Terminale Acqua)
echo [+] Avvio del server Flask...
start "Flask Server" cmd /k "color 0B && cd /d "%PROJECT_PATH%" && %VENV_ACTIVATE% && python run.py"

:: Attende 2 secondi
timeout /t 2 /nobreak >nul

:: 3. Worker Celery (Terminale Verde)
echo [+] Avvio del worker Celery con Watchdog...
start "Celery Worker (+ Watchdog)" cmd /k "color 0A && cd /d "%PROJECT_PATH%" && %VENV_ACTIVATE% && python run_celery_dev.py"

:: Attende 2 secondi
timeout /t 2 /nobreak >nul

:: 4. Server Frontend Vite (Terminale Viola)
echo [+] Avvio del server di sviluppo Vite per Vue.js...
start "Vite Frontend Server" cmd /k "color 0D && cd /d "%PROJECT_PATH%\app\static" && npm run dev"


echo.
echo [SUCCESSO] I quattro ambienti di sviluppo sono stati avviati.
echo Questa finestra si chiudera' tra 5 secondi.
timeout /t 5 >nul

exit