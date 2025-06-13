@echo off

:: =================================================================
:: Script per avviare il server Python con privilegi di Amministratore
:: =================================================================

REM --- Controllo dei Privilegi di Amministratore ---
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

REM Se il controllo fallisce (codice di errore diverso da 0), non siamo amministratori.
if '%errorlevel%' NEQ '0' (
    echo Richiesta dei privilegi di Amministratore in corso...
    goto UACPrompt
) else (
    goto gotAdmin
)

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"
:: --- Fine Controllo Privilegi ---


echo.
echo ===================================================
echo   Avvio di AI Face Swap Studio (come Amministratore)
echo ===================================================
echo.

REM Imposta la cartella del progetto.
set PROJECT_DIR=%~dp0

REM Naviga alla cartella del progetto
cd /d %PROJECT_DIR%

REM Controlla se l'ambiente virtuale esiste
if not exist "venv\Scripts\activate" (
    echo Errore: Ambiente virtuale non trovato.
    pause
    exit
)

echo [+] Attivazione dell'ambiente virtuale Python...
call venv\Scripts\activate

echo [+] Avvio del server Python...
REM Viene avviato nella stessa finestra, cosi vediamo subito i log
python ../run.py

echo.
echo Il server e' stato chiuso. Premi un tasto per uscire.
pause
