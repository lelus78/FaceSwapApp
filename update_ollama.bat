@echo off
echo --- Script di Aggiornamento di Ollama ---
echo.

REM Step 1: Scarica l'immagine piu' recente di Ollama
echo [1/4] Sto scaricando l'ultima versione di ollama/ollama...
docker pull ollama/ollama
if %errorlevel% neq 0 (
    echo ERRORE: Il download dell'immagine e' fallito.
    goto :eof
)
echo.

REM Step 2: Ferma il container Ollama attualmente in esecuzione (se esiste)
echo [2/4] Sto fermando il container 'ollama' esistente...
docker stop ollama
echo.

REM Step 3: Rimuove il vecchio container
echo [3/4] Sto rimuovendo il vecchio container 'ollama'...
docker rm ollama
echo.

REM Step 4: Avvia un nuovo container con l'immagine aggiornata
echo [4/4] Sto avviando un nuovo container 'ollama' con l'immagine aggiornata...
docker run -d -p 11434:11434 --name ollama ollama/ollama
if %errorlevel% neq 0 (
    echo ERRORE: L'avvio del nuovo container e' fallito.
    goto :eof
)
echo.

echo SUCCESSO! Ollama e' stato aggiornato e avviato.
pause