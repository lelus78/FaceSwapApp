# FaceSwapApp con IP-Adapter

Questa è un'applicazione web basata su Flask per eseguire operazioni di face swapping (sostituzione del volto) e generazione di immagini utilizzando il modello IP-Adapter. L'app consente di caricare un'immagine di input (con un volto) e un'immagine di destinazione, generando una nuova immagine in cui il volto dell'input viene trasferito sulla destinazione. Include anche un "Meme Studio" per creare meme e un sistema di prompt enhancement basato su LLM locali.

## Indice
- [Funzionalità Principali](#funzionalità-principali)
- [Architettura](#architettura)
- [Tech Stack](#tech-stack)
- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Configurazione dell'Ambiente](#configurazione-dellambiente)
- [Esecuzione dell'Applicazione](#esecuzione-dellapplicazione)
- [Struttura del Progetto](#struttura-del-progetto)

## Funzionalità Principali

- **Face Swapping**: Sostituisce il volto in un'immagine di destinazione con un volto da un'immagine di input.
- **Generazione Immagine da Testo (Text-to-Image)**: Genera immagini basate su un prompt di testo e un'immagine di input per lo stile.
- **Inpainting/Outpainting**: Modifica parti specifiche di un'immagine o ne estende i bordi.
- **Miglioramento Prompt**: Utilizza un modello linguistico multimodale locale (tramite **Ollama**) per migliorare i prompt testuali basandosi su un'immagine.
- **Meme Studio**: Un editor per creare meme aggiungendo testo e sticker alle immagini generate o caricate.
- **Galleria Utente**: Salva le immagini generate in una galleria personale.
- **Gestione Asincrona**: Le operazioni di generazione delle immagini vengono gestite in background con Celery e Redis per non bloccare l'interfaccia utente.

## Architettura

L'applicazione segue un'architettura client-server con un task queue per i processi a lunga esecuzione.

```
+----------------+      +---------------------+      +-------------------+
|                |      |                     |      |                   |
|  Browser Utente+------>  Flask Web Server   +------>  Celery Worker(s) |
| (HTML/JS/CSS)  |      |    (run.py)         |      | (Processamento AI)|
|                |      |                     |      |                   |
+-------+--------+      +----------+----------+      +---------+---------+
        |                          |                        ^
        |                          |                        |
        | (AJAX Requests)          | (Task)                 | (Polling Risultati)
        |                          |                        |
        v                          v                        v
+-------+--------+      +----------+----------+      +---------+---------+
|                |      |                     |      |                   |
|      API       <------+      Redis        <----+      Modelli AI      |
|  (Flask Routes)|      | (Message Broker &   |      |   (IP-Adapter,    |
|                |      |   Result Backend)   |      |   Ollama, etc.)   |
+----------------+      +---------------------+      +-------------------+
```

- **Frontend**: Pagina web interattiva costruita con HTML, CSS e JavaScript.
- **Backend (Flask)**: Gestisce le richieste HTTP, l'autenticazione e avvia i task di generazione.
- **Task Queue (Celery & Redis)**: Il server Flask invia i lavori pesanti (generazione AI) a un worker Celery tramite un broker Redis. Questo evita timeout HTTP e mantiene l'UI reattiva.
- **Worker(s) Celery**: Uno o più processi Python che ascoltano per nuovi task, li eseguono (caricando i modelli AI e processando le immagini) e salvano il risultato.
- **AI Models**: I modelli di deep learning (`IP-Adapter`, `Stable Diffusion`, `Ollama` per il linguaggio) che eseguono il lavoro vero e proprio.

## Tech Stack

| Componente      | Tecnologia Utilizzata                                      |
|-----------------|------------------------------------------------------------|
| **Backend** | Python, Flask                                              |
| **Frontend** | HTML, CSS, JavaScript (jQuery)                             |
| **Task Queue** | Celery, Redis                                              |
| **AI Models** | Diffusers (Hugging Face), IP-Adapter, PyTorch, InsightFace |
| **LLM Support** | **Ollama** (con un modello multimodale come LLaVA)         |
| **Database** | Flask-SQLAlchemy (SQLite per default)                      |
| **Deployment** | Gunicorn (consigliato per produzione)                      |


## Prerequisiti

- Python 3.9+
- `pip` e `venv`
- Redis (installato e in esecuzione)
- Un'istanza di **Ollama** con un modello multimodale (es. `llava`) scaricato e in esecuzione.

## Installazione

1.  **Clonare il repository:**
    ```bash
    git clone [https://github.com/lelus78/FaceSwapApp-ip-adapter.git](https://github.com/lelus78/FaceSwapApp-ip-adapter.git)
    cd FaceSwapApp-ip-adapter
    ```

2.  **Creare e attivare un ambiente virtuale:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```

3.  **Installare le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

## Configurazione dell'Ambiente

Prima di avviare l'applicazione, è necessario configurare le variabili d'ambiente.

1.  **Crea una copia del file di esempio:**
    ```bash
    # Questo comando non è disponibile, quindi crea manualmente un file .env
    ```

2.  **Crea e modifica il file `.env`** nella root del progetto e aggiungi le seguenti variabili:

    ```bash
    # --- CONFIGURAZIONE OLLAMA ---
    # L'URL base della tua istanza Ollama
    OLLAMA_BASE_URL="http://localhost:11434"
    
    # Il nome del modello multimodale che hai scaricato in Ollama (es: "llava", "bakllava")
    OLLAMA_MODEL="llava" 

    # (Opzionale) Abilita il logging verboso per il debug
    DEBUG_MODE=1

    # URL per il broker e il backend di Celery (default per Redis locale)
    CELERY_BROKER_URL="redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND="redis://127.0.0.1:6379/0"
    ```

## Esecuzione dell'Applicazione

Per eseguire l'app, devi avviare il server web Flask e almeno un worker Celery in due terminali separati. **Assicurati che Redis e Ollama siano già in esecuzione.**

**Terminale 1: Avviare il Server Flask**

```bash
source venv/bin/activate
python run.py
```
Il server sarà disponibile all'indirizzo `http://127.0.0.1:5000`.

**Terminale 2: Avviare il Worker Celery**

```bash
source venv/bin/activate
# L'opzione -P solo è utile per il debug su macOS e Windows
celery -A app.server.celery worker -l info -P solo
```

Il worker inizierà ad ascoltare per nuovi task di generazione immagini.

## Struttura del Progetto

```
.
├── app/                  # Core dell'applicazione Flask
│   ├── static/           # File statici (CSS, JS, immagini)
│   ├── templates/        # Template HTML (Jinja2)
│   ├── __init__.py       # Inizializzazione dell'app Flask e Celery
│   ├── server.py         # Routes principali, logica di business e definizione dei task Celery
│   ├── meme_studio.py    # Routes e logica per il Meme Studio
│   └── ...               # Altri moduli (modelli, helper, etc.)
├── models/               # Directory (da creare) per i modelli AI scaricati
├── outputs/              # Directory (da creare) dove vengono salvate le immagini generate
├── .env                  # File di configurazione delle variabili d'ambiente (da creare)
├── requirements.txt      # Dipendenze Python
├── run.py                # Script per avviare il server Flask
└── ...
```