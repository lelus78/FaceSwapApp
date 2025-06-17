# 🎨 AI Face Swap Studio Pro 2.0

*Un workflow completo e guidato per la composizione di immagini a livello professionale — dal face swap alla generazione di scene AI e al ritocco creativo finale.*

> **Perché?**
> La maggior parte degli strumenti di "face-swap" si ferma allo scambio del volto.
> AFSS-Pro concatena diversi modelli allo stato dell'arte in modo che l'**output di una fase diventi l'input di quella successiva**, producendo un risultato completo, fotorealistico e pronto per la condivisione.

-----

## 🚀 Funzionalità Principali

| Categoria                     | Punti Salienti                                                                                                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workflow Guidato in 4 Passi** | L'interfaccia guida l'utente attraverso ogni fase, senza sorprese o errori improvvisi della GPU.                                                                                                          |
| **Workflow Asincrono** | La coda di task Celery elabora i processi AI pesanti in background, mantenendo l'interfaccia utente sempre reattiva e mostrando il progresso in tempo reale. |
| **Rimozione Sfondo Intelligente** | Isolamento automatico del soggetto tramite `rembg`.                                                                                                                                 |
| **Generazione Scena AI**       | Stable Diffusion XL (in-paint) trasforma i prompt testuali in sfondi realistici.                                                                                                       |
| **Miglioramento Prompt**       | Google Gemini analizza il soggetto e arricchisce il prompt dell'utente.                                                                                                                     |
| **Upscale e Dettaglio Hi-Res**  | Real-ESRGAN + ControlNet (Canny) ripristinano la nitidezza senza causare errori di memoria (OOM).                                                                                          |
| **Face Swap Mirato**          | InsightFace con selezione basata su indice per sorgente e destinazione.                                                                                                           |
| **Restauro Volto**            | GFPGAN per una coerenza finale della pelle e dei lineamenti.                                                                                                                  |
| **Studio Creativo Finale**     | \<ul\>\<li\>Controlli per testo e meme (font, dimensione, tratto, ecc.)\</li\>\<li\>Galleria di sticker (PNG · WebM · Lottie/.tgs) con drag‑rotate‑resize\</li\>\<li\>Esportazione come PNG **o** MP4 / GIF animati\</li\>\</ul\> |

-----

## 🛠️ Tech Stack

| Layer         | Librerie / Strumenti Principali                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **Backend**   | Python 3.9+, Flask, **Celery**, **Redis**, `python-dotenv`, `imageio-ffmpeg`                     |
| **AI Models** | PyTorch, Diffusers (SDXL + ControlNet), InsightFace, GFPGAN, Real‑ESRGAN, `rembg`, Google Gemini |
| **Frontend**  | HTML 5, Tailwind CSS, Vanilla ES 6 Modules, `<canvas>` API, `lottie-web`                         |

-----

## 🧠 Architettura

```
rembg  →  Gemini  →  SDXL  →  Real-ESRGAN  →  ControlNet  →  InsightFace  →  GFPGAN
(mask)    (prompt)   (scene)     (hi-res)      (detail)        (swap)        (restore)
```

  * **Worker Asincroni (Celery)** gestiscono i calcoli pesanti (GPU), liberando il server web e mantenendo l'interfaccia reattiva.
  * **Web Server (Flask)** si occupa di servire le pagine, gestire l'autenticazione e avviare i task in background.
  * **Broker di Messaggi (Redis)** gestisce la coda dei lavori tra Flask e Celery.
  * **Finishing lato Client** (testo, sticker, filtri) tramite l'API `<canvas>` per non sovraccaricare il server con micro-modifiche.
  * **Esportazione animata**: MediaRecorder cattura il canvas → WebM leggero → il backend converte in MP4 / GIF con `ffmpeg`.

-----

## ⚙️ Installazione e Avvio

### 1\. Prerequisiti

  * **Python 3.9+**
  * **Git**
  * **Redis**: Celery richiede un message broker per funzionare. Redis è la scelta più comune e raccomandata.
      * [Guida all'installazione di Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/) (per Linux/macOS)
      * Per Windows, è consigliato usare [Memurai](https://www.memurai.com/) o WSL. Assicurati che il servizio Redis sia in esecuzione.

### 2\. Setup del Progetto

```bash
git clone https://github.com/lelus78/FaceSwapApp.git
cd FaceSwapApp

# Crea un ambiente virtuale
python -m venv venv

# Attiva l'ambiente virtuale
# Su Windows
.\venv\Scripts\activate
# Su macOS / Linux
source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
```

### 3\. Download dei Modelli AI

Scarica i seguenti modelli pre-addestrati e posizionali nelle cartelle indicate.

| Modello                             | Percorso                  |
| --------------------------------- | ----------------------- |
| Checkpoint SDXL (`*.safetensors`) | `./models/checkpoints/` |
| InsightFace `inswapper_128.onnx`  | `./models/`             |
| GFPGANv1.4 & RealESRGAN\_x2plus   | `./models/`             |

> Gli altri modelli (ControlNet, analisi InsightFace, ecc.) verranno scaricati automaticamente al primo avvio.

### 4\. Configurazione dell'Ambiente

Crea un file `.env` nella cartella principale del progetto e aggiungi le seguenti variabili:

```bash
# Chiave API per Google Gemini
GEMINI_API_KEY="LA_TUA_CHIAVE_GOOGLE_GEMINI"

# (Opzionale) Abilita il logging verboso per il debug
DEBUG_MODE=1

# URL per il broker e il backend di Celery (default per Redis locale)
CELERY_BROKER_URL="redis://127.0.0.1:6379/0"
CELERY_RESULT_BACKEND="redis://127.0.0.1:6379/0"
```

### 5\. Avvio dell'Applicazione

L'applicazione ora richiede **due terminali separati**: uno per il server web Flask e uno per il worker Celery. Assicurati che l'ambiente virtuale `venv` sia attivo in entrambi.

#### Su Windows

**Terminale 1: Avvia il Server Flask**

```bash
python run.py
```

**Terminale 2: Avvia il Worker Celery**
(L'opzione `-P solo` è raccomandata per la massima stabilità su Windows)

```bash
celery -A app.server.celery worker -l info -P solo
```

#### Su macOS / Linux

**Terminale 1: Avvia il Server Flask**

```bash
python run.py
```

**Terminale 2: Avvia il Worker Celery**

```bash
celery -A app.server.celery worker -l info
```

-----

Una volta avviati entrambi i processi, vai su [http://127.0.0.1:8765](http://127.0.0.1:8765) nel tuo browser.

### API Asincrone Principali

| Rotta | Scopo |
| ----- | ------- |
| `/prepare_subject` | Rimuove lo sfondo e restituisce un PNG. (Sincrona e veloce) |
| `/async/create_scene` | Avvia un task in background per creare la scena AI. |
| `/async/detail_and_upscale` | Avvia un task in background per l'upscaling e il dettaglio. |
| `/async/final_swap` | Avvia un task in background per eseguire il face swap. |
| `/task_status/<task_id>` | Controlla lo stato di un task in background. |

-----

## Esempi di Immagini

## 🔧 Contribuire

1.  Forka → crea un branch per la feature → crea una Pull Request
2.  Segui le convenzioni di PEP‑8 & Prettier
3.  Includi screenshot prima/dopo per le modifiche all'interfaccia

### Roadmap

  * [x] Coda di task asincrona (Celery) per i lavori AI lunghi
  * [ ] Refactoring del frontend con Vue/Svelte per una gestione dello stato più complessa
  * [ ] UI per lo scambio a caldo dei modelli (scelta del checkpoint SDXL, upscaler, ecc.)
  * [ ] Scene con soggetti multipli
  * [ ] Trasformazioni avanzate per gli sticker (prospettiva / warp)

-----

## 📜 Licenza

MIT — sentiti libero di fare ciò che vuoi, un ringraziamento è apprezzato.