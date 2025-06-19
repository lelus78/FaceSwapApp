# Architettura degli Agenti

Questo documento descrive gli "agenti" o i componenti intelligenti che alimentano le funzionalità principali dell'applicazione. Ogni agente è responsabile di un compito specifico e opera all'interno del worker Celery.

---

## Agent: Image Processor — IP-Adapter + Stable Diffusion

Questo è l'agente principale, responsabile della generazione e della manipolazione delle immagini.

-   **Model/API**: `diffusers` (Hugging Face), `IP-Adapter`, `Stable Diffusion 1.5`
-   **Trigger**: Chiamata API da `/process_image`
-   **Stage**: Eseguito nel worker Celery (`process_image_task`)

### Descrizione del Flusso

1.  **Ricezione del Task**: Il worker Celery riceve un task con i dati necessari: prompt, immagine di input (volto), immagine di destinazione (opzionale), e parametri (scale, steps, etc.).
2.  **Caricamento dei Modelli**: Carica i modelli richiesti (Stable Diffusion, IP-Adapter, InsightFace per il face detection) se non sono già in memoria.
3.  **Preparazione delle Immagini**:
    -   Estrae il volto dall'immagine di input usando `insightface`.
    -   Pre-processa l'immagine di destinazione (se presente) o crea un'immagine di base per il text-to-image.
4.  **Generazione**: Esegue il loop di denoising di Stable Diffusion, utilizzando l'IP-Adapter per guidare la generazione basandosi sulle feature del volto di input.
5.  **Salvataggio**: Salva l'immagine risultante nella directory `outputs/` e aggiorna il database.
6.  **Restituzione del Risultato**: Memorizza l'URL dell'immagine nel backend dei risultati di Celery (Redis) in modo che il frontend possa recuperarla.

---

## Agent: Prompt Enhancer — Ollama

Questo agente migliora i prompt forniti dall'utente per ottenere risultati migliori dal modello di generazione di immagini. Sfrutta un modello linguistico multimodale per analizzare sia il testo che un'immagine di riferimento.

-   **Model/API**: **Ollama (con modello multimodale, es. `llava`)**
-   **Trigger**: Chiamate API da `/enhance_prompt` e `/enhance_part_prompt`
-   **Stage**: Eseguito direttamente nel server Flask (richiesta sincrona veloce)

### Descrizione del Flusso

1.  **Richiesta API**: L'utente clicca sul pulsante "Migliora" nell'interfaccia. Il frontend invia il testo del prompt corrente e l'immagine di riferimento (in formato base64) all'endpoint Flask.
2.  **Costruzione del Prompt per l'LLM**: Il backend Flask costruisce un meta-prompt, istruendo il modello Ollama su come agire. Ad esempio: *"Sei un esperto prompt engineer. Migliora il seguente prompt basandoti sull'immagine fornita. Restituisci solo il prompt ottimizzato..."*.
3.  **Chiamata a Ollama**: Effettua una richiesta HTTP all'istanza locale di Ollama, inviando il meta-prompt e l'immagine.
4.  **Parsing della Risposta**: Estrae il testo generato dalla risposta JSON di Ollama.
5.  **Invio al Frontend**: Restituisce il prompt migliorato al frontend, che lo aggiorna nell'area di testo.

Questo agente permette di creare descrizioni più ricche e dettagliate, migliorando significativamente la qualità e la coerenza delle immagini generate.

---

## Agent: Meme Content Generator — Ollama

Questo agente viene utilizzato all'interno del "Meme Studio" per generare automaticamente contenuti testuali (didascalie e tag) basati su un'immagine.

-   **Model/API**: **Ollama (con modello multimodale, es. `llava`)**
-   **Trigger**: Chiamate API da `/meme/generate_caption` e `/meme/generate_tags`
-   **Stage**: Eseguito direttamente nel server Flask

### Descrizione del Flusso

1.  **Richiesta API**: Dal Meme Studio, l'utente richiede una didascalia o dei tag per l'immagine corrente. L'immagine e il "tono" desiderato (es. scherzoso, sarcastico) vengono inviati al backend.
2.  **Prompting Specifico**: Il backend crea un prompt specifico per il compito.
    -   *Per le didascalie*: *"Genera una singola, breve e brillante didascalia per l'immagine fornita con un tono [scherzoso]..."*
    -   *Per i tag*: *"Genera da 3 a 5 brevi tag in italiano per l'immagine fornita. Rispondi con una lista separata da virgole..."*
3.  **Chiamata a Ollama**: Invia la richiesta all'istanza locale di Ollama.
4.  **Restituzione dei Contenuti**: Il testo generato (didascalia o tag) viene restituito al frontend e visualizzato nell'editor del Meme Studio.