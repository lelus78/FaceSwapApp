# ===================================================================================
# === SCRIPT PER CONVERTIRE UN CHECKPOINT .safetensors IN FORMATO DIFFUSERS ===
# ===================================================================================

import torch
from diffusers import StableDiffusionXLPipeline
import os
import logging

# --- IMPOSTAZIONI GIÀ CONFIGURATE PER TE ---

# 1. Percorso del file .safetensors che hai scaricato.
#    Assicurati che sia in una cartella 'downloads'.
single_file_path = os.path.join(
    "downloads", "sdxlYamersRealistic5_v5Rundiffusion.safetensors")

# 2. Nome della cartella di output del modello convertito.
output_model_name = "sdxl-yamers-realistic5-v5Rundiffusion"

# --- FINE IMPOSTAZIONI ---

# Costruisce il percorso completo per la cartella di output
output_directory_path = os.path.join("models", "checkpoints",
                                     output_model_name)

logging.basicConfig(level=logging.INFO)

# Controlla se il file di input esiste
if not os.path.exists(single_file_path):
    logging.error("[ERRORE] File non trovato: %s", single_file_path)
    logging.error(
        "Per favore, scarica il modello da Civitai e mettilo nella cartella 'downloads'."
    )
else:
    logging.info("[*] Caricamento del modello dal file singolo: %s...",
                 single_file_path)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        single_file_path, torch_dtype=torch.float16, use_safetensors=True)

    logging.info(
        "[*] Salvataggio del modello nel formato 'diffusers' in: %s...",
        output_directory_path,
    )

    pipeline.save_pretrained(output_directory_path)

    logging.info("\n[SUCCESSO!] Conversione completata.")
    logging.info(
        "Ora puoi usare il seguente nome nel tuo server.py: '%s'",
        output_model_name,
    )
