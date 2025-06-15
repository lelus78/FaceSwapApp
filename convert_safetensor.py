# ===================================================================================
# === SCRIPT PER CONVERTIRE UN CHECKPOINT .safetensors IN FORMATO DIFFUSERS ===
# ===================================================================================

import torch
from diffusers import StableDiffusionXLPipeline
import os

# --- IMPOSTAZIONI GIÀ CONFIGURATE PER TE ---

# 1. Percorso del file .safetensors che hai scaricato.
#    Assicurati che sia in una cartella 'downloads'.
single_file_path = os.path.join("downloads", "sdxlYamersRealistic5_v5Rundiffusion.safetensors")

# 2. Nome della cartella di output del modello convertito.
output_model_name = "sdxl-yamers-realistic5-v5Rundiffusion"

# --- FINE IMPOSTAZIONI ---


# Costruisce il percorso completo per la cartella di output
output_directory_path = os.path.join("models", "checkpoints", output_model_name)

# Controlla se il file di input esiste
if not os.path.exists(single_file_path):
    print(f"[ERRORE] File non trovato: {single_file_path}")
    print("Per favore, scarica il modello da Civitai e mettilo nella cartella 'downloads'.")
else:
    print(f"[*] Caricamento del modello dal file singolo: {single_file_path}...")

    pipeline = StableDiffusionXLPipeline.from_single_file(
        single_file_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    print(f"[*] Salvataggio del modello nel formato 'diffusers' in: {output_directory_path}...")

    pipeline.save_pretrained(output_directory_path)

    print("\n[SUCCESSO!] Conversione completata.")
    print(f"Ora puoi usare il seguente nome nel tuo server.py: '{output_model_name}'")
