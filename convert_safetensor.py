# Contenuto aggiornato per convert_safetensor.py

import torch
from diffusers import StableDiffusionXLPipeline
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_model(input_file, output_name):
    output_directory_path = os.path.join("models", "checkpoints", output_name)

    if not os.path.exists(input_file):
        logging.error("[ERRORE] File di input non trovato: %s", input_file)
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if os.path.exists(output_directory_path):
        logging.warning("[ATTENZIONE] La cartella di output '%s' esiste già. Potrebbe essere sovrascritta.", output_directory_path)
    
    os.makedirs(output_directory_path, exist_ok=True)

    logging.info("[*] Caricamento del modello da: %s", input_file)
    try:
        pipeline = StableDiffusionXLPipeline.from_single_file(
            input_file, torch_dtype=torch.float16, use_safetensors=True
        )
        logging.info("[*] Salvataggio del modello in: %s", output_directory_path)
        pipeline.save_pretrained(output_directory_path, safe_serialization=True)
        logging.info("\n[SUCCESSO!] Conversione completata per il modello '%s'.", output_name)
    except Exception as e:
        logging.error("Errore durante la conversione del modello: %s", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertitore da .safetensors a Diffusers.")
    parser.add_argument("--input", required=True, help="Percorso del file .safetensors di input.")
    parser.add_argument("--output", required=True, help="Nome della cartella di output del modello (non il percorso completo).")
    
    args = parser.parse_args()
    convert_model(args.input, args.output)