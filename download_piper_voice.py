import os
import requests
import argparse

# Base URL for Piper voices (Hugging Face)
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# Default voice
DEFAULT_VOICE = "pl/pl_PL/gosia/medium"

def download_file(url, dest_path):
    print(f"Pobieranie: {url} -> {dest_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Pobrano pomyślnie.")
        return True
    except Exception as e:
        print(f"Błąd pobierania: {e}")
        return False

def download_piper_voice(voice_path, output_dir="models/piper"):
    """
    Pobiera model ONNX i plik JSON konfiguracyjny dla danego głosu.
    voice_path: np. 'pl/pl_PL/gosia/medium'
    """
    # Struktura URL: BASE_URL/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx
    parts = voice_path.split('/')
    if len(parts) < 4:
        print("Nieprawidłowy format ścieżki głosu. Oczekiwano: lang/region/name/quality")
        return

    voice_name_full = f"{parts[1]}-{parts[2]}-{parts[3]}" # pl_PL-gosia-medium
    
    onnx_url = f"{BASE_URL}/{voice_path}/{voice_name_full}.onnx"
    json_url = f"{BASE_URL}/{voice_path}/{voice_name_full}.onnx.json"

    # Przygotuj katalog docelowy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    onnx_path = os.path.join(output_dir, f"{voice_name_full}.onnx")
    json_path = os.path.join(output_dir, f"{voice_name_full}.onnx.json")

    success = True
    if not os.path.exists(onnx_path):
        if not download_file(onnx_url, onnx_path): success = False
    else:
        print(f"Plik ONNX już istnieje: {onnx_path}")

    if not os.path.exists(json_path):
        if not download_file(json_url, json_path): success = False
    else:
        print(f"Plik JSON już istnieje: {json_path}")
    
    return success, onnx_path, json_path

def main():
    parser = argparse.ArgumentParser(description="Pobieranie głosów Piper (ONNX).")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, 
                        help=f"Ścieżka głosu w repozytorium (domyślnie: {DEFAULT_VOICE})")
    parser.add_argument("--output_dir", type=str, default="models/piper", 
                        help="Katalog docelowy.")
    
    args = parser.parse_args()

    success, onnx, json_conf = download_piper_voice(args.voice, args.output_dir)
    if success:
        print("\nGotowe! Skonfiguruj .env:")
        print(f"PIPER_MODEL_PATH={os.path.abspath(onnx)}")
        print(f"PIPER_CONFIG={os.path.abspath(json_conf)}")

if __name__ == "__main__":
    main()
