import os
import sys

def download_xtts():
    print("Inicjalizacja pobierania modelu XTTS-v2...")
    try:
        from TTS.api import TTS
    except ImportError:
        print("Błąd: Biblioteka TTS nie jest zainstalowana. Uruchom: pip install TTS")
        return

    # To wywołanie automatycznie pobierze model do domyślnego cache TTS
    # (zazwyczaj C:\Users\user\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2)
    print("Pobieranie modelu 'tts_models/multilingual/multi-dataset/xtts_v2'...")
    try:
        TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("Pobieranie zakończone pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas pobierania: {e}")

if __name__ == "__main__":
    download_xtts()
