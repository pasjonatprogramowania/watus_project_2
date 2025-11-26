import json
from . import config

def log_message(message: str):
    """
    Wypisuje wiadomość do standardowego wyjścia (konsoli) z wymuszonym flushowaniem bufora.
    
    Argumenty:
        message (str): Treść wiadomości do wypisania.
    """
    print(message, flush=True)

def append_line_to_dialog_history(dialog_object: dict, file_path=config.DIALOG_PATH):
    """
    Dopisuje obiekt dialogu (jako linię JSON) do pliku historii dialogów.
    
    Argumenty:
        dialog_object (dict): Obiekt reprezentujący wpis w dialogu (np. wypowiedź użytkownika).
        file_path (str): Ścieżka do pliku, domyślnie pobierana z konfiguracji.
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dialog_object, ensure_ascii=False) + "\n")
    except Exception as e:
        log_message(f"[Watus] Failed to write dialog line: {e}")

def write_object_to_jsonl_file(file_path: str, data_object: dict):
    """
    Zapisuje dowolny obiekt jako linię JSON do wskazanego pliku (tryb append).
    
    Argumenty:
        file_path (str): Ścieżka do pliku docelowego.
        data_object (dict): Obiekt danych do zapisania.
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_object, ensure_ascii=False) + "\n")
    except Exception:
        pass
