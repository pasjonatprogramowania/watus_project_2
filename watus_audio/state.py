import time
import threading
from . import config

class SystemState:
    """
    Klasa zarządzająca stanem systemu Watus (np. czy aktualnie mówi, czy czeka na odpowiedź).
    Służy do synchronizacji wątków i blokowania nasłuchiwania w odpowiednich momentach.
    """
    
    def __init__(self):
        self.session_id = f"live_{int(time.time())}"
        self._tts_active_flag = False
        self._awaiting_reply_flag = False
        self._lock = threading.Lock()
        self.tts_pending_until_timestamp = 0.0
        self.waiting_reply_until_timestamp = 0.0
        self.last_tts_reply_id = None

    def set_tts_active_flag(self, is_active: bool):
        """
        Ustawia flagę aktywności TTS (czy system aktualnie mówi).
        
        Argumenty:
            is_active (bool): True jeśli system mówi, False w przeciwnym razie.
        """
        with self._lock:
            self._tts_active_flag = is_active

    def set_awaiting_reply_flag(self, is_awaiting: bool):
        """
        Ustawia flagę oczekiwania na odpowiedź od LLM.
        
        Argumenty:
            is_awaiting (bool): True jeśli oczekujemy na odpowiedź.
        """
        with self._lock:
            self._awaiting_reply_flag = is_awaiting

    def block_input_until_reply_received(self):
        """
        Blokuje wejście audio na określony czas (zdefiniowany w configu), dając czas na nadejście odpowiedzi.
        """
        with self._lock:
            self.waiting_reply_until_timestamp = time.time() + config.WAIT_REPLY_S

    def is_input_blocked(self) -> bool:
        """
        Sprawdza, czy wejście audio powinno być zablokowane (np. bo system mówi lub czeka na odpowiedź).
        
        Zwraca:
            bool: True jeśli wejście jest zablokowane.
        """
        with self._lock:
            return (
                    self._tts_active_flag
                    or self._awaiting_reply_flag
                    or (time.time() < self.tts_pending_until_timestamp)
                    or (time.time() < self.waiting_reply_until_timestamp)
            )
