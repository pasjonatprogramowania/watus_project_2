import zmq
import json
import time
import threading
import queue
import sys
import atexit
from . import config
from .common import log_message

class ZMQMessageBus:
    """
    Klasa obsługująca komunikację ZeroMQ (PUB/SUB) między procesem Watus a Reporterem.
    Zarządza wysyłaniem komunikatów o liderze i odbieraniem poleceń TTS.
    """
    
    def __init__(self, pub_addr: str, sub_addr: str):
        """
        Inicjalizuje szynę komunikatów ZMQ.
        
        Argumenty:
            pub_addr (str): Adres gniazda PUB (do wysyłania).
            sub_addr (str): Adres gniazda SUB (do odbierania).
        """
        self.zmq_context = zmq.Context.instance()

        # Utworzenie gniazd
        self.publisher_socket = self.zmq_context.socket(zmq.PUB)
        self.publisher_socket.setsockopt(zmq.SNDHWM, 100)
        self.publisher_socket.bind(pub_addr)

        self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.subscriber_socket.connect(sub_addr)
        self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "tts.speak")

        self._subscriber_queue = queue.Queue()

        # Śledzenie stanu działania
        self._is_running = True
        self._subscriber_thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self._subscriber_thread.start()

        # Rejestracja sprzątania przy wyjściu
        atexit.register(self._cleanup_resources)

    def _cleanup_resources(self):
        """
        Bezpiecznie zamyka gniazda ZMQ i zatrzymuje wątek nasłuchujący.
        """
        try:
            self._is_running = False

            # Zamknij gniazda
            if hasattr(self, 'publisher_socket') and self.publisher_socket:
                try:
                    self.publisher_socket.close()
                except:
                    pass

            if hasattr(self, 'subscriber_socket') and self.subscriber_socket:
                try:
                    self.subscriber_socket.close()
                except:
                    pass

            # Uwaga: Nie zamykamy kontekstu globalnego, bo może być używany gdzie indziej

        except Exception as e:
            print(f"[Watus][BUS] Cleanup error: {e}", file=sys.stderr)

    def publish_leader_utterance(self, message_payload: dict):
        """
        Publikuje wiadomość o wykrytej wypowiedzi lidera na kanale 'dialog.leader'.
        
        Argumenty:
            message_payload (dict): Dane wypowiedzi (tekst, metadane).
        """
        try:
            publish_start_time = time.time()
            self.publisher_socket.send_multipart([b"dialog.leader", json.dumps(message_payload, ensure_ascii=False).encode("utf-8")])
            log_message(f"[Perf] BUS_ms={int((time.time() - publish_start_time) * 1000)}")
        except Exception as e:
            log_message(f"[Watus][BUS] Publish leader error: {e}")

    def publish_system_state(self, state_name: str, data: dict = None):
        """
        Publikuje aktualny stan systemu (np. listening, processing) dla UI.
        
        Argumenty:
            state_name (str): Nazwa stanu.
            data (dict, optional): Dodatkowe dane stanu.
        """
        try:
            if data is None:
                data = {}
            message = {
                "state": state_name,
                "timestamp": time.time(),
                **data
            }
            self.publisher_socket.send_multipart([b"watus.state", json.dumps(message, ensure_ascii=False).encode("utf-8")])
        except Exception as e:
            log_message(f"[Watus] Failed to publish state message: {e}")

    def _subscriber_loop(self):
        """
        Wewnętrzna pętla wątku odbierającego wiadomości z ZMQ.
        """
        while self._is_running:
            try:
                topic, message_payload_bytes = self.subscriber_socket.recv_multipart()
                if topic != b"tts.speak": continue
                decoded_data = json.loads(message_payload_bytes.decode("utf-8", "ignore"))
                self._subscriber_queue.put(decoded_data)
            except Exception as e:
                if self._is_running:  # Loguj błędy tylko jeśli powinniśmy działać
                    time.sleep(0.01)

    def get_next_tts_message(self, timeout=0.1):
        """
        Pobiera następną wiadomość TTS z kolejki (jeśli dostępna).
        
        Argumenty:
            timeout (float): Czas oczekiwania na wiadomość w sekundach.
            
        Zwraca:
            dict lub None: Wiadomość TTS lub None jeśli kolejka pusta.
        """
        try:
            return self._subscriber_queue.get(timeout=timeout)
        except queue.Empty:
            return None
