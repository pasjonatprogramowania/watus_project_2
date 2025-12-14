#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision Trigger - System wyzwalacza głosowego do analizy obrazu z kamery IP.

System nasłuchuje mikrofonu, wykrywa frazę "co widzisz" (i warianty),
wykonuje zdjęcie z kamery IP, analizuje obraz przez VisionAgent
i odtwarza odpowiedź przez TTS.
"""

import os
import sys
import time
import re
import tempfile
import urllib.request
from typing import List, Dict, Any, Optional
from pathlib import Path

# Dodanie ścieżki do modułów watus_audio
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Import modułów watus_audio
from watus_audio import config
from watus_audio.common import log_message
from watus_audio.tts import synthesize_speech_and_play
from watus_audio.stt import SpeechToTextProcessingEngine
from watus_audio.state import SystemState
from watus_audio.bus import ZMQMessageBus

# Sprawdzenie dostępności OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    log_message("[VisionTrigger] OpenCV niedostępne - zainstaluj: pip install opencv-python")

# Próba importu VisionAgent z watus-ai
VISION_AGENT_PATH = os.environ.get("VISION_AGENT_PATH", r"C:\Users\pawel\Desktop\watus-ai")
sys.path.insert(0, VISION_AGENT_PATH)

try:
    from src.vision_agent import VisionAgent, simulate_detection_data
    VISION_AGENT_AVAILABLE = True
except ImportError as e:
    VISION_AGENT_AVAILABLE = False
    log_message(f"[VisionTrigger] VisionAgent niedostępny: {e}")


# === KONFIGURACJA ===
IP_CAMERA_URL = os.environ.get("IP_CAMERA_URL", "http://192.168.1.10")
IP_CAMERA_SNAPSHOT_PATH = os.environ.get("IP_CAMERA_SNAPSHOT_PATH", "/snapshot.jpg")

# Frazy wyzwalające
VISION_TRIGGER_PHRASES = [
    phrase.strip().lower() for phrase in 
    os.environ.get("VISION_TRIGGER_PHRASES", "co widzisz,co to jest,co masz przed sobą,co jest przed tobą,opisz co widzisz").split(",")
    if phrase.strip()
]


def check_vision_trigger(text: str) -> bool:
    """
    Sprawdza czy tekst zawiera frazę wyzwalającą analizę wizualną.
    
    Args:
        text: Tekst do sprawdzenia (transkrypcja mowy).
        
    Returns:
        True jeśli znaleziono frazę wyzwalającą.
    """
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    
    for phrase in VISION_TRIGGER_PHRASES:
        normalized_phrase = re.sub(r'[^\w\s]', '', phrase)
        if normalized_phrase in normalized_text:
            return True
    return False


def capture_image_from_ip_camera(
    camera_url: str = IP_CAMERA_URL,
    snapshot_path: str = IP_CAMERA_SNAPSHOT_PATH,
    timeout: float = 5.0
) -> Optional[bytes]:
    """
    Wykonuje zdjęcie z kamery IP poprzez HTTP.
    
    Args:
        camera_url: Bazowy URL kamery (np. http://192.168.1.10).
        snapshot_path: Ścieżka do endpointu snapshot (np. /snapshot.jpg).
        timeout: Timeout połączenia w sekundach.
        
    Returns:
        Bajty obrazu lub None w przypadku błędu.
    """
    full_url = f"{camera_url.rstrip('/')}{snapshot_path}"
    
    log_message(f"[VisionTrigger] Pobieranie zdjęcia z: {full_url}")
    
    try:
        request = urllib.request.Request(full_url)
        request.add_header('User-Agent', 'WatusVisionTrigger/1.0')
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            image_bytes = response.read()
            log_message(f"[VisionTrigger] Pobrano obraz: {len(image_bytes)} bajtów")
            return image_bytes
            
    except urllib.error.URLError as e:
        log_message(f"[VisionTrigger] Błąd połączenia z kamerą: {e}")
        return None
    except Exception as e:
        log_message(f"[VisionTrigger] Błąd pobierania obrazu: {e}")
        return None


def capture_image_from_ip_camera_opencv(
    camera_url: str = IP_CAMERA_URL,
    stream_path: str = "/video"
) -> Optional[bytes]:
    """
    Wykonuje zdjęcie z kamery IP poprzez strumień RTSP/MJPEG (OpenCV).
    
    Args:
        camera_url: Bazowy URL kamery.
        stream_path: Ścieżka do strumienia video.
        
    Returns:
        Bajty obrazu JPEG lub None w przypadku błędu.
    """
    if not CV2_AVAILABLE:
        log_message("[VisionTrigger] OpenCV niedostępne")
        return None
    
    stream_url = f"{camera_url.rstrip('/')}{stream_path}"
    log_message(f"[VisionTrigger] Łączenie ze strumieniem: {stream_url}")
    
    try:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            log_message("[VisionTrigger] Nie można otworzyć strumienia kamery")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            log_message("[VisionTrigger] Nie udało się pobrać klatki")
            return None
        
        # Konwersja do JPEG
        success, jpeg_bytes = cv2.imencode('.jpg', frame)
        if success:
            log_message(f"[VisionTrigger] Przechwycono klatkę: {len(jpeg_bytes)} bajtów")
            return jpeg_bytes.tobytes()
        
        return None
        
    except Exception as e:
        log_message(f"[VisionTrigger] Błąd OpenCV: {e}")
        return None


def get_mock_detection_data() -> List[Dict[str, Any]]:
    """
    Zwraca symulowane dane detekcji obiektów (mockup JSONL).
    
    W rzeczywistej implementacji dane pochodziłyby z modelu YOLO/RT-DETR
    uruchomionego na pobranym obrazie.
    
    Returns:
        Lista wykrytych obiektów z ich właściwościami.
    """
    if VISION_AGENT_AVAILABLE:
        return simulate_detection_data()
    
    # Fallback mockup
    return [
        {
            "name": "person",
            "conf": 0.89,
            "bbox": [100, 50, 300, 400],
            "description": "osoba"
        },
        {
            "name": "desk",
            "conf": 0.82,
            "bbox": [50, 300, 500, 480],
            "description": "biurko"
        },
        {
            "name": "monitor",
            "conf": 0.91,
            "bbox": [150, 100, 350, 280],
            "description": "monitor komputerowy"
        }
    ]


class VisionTrigger:
    """
    System wyzwalacza głosowego do analizy obrazu.
    
    Nasłuchuje mikrofonu, wykrywa frazy wyzwalające, wykonuje zdjęcie
    z kamery IP i generuje odpowiedź głosową.
    """
    
    def __init__(
        self,
        camera_url: str = IP_CAMERA_URL,
        output_device_index: Optional[int] = None
    ):
        """
        Inicjalizuje system wyzwalacza.
        
        Args:
            camera_url: URL kamery IP.
            output_device_index: Indeks urządzenia wyjściowego audio (TTS).
        """
        self.camera_url = camera_url
        self.output_device_index = output_device_index
        
        # Inicjalizacja VisionAgent
        if VISION_AGENT_AVAILABLE:
            self.vision_agent = VisionAgent()
            log_message("[VisionTrigger] VisionAgent zainicjalizowany")
        else:
            self.vision_agent = None
            log_message("[VisionTrigger] OSTRZEŻENIE: VisionAgent niedostępny")
        
        # Rozwiązanie indeksu urządzenia wyjściowego
        if self.output_device_index is None:
            self.output_device_index = self._resolve_output_device()
    
    def _resolve_output_device(self) -> Optional[int]:
        """Rozwiązuje indeks urządzenia wyjściowego z konfiguracji."""
        try:
            import sounddevice as sd
            out_env = config.OUT_DEV_ENV
            
            if out_env is None:
                return None
            
            if out_env.isdigit():
                return int(out_env)
            
            # Szukanie po nazwie
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if out_env.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    return i
            
            return None
        except Exception:
            return None
    
    def process_vision_request(self, question: str = "Co widzisz?") -> str:
        """
        Przetwarza żądanie wizualne: zdjęcie + analiza + odpowiedź.
        
        Args:
            question: Pytanie do zadania agentowi.
            
        Returns:
            Odpowiedź tekstowa od agenta.
        """
        log_message(f"[VisionTrigger] Przetwarzanie żądania: '{question}'")
        
        # 1. Wykonanie zdjęcia z kamery IP
        image_bytes = capture_image_from_ip_camera(self.camera_url)
        
        if image_bytes is None:
            # Próba przez OpenCV
            image_bytes = capture_image_from_ip_camera_opencv(self.camera_url)
        
        if image_bytes is None:
            log_message("[VisionTrigger] Nie udało się pobrać obrazu z kamery")
            return "Przepraszam, nie mogę teraz połączyć się z kamerą."
        
        # 2. Symulacja danych detekcji
        detection_data = get_mock_detection_data()
        log_message(f"[VisionTrigger] Dane detekcji: {len(detection_data)} obiektów")
        
        # 3. Analiza przez VisionAgent
        if self.vision_agent is None:
            log_message("[VisionTrigger] VisionAgent niedostępny, używam fallback")
            objects = [obj["name"] for obj in detection_data]
            return f"Widzę: {', '.join(objects)}."
        
        try:
            response = self.vision_agent.analyze_image_bytes(
                image_bytes=image_bytes,
                mime_type="image/jpeg",
                detection_data=detection_data,
                question=question
            )
            log_message(f"[VisionTrigger] Odpowiedź agenta: {response[:100]}...")
            return response
            
        except Exception as e:
            log_message(f"[VisionTrigger] Błąd analizy: {e}")
            return "Przepraszam, wystąpił błąd podczas analizy obrazu."
    
    def speak_response(self, text: str):
        """
        Odtwarza odpowiedź przez TTS.
        
        Args:
            text: Tekst do wypowiedzenia.
        """
        log_message(f"[VisionTrigger] TTS: '{text[:50]}...'")
        try:
            synthesize_speech_and_play(text, self.output_device_index)
        except Exception as e:
            log_message(f"[VisionTrigger] Błąd TTS: {e}")
    
    def handle_utterance(self, text: str) -> bool:
        """
        Obsługuje wypowiedź użytkownika - sprawdza trigger i reaguje.
        
        Args:
            text: Transkrypcja wypowiedzi użytkownika.
            
        Returns:
            True jeśli wypowiedź została obsłużona jako vision trigger.
        """
        if not check_vision_trigger(text):
            return False
        
        log_message(f"[VisionTrigger] Wykryto trigger: '{text}'")
        
        # Przetwarzanie żądania wizualnego
        response = self.process_vision_request(text)
        
        # Odtworzenie odpowiedzi
        self.speak_response(response)
        
        return True


def create_vision_trigger_callbacks(vision_trigger: VisionTrigger, original_callbacks: dict) -> dict:
    """
    Tworzy wrapper dla callbacków STT, dodając obsługę vision trigger.
    
    Args:
        vision_trigger: Instancja VisionTrigger.
        original_callbacks: Oryginalne callbacki systemu.
        
    Returns:
        Zmodyfikowane callbacki.
    """
    return original_callbacks  # Callbacki pozostają bez zmian


def run_vision_trigger_standalone():
    """
    Uruchamia VisionTrigger w trybie standalone (do testów).
    """
    log_message("[VisionTrigger] === Uruchamianie w trybie standalone ===")
    
    trigger = VisionTrigger()
    
    # Test mockup
    log_message("[VisionTrigger] Test danych detekcji:")
    detection_data = get_mock_detection_data()
    for obj in detection_data:
        log_message(f"  - {obj['name']} ({obj['conf']*100:.0f}%)")
    
    # Test bez kamery (mockup odpowiedź)
    log_message("\n[VisionTrigger] Test process_vision_request (bez kamery):")
    
    # Symulacja odpowiedzi gdy kamera niedostępna
    if trigger.vision_agent:
        response = trigger.vision_agent.analyze_image_mock("Co widzisz?")
        log_message(f"Odpowiedź: {response}")
        
        # Odtworzenie odpowiedzi
        log_message("\n[VisionTrigger] Odtwarzanie odpowiedzi przez TTS...")
        trigger.speak_response(response)
    else:
        log_message("VisionAgent niedostępny - test pominięty")
    
    log_message("\n[VisionTrigger] === Test zakończony ===")


# === URUCHOMIENIE ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Trigger - głosowy wyzwalacz analizy obrazu")
    parser.add_argument("--test", action="store_true", help="Uruchom w trybie testowym")
    parser.add_argument("--camera", default=IP_CAMERA_URL, help="URL kamery IP")
    args = parser.parse_args()
    
    if args.test:
        run_vision_trigger_standalone()
    else:
        print("Vision Trigger - System nasłuchiwania głosowego")
        print("=" * 50)
        print(f"Kamera IP: {args.camera}")
        print(f"Frazy wyzwalające: {VISION_TRIGGER_PHRASES}")
        print()
        print("Aby uruchomić test standalone, użyj: python vision_trigger.py --test")
        print("Aby zintegrować z głównym systemem Watus, zaimportuj moduł.")
