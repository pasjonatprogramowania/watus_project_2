import sys
import threading
import atexit
from . import config
from .common import log_message
from .audio_utils import print_available_audio_devices, get_default_input_device_index, get_default_output_device_index
from .bus import ZMQMessageBus
from .state import SystemState
from .led import LEDStatusController
from .tts import synthesize_speech_and_play
from .stt import SpeechToTextProcessingEngine

# Globalny kontroler LED (potrzebny do callbacków)
led_controller = LEDStatusController()
atexit.register(led_controller.cleanup)
zmq_bus = None # Zostanie zainicjalizowany w main

def indicate_listen_state():
    """Sygnalizuje stan nasłuchiwania (log + LED + ZMQ)."""
    log_message("[Watus][STATE] LISTENING")
    led_controller.indicate_listening_state()
    if zmq_bus:
        try:
            zmq_bus.publish_system_state("listening")
        except:
            pass

def indicate_think_state():
    """Sygnalizuje stan myślenia/przetwarzania."""
    log_message("[Watus][STATE] THINKING")
    led_controller.indicate_processing_state()
    if zmq_bus:
        try:
            zmq_bus.publish_system_state("processing")
        except:
            pass

def indicate_speak_state():
    """Sygnalizuje stan mówienia."""
    log_message("[Watus][STATE] SPEAKING")
    led_controller.indicate_processing_state()
    if zmq_bus:
        try:
            zmq_bus.publish_system_state("speaking")
        except:
            pass

def indicate_idle_state():
    """Sygnalizuje stan bezczynności."""
    log_message("[Watus][STATE] IDLE")
    led_controller.indicate_processing_state()

def tts_worker_thread(system_state: SystemState, message_bus: ZMQMessageBus, output_device_index):
    """
    Wątek obsługujący kolejkę TTS. Odbiera wiadomości z busa i uruchamia syntezę mowy.
    """
    log_message("[Watus] Piper ready.")
    while True:
        tts_message_payload = message_bus.get_next_tts_message(timeout=0.1)
        if not tts_message_payload: continue
        text_to_speak = (tts_message_payload.get("text") or "").strip()
        reply_to_id = tts_message_payload.get("reply_to") or ""
        if system_state.last_tts_reply_id == reply_to_id and reply_to_id:
            log_message(f"[Watus][SUB] tts.speak DUP reply_to={reply_to_id} – pomijam")
            continue
        system_state.last_tts_reply_id = reply_to_id

        if text_to_speak:
            log_message(f"[Watus][LLM] answer len={len(text_to_speak)} (reply_to={reply_to_id})")

        # OD TERAZ prawdziwy TTS – blokujemy słuchanie
        system_state.set_awaiting_reply_flag(False)
        system_state.set_tts_active_flag(True)
        indicate_speak_state()
        try:
            synthesize_speech_and_play(text_to_speak, audio_output_device_index=output_device_index)
        finally:
            system_state.set_tts_active_flag(False)
            indicate_listen_state()

def main():
    """Główna funkcja uruchamiająca proces Watus."""
    global zmq_bus
    log_message(f"[Env] ASR=Faster WHISPER_MODEL={config.WHISPER_MODEL_NAME} WHISPER_DEVICE={config.WHISPER_DEVICE} "
        f"WHISPER_COMPUTE={config.WHISPER_COMPUTE} WATUS_BLOCKSIZE={config.BLOCK_SIZE}")
    log_message(f"[Watus] Wake words: {config.WAKE_WORDS}")
    log_message(f"[Watus] PUB dialog.leader @ {config.PUB_ADDR} | SUB tts.speak @ {config.SUB_ADDR}")
    
    print_available_audio_devices()
    
    input_device_index = get_default_input_device_index()
    output_device_index = get_default_output_device_index()
    log_message(f"[Watus] IO: input={input_device_index!r} | output={output_device_index!r}")

    zmq_bus = ZMQMessageBus(config.PUB_ADDR, config.SUB_ADDR)
    state = SystemState()
    
    threading.Thread(target=tts_worker_thread, args=(state, zmq_bus, output_device_index), daemon=True).start()

    callbacks = {
        'indicate_listen_state': indicate_listen_state,
        'indicate_think_state': indicate_think_state,
        'indicate_idle_state': indicate_idle_state
    }

    try:
        stt_engine = SpeechToTextProcessingEngine(state, zmq_bus, callbacks)
    except Exception as e:
        log_message(f"[Watus] STT init error: {e}")
        sys.exit(1)

    led_controller.indicate_listening_state()  # Start with listening state
    try:
        stt_engine.start_listening_loop(input_device_index)
    except KeyboardInterrupt:
        log_message("[Watus] stop")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_message(f"[Watus] fatal: {e}")
        sys.exit(1)
