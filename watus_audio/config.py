import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych z pliku .env
# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# === Ustawienia Ogólne (General) ===
# Zapobieganie błędom bibliotek Intel MKL/OpenMP
KMP_DUPLICATE_LIB_OK = os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")
OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "1")
MKL_NUM_THREADS = os.environ.get("MKL_NUM_THREADS", "1")
CT2_SKIP_CONVERTERS = os.environ.get("CT2_SKIP_CONVERTERS", "1")

# === Komunikacja ZMQ (ZMQ Communication) ===
# Adresy gniazd ZMQ do komunikacji między procesami
PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")

# === Synteza Mowy (TTS) ===
# Wybór dostawcy TTS: 'piper' lub 'gemini'
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "gemini").lower()

# Konfiguracja Piper TTS
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "models/piper/voices/pl_PL-jarvis_wg_glos-medium.onnx")
PIPER_SR = int(os.environ.get("PIPER_SAMPLE_RATE", "22050"))
PIPER_BIN = os.environ.get("PIPER_BIN")
PIPER_CONFIG = os.environ.get("PIPER_CONFIG")

# Konfiguracja Gemini TTS
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Callirrhoe")

# Konfiguracja XTTS-v2
XTTS_MODEL_PATH = os.environ.get("XTTS_MODEL_PATH") # Opcjonalnie, jeśli manualnie
XTTS_SPEAKER_WAV = os.environ.get("XTTS_SPEAKER_WAV", "models/xtts/ref.wav")
XTTS_LANGUAGE = os.environ.get("XTTS_LANGUAGE", "pl")

# === Rozpoznawanie Mowy (STT) ===
# Wybór dostawcy STT: 'local' (Whisper) lub 'groq' (obecnie nieużywane)
STT_PROVIDER = os.environ.get("STT_PROVIDER", "local").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "whisper-large-v3")

def _normalize_fw_model(name: str) -> str:
    """
    Normalizuje nazwę modelu Faster Whisper.
    Jeśli podano krótką nazwę (np. 'small'), zamienia ją na pełną ścieżkę repozytorium.
    """
    name = (name or "").strip()
    short = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"}
    if "/" not in name and name.lower() in short:
        return f"guillaumekln/faster-whisper-{name.lower()}"
    return name

# === Konfiguracja Whisper (Uproszczona) ===
# Odczytujemy proste zmienne konfiguracyjne
_whisper_size = os.environ.get("WHISPER_SIZE", "medium").lower()
_whisper_device_type = os.environ.get("WHISPER_DEVICE_TYPE", "cpu").lower()

# Automatyczne ustalenie ścieżki do modelu
# Najpierw sprawdzamy lokalny katalog models/faster-whisper-{size}
_local_model_path = f"models/faster-whisper-{_whisper_size}"
if os.path.exists(_local_model_path):
    WHISPER_MODEL_NAME = _local_model_path
else:
    # Fallback do nazwy repozytorium (pobierze do cache)
    WHISPER_MODEL_NAME = f"Systran/faster-whisper-{_whisper_size}"

# Automatyczne ustalenie parametrów compute/device
if _whisper_device_type == "gpu":
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE = "float16"
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE = "int8"

# Nadpisanie (opcjonalne) przez stare zmienne, jeśli ktoś ich użył ręcznie w .env
if os.environ.get("WHISPER_MODEL"):
    WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL")
if os.environ.get("WHISPER_DEVICE"):
    WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE")
if os.environ.get("WHISPER_COMPUTE"):
    WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE")


WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
CPU_THREADS = int(os.environ.get("WATUS_CPU_THREADS", str(os.cpu_count() or 4)))

# === Audio i VAD (Voice Activity Detection) ===
SAMPLE_RATE = int(os.environ.get("WATUS_SR", "16000"))
BLOCK_SIZE = int(os.environ.get("WATUS_BLOCKSIZE", str(int(round(SAMPLE_RATE * 0.02)))))
VAD_MODE = int(os.environ.get("WATUS_VAD_MODE", "1"))
VAD_MIN_MS = int(os.environ.get("WATUS_VAD_MIN_MS", "150"))
SIL_MS_END = int(os.environ.get("WATUS_SIL_MS_END", "450"))
ASR_MIN_DBFS = float(os.environ.get("ASR_MIN_DBFS", "-34"))

# Parametry detekcji mowy
PREBUFFER_FRAMES = int(os.environ.get("WATUS_PREBUFFER_FRAMES", "15"))
START_MIN_FRAMES = int(os.environ.get("WATUS_START_MIN_FRAMES", "4"))
START_MIN_DBFS = float(os.environ.get("WATUS_START_MIN_DBFS", str(ASR_MIN_DBFS + 4.0)))
MIN_MS_BEFORE_ENDPOINT = int(os.environ.get("WATUS_MIN_MS_BEFORE_ENDPOINT", "500"))
END_AT_DBFS_DROP = float(os.environ.get("END_AT_DBFS_DROP", "0"))
EMIT_COOLDOWN_MS = int(os.environ.get("EMIT_COOLDOWN_MS", "300"))
MAX_UTT_MS = int(os.environ.get("MAX_UTT_MS", "6500"))
GAP_TOL_MS = int(os.environ.get("WATUS_GAP_TOL_MS", "450"))

# Urządzenia audio
IN_DEV_ENV = os.environ.get("WATUS_INPUT_DEVICE")
OUT_DEV_ENV = os.environ.get("WATUS_OUTPUT_DEVICE")

DIALOG_PATH = os.environ.get("DIALOG_PATH", "data/watus_audio/dialog.jsonl")

# === Weryfikacja Mówcy (Speaker Verification) ===
SPEAKER_VERIFY = int(os.environ.get("SPEAKER_VERIFY", "1"))
WAKE_WORDS = [w.strip() for w in
              os.environ.get("WAKE_WORDS", "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu").split(",") if
              w.strip()]
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.64"))
SPEAKER_STICKY_THRESHOLD = float(os.environ.get("SPEAKER_STICKY_THRESHOLD", str(SPEAKER_THRESHOLD)))
SPEAKER_GRACE = float(os.environ.get("SPEAKER_GRACE", "0.12"))
SPEAKER_STICKY_SEC = float(os.environ.get("SPEAKER_STICKY_SEC", os.environ.get("SPEAKER_STICKY_S", "3600")))
SPEAKER_MIN_ENROLL_SCORE = float(os.environ.get("SPEAKER_MIN_ENROLL_SCORE", "0.55"))
SPEAKER_MIN_DBFS = float(os.environ.get("SPEAKER_MIN_DBFS", "-40"))
SPEAKER_MAX_DBFS = float(os.environ.get("SPEAKER_MAX_DBFS", "-5"))
SPEAKER_BACK_THRESHOLD = float(os.environ.get("SPEAKER_BACK_THRESHOLD", "0.56"))
SPEAKER_REQUIRE_MATCH = int(os.environ.get("SPEAKER_REQUIRE_MATCH", "1"))

# === Zachowanie (Behavior) ===
WAIT_REPLY_S = float(os.environ.get("WAIT_REPLY_S", "0.6"))

# === Reporter ===
LLM_HTTP_URL = (os.environ.get("LLM_HTTP_URL") or "").strip()
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", os.environ.get("LLM_HTTP_TIMEOUT", "30")))
SCENARIOS_DIR = os.environ.get("WATUS_SCENARIOS_DIR", "./scenarios_text")
SCENARIO_ACTIVE_PATH = os.environ.get("SCENARIO_ACTIVE_PATH", os.path.join(SCENARIOS_DIR, "active.jsonl"))
CAMERA_NAME  = os.environ.get("CAMERA_NAME", "cam_front")
CAMERA_JSONL = os.environ.get("CAMERA_JSONL", "data/watus_audio/camera.jsonl") # Domyślnie plik lokalny
LOG_DIR   = os.environ.get("LOG_DIR", "./")
RESP_FILE = os.path.join(LOG_DIR, "data/watus_audio/responses.jsonl")
MELD_FILE = os.path.join(LOG_DIR, "data/watus_audio/meldunki.jsonl")
CAM_WINDOW_SEC = float(os.environ.get("CAMERA_WINDOW_SEC", "2.5"))
