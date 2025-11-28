import os
import sys
import time
import struct
import subprocess
import tempfile
import sounddevice as sd
import soundfile as sf
from .common import log_message
from . import config

# Sprawdzenie dostępności Piper
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    log_message("[Watus][TTS] Piper Python API nie dostępne, używam binary metod")

# Sprawdzenie dostępności Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log_message("[Watus][TTS] Google Gemini TTS nie dostępne - pip install google-genai")

LOADED_PIPER_VOICE_MODEL = None

def _prepare_env_for_piper_binary(piper_bin_path: str) -> dict:
    """Przygotowuje zmienne środowiskowe dla binarnego Pipera (biblioteki)."""
    env_vars = os.environ.copy()
    bin_dir = os.path.dirname(piper_bin_path) if piper_bin_path else ""
    phonemize_lib_path = os.path.join(bin_dir, "piper-phonemize", "lib")
    extra_library_paths = []
    if os.path.isdir(bin_dir): extra_library_paths.append(bin_dir)
    if os.path.isdir(phonemize_lib_path): extra_library_paths.append(phonemize_lib_path)
    if not extra_library_paths: return env_vars

    if sys.platform == "darwin":
        path_key = "DYLD_LIBRARY_PATH"
    elif sys.platform.startswith("linux"):
        path_key = "LD_LIBRARY_PATH"
    else:
        path_key = "PATH"
    current_path_value = env_vars.get(path_key, "")
    separator = (":" if path_key != "PATH" else ";")
    env_vars[path_key] = (separator.join([*extra_library_paths, current_path_value]) if current_path_value else separator.join(extra_library_paths))
    return env_vars


def _initialize_piper_voice_model():
    """Ładuje model Piper do pamięci (jeśli używamy Python API)."""
    global LOADED_PIPER_VOICE_MODEL
    if PIPER_AVAILABLE:
        if not LOADED_PIPER_VOICE_MODEL:  # Initialize if not already loaded
            try:
                model_file_path = config.PIPER_MODEL_PATH
                if not os.path.isfile(model_file_path):
                    log_message(f"[Watus][TTS] Brak modelu Piper: {model_file_path}")
                    return False
                LOADED_PIPER_VOICE_MODEL = PiperVoice.load(model_file_path)
                log_message(f"[Watus][TTS] Piper voice załadowany z: {model_file_path}")
                return True
            except Exception as e:
                log_message(f"[Watus][TTS] Błąd ładowania Piper voice: {e}")
                return False
        else:
            # Already loaded
            return True
    return False


def synthesize_speech_piper(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Piper TTS i odtwarza ją.
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy.
        audio_output_device_index (int): Indeks urządzenia wyjściowego audio.
    """
    if not text_to_synthesize or not text_to_synthesize.strip(): return

    # Próba użycia nowego Python API
    if PIPER_AVAILABLE and _initialize_piper_voice_model():
        try:
            # Generate speech using new Python API (returns AudioChunk iterator)
            import numpy as np
            from piper import AudioChunk

            synthesized_audio_bytes = []
            for audio_stream_chunk in LOADED_PIPER_VOICE_MODEL.synthesize(text_to_synthesize):
                if isinstance(audio_stream_chunk, AudioChunk):
                    chunk_data_float32 = audio_stream_chunk.audio_int16_array.astype(np.float32) / 32768.0
                    synthesized_audio_bytes.append(chunk_data_float32)
                else:
                    log_message(f"[Watus][TTS] Nieoczekiwany typ chunk: {type(audio_stream_chunk)}")

            if synthesized_audio_bytes:
                # Combine all audio chunks
                concatenated_audio_samples = np.concatenate(synthesized_audio_bytes)
                sample_rate_hz = LOADED_PIPER_VOICE_MODEL.config.sample_rate
                sd.play(concatenated_audio_samples, sample_rate_hz, device=audio_output_device_index, blocking=True)
                return
            else:
                log_message(f"[Watus][TTS] Brak danych audio z nowego API")
        except Exception as e:
            log_message(f"[Watus][TTS] Błąd Python API, próbuję binary fallback: {e}")

    # Fallback to binary method (legacy)
    if not config.PIPER_BIN or not os.path.isfile(config.PIPER_BIN):
        log_message(f"[Watus][TTS] Uwaga: brak/niepoprawny PIPER_BIN: {config.PIPER_BIN}")
        return
    if not config.PIPER_MODEL_PATH or not os.path.isfile(config.PIPER_MODEL_PATH):
        log_message(f"[Watus][TTS] Brak/niepoprawny PIPER_MODEL_PATH: {config.PIPER_MODEL_PATH}")
        return

    try:
        if sys.platform == "darwin":
            bin_dir = os.path.dirname(config.PIPER_BIN)
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", bin_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

    piper_config_args = ["--config", config.PIPER_CONFIG] if config.PIPER_CONFIG and os.path.isfile(config.PIPER_CONFIG) else []
    process_env = _prepare_env_for_piper_binary(config.PIPER_BIN)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temporary_wav_file_path = tmp_file.name
    start_time = time.time()
    try:
        command_args = [config.PIPER_BIN, "--model", config.PIPER_MODEL_PATH, *piper_config_args, "--output_file", temporary_wav_file_path]
        if config.PIPER_SR: command_args += ["--sample_rate", str(config.PIPER_SR)]
        subprocess.run(command_args, input=(text_to_synthesize or "").encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=process_env)
        audio_data_array, sample_rate_hz = sf.read(temporary_wav_file_path, dtype="float32")
        sd.play(audio_data_array, sample_rate_hz, device=audio_output_device_index, blocking=True)
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        log_message(f"[Watus][TTS] Piper błąd (proc): {error_message}")
    except Exception as e:
        log_message(f"[Watus][TTS] Odtwarzanie nieudane: {e}")
    finally:
        try:
            os.unlink(temporary_wav_file_path)
        except Exception:
            pass
        log_message(f"[Perf] TTS_play_ms={int((time.time() - start_time) * 1000)}")


def _add_wav_header(raw_audio_bytes: bytes, mime_type_string: str) -> bytes:
    """Generuje nagłówek WAV dla surowych danych audio."""
    audio_params = _parse_audio_mime_type(mime_type_string)
    bits_per_sample = audio_params["bits_per_sample"]
    sample_rate_hz = audio_params["rate"]
    num_channels = 1
    data_size_bytes = len(raw_audio_bytes)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate_hz * block_align
    chunk_size = 36 + data_size_bytes

    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate_hz,   # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size_bytes   # Subchunk2Size (size of audio data)
    )
    return wav_header + raw_audio_bytes


def _parse_audio_mime_type(mime_type_string: str) -> dict:
    """Parsuje parametry audio (rate, bits) z typu MIME."""
    bits_per_sample = 16
    rate_hz = 24000

    parts = mime_type_string.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate_hz = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate_hz}


def synthesize_speech_gemini(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Gemini TTS i odtwarza ją.
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy.
        audio_output_device_index (int): Indeks urządzenia wyjściowego.
    """
    if not text_to_synthesize or not text_to_synthesize.strip():
        return

    if not GEMINI_AVAILABLE:
        log_message("[Watus][TTS] Gemini TTS not available")
        return

    if not config.GEMINI_API_KEY:
        log_message("[Watus][TTS] GEMINI_API_KEY not set")
        return

    try:
        log_message(f"[Watus][TTS] Generating Gemini TTS for: {text_to_synthesize[:50]}...")

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        content_parts = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text_to_synthesize),
                ],
            ),
        ]

        gen_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.GEMINI_VOICE
                    )
                )
            ),
        )

        start_time = time.time()

        synthesized_audio_bytes = b""
        mime_type_string = ""
        for response_chunk in client.models.generate_content_stream(
            model=config.GEMINI_MODEL,
            contents=content_parts,
            config=gen_config,
        ):
            if (
                response_chunk.candidates is None
                or response_chunk.candidates[0].content is None
                or response_chunk.candidates[0].content.parts is None
            ):
                continue
            if response_chunk.candidates[0].content.parts[0].inline_data and response_chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data_obj = response_chunk.candidates[0].content.parts[0].inline_data
                synthesized_audio_bytes = inline_data_obj.data
                mime_type_string = inline_data_obj.mime_type
                break
            else:
                if hasattr(response_chunk, 'text') and response_chunk.text:
                    log_message(f"[Watus][TTS] No audio, got text: {response_chunk.text}")

        if synthesized_audio_bytes:
            if mime_type_string and mime_type_string.startswith("audio/"):
                wav_data_bytes = _add_wav_header(synthesized_audio_bytes, mime_type_string)
            else:
                log_message("[Watus][TTS] Unsupported audio format")
                return

            import io
            wav_buffer_io = io.BytesIO(wav_data_bytes)
            audio_samples_array, sample_rate_hz = sf.read(wav_buffer_io, dtype='float32')

            sd.play(audio_samples_array, sample_rate_hz, device=audio_output_device_index, blocking=True)

            log_message(f"[Perf] Gemini_TTS_play_ms={int((time.time() - start_time) * 1000)}")
        else:
            log_message("[Watus][TTS] No audio data generated by Gemini")

    except Exception as e:
        log_message(f"[Watus][TTS] Gemini TTS error: {e}")


LOADED_XTTS_MODEL = None

def _initialize_xtts_model():
    """Ładuje model XTTS do pamięci."""
    global LOADED_XTTS_MODEL
    if LOADED_XTTS_MODEL: return True
    
    try:
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"[Watus][TTS] Loading XTTS model on {device}...")
        
        # Ładujemy model XTTS v2
        LOADED_XTTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        log_message("[Watus][TTS] XTTS model loaded.")
        return True
    except Exception as e:
        log_message(f"[Watus][TTS] Failed to load XTTS: {e}")
        return False

def synthesize_speech_xtts(text_to_synthesize: str, audio_output_device_index):
    """Generuje mowę za pomocą Coqui XTTS-v2."""
    if not text_to_synthesize or not text_to_synthesize.strip(): return
    
    if not _initialize_xtts_model():
        log_message("[Watus][TTS] XTTS initialization failed.")
        return

    speaker_wav = config.XTTS_SPEAKER_WAV
    if not os.path.isfile(speaker_wav):
        log_message(f"[Watus][TTS] Brak pliku referencyjnego głosu: {speaker_wav}")
        return

    try:
        start_time = time.time()
        # Generujemy do pliku tymczasowego (API TTS często preferuje pliki)
        # Można też użyć .tts() i dostać wav w pamięci, ale .tts_to_file jest prostsze w obsłudze formatów
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Synteza
        LOADED_XTTS_MODEL.tts_to_file(
            text=text_to_synthesize,
            speaker_wav=speaker_wav,
            language=config.XTTS_LANGUAGE,
            file_path=output_path
        )
        
        # Odtwarzanie
        audio_data, sr = sf.read(output_path, dtype="float32")
        sd.play(audio_data, sr, device=audio_output_device_index, blocking=True)
        
        log_message(f"[Perf] XTTS_play_ms={int((time.time() - start_time) * 1000)}")
        
        try:
            os.unlink(output_path)
        except:
            pass
            
    except Exception as e:
        log_message(f"[Watus][TTS] XTTS error: {e}")


def synthesize_speech_and_play(text_to_synthesize: str, audio_output_device_index):
    """
    Uniwersalna funkcja TTS - wybiera Piper, Gemini lub XTTS w zależności od konfiguracji.
    
    Argumenty:
        text_to_synthesize (str): Tekst do wypowiedzenia.
        audio_output_device_index (int): Indeks urządzenia wyjściowego.
    """
    provider = config.TTS_PROVIDER
    
    if provider == "gemini":
        log_message("[Watus][TTS] Using Gemini TTS")
        synthesize_speech_gemini(text_to_synthesize, audio_output_device_index)
    elif provider == "xtts":
        log_message("[Watus][TTS] Using XTTS-v2")
        synthesize_speech_xtts(text_to_synthesize, audio_output_device_index)
    else:
        log_message("[Watus][TTS] Using Piper TTS")
        synthesize_speech_piper(text_to_synthesize, audio_output_device_index)
