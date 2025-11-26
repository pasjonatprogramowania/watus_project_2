import time
import re
import sys
import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from faster_whisper import WhisperModel
from .common import log_message, append_line_to_dialog_history
from . import config
from .verifier import create_speaker_verifier

def check_if_text_contains_wake_word(text_to_check: str) -> bool:
    """
    Sprawdza, czy w tekście występuje słowo wybudzające (wake word).
    
    Argumenty:
        text_to_check (str): Tekst do sprawdzenia.
        
    Zwraca:
        bool: True jeśli znaleziono wake word.
    """
    normalized_text = re.sub(r'[^\w\s]', '', text_to_check.lower())
    for wake_phrase in config.WAKE_WORDS:
        normalized_wake_phrase = re.sub(r'[^\w\s]', '', wake_phrase.lower())
        if normalized_wake_phrase in normalized_text:
            return True
    return False

class SpeechToTextProcessingEngine:
    """
    Silnik rozpoznawania mowy (STT) i zarządzania pętlą audio.
    Obsługuje VAD (wykrywanie głosu), transkrypcję (Whisper) i logikę lidera.
    """
    
    def __init__(self, system_state, message_bus, callbacks):
        """
        Inicjalizuje silnik STT.
        
        Argumenty:
            system_state (SystemState): Obiekt stanu systemu.
            message_bus (ZMQMessageBus): Szyna komunikatów.
            callbacks (dict): Słownik funkcji zwrotnych (np. do zmiany stanu UI).
        """
        self.state = system_state
        self.bus = message_bus
        self.callbacks = callbacks # {indicate_think_state, indicate_listen_state, indicate_idle_state}
        self.vad_processor = webrtcvad.Vad(config.VAD_MODE)
        self.stt_provider_name = config.STT_PROVIDER

        log_message(f"[Watus] STT ready (device={config.IN_DEV_ENV} sr={config.SAMPLE_RATE} block={config.BLOCK_SIZE})")

        self._initialize_local_whisper_model()
        self.speaker_verifier = create_speaker_verifier()
        self.emit_cooldown_ms = config.EMIT_COOLDOWN_MS
        self.cooldown_until_timestamp_ms = 0

    def _initialize_local_whisper_model(self):
        """Ładuje lokalny model Faster Whisper."""
        log_message(f"[Watus] FasterWhisper init: model={config.WHISPER_MODEL_NAME} device={config.WHISPER_DEVICE} "
            f"compute={config.WHISPER_COMPUTE} cpu_threads={config.CPU_THREADS} workers={config.WHISPER_NUM_WORKERS}")
        start_time = time.time()
        self.whisper_model = WhisperModel(
            config.WHISPER_MODEL_NAME,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
            cpu_threads=config.CPU_THREADS,
            num_workers=config.WHISPER_NUM_WORKERS
        )
        log_message(f"[Watus] STT FasterWhisper loaded ({int((time.time() - start_time) * 1000)} ms)")
        self.stt_provider_name = "local"

    @staticmethod
    def _calculate_rms_dbfs(audio_samples: np.ndarray, epsilon=1e-9):
        """Oblicza głośność (RMS dBFS) dla ramki audio."""
        root_mean_square = np.sqrt(np.mean(np.square(audio_samples) + epsilon))
        return 20 * np.log10(max(root_mean_square, epsilon))

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        """Sprawdza czy ramka zawiera mowę (używając WebRTC VAD)."""
        try:
            return self.vad_processor.is_speech(frame_bytes, config.SAMPLE_RATE)
        except Exception:
            return False

    def _transcribe_audio_segment(self, audio_samples_float32: np.ndarray) -> str:
        """Transkrybuje segment audio używając modelu Whisper."""
        start_time = time.time()
        segments, _ = self.whisper_model.transcribe(
            audio_samples_float32, language="pl", beam_size=1, vad_filter=False
        )
        transcribed_text = "".join(seg.text for seg in segments)
        log_message(f"[Perf] ASR_local_ms={int((time.time() - start_time) * 1000)} len={len(transcribed_text)}")
        return transcribed_text

    def start_listening_loop(self, input_device_index):
        """
        Główna pętla nasłuchiwania audio.
        Czyta z mikrofonu, wykrywa mowę (VAD) i uruchamia przetwarzanie.
        
        Argumenty:
            input_device_index (int): Indeks urządzenia wejściowego.
        """
        audio_input_stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=config.BLOCK_SIZE, device=input_device_index
        )

        frame_duration_ms = int(1000 * config.BLOCK_SIZE / config.SAMPLE_RATE)
        silence_frames_threshold = max(1, config.SIL_MS_END // frame_duration_ms)
        min_speech_frames_threshold = max(1, config.VAD_MIN_MS // frame_duration_ms)

        pre_speech_audio_buffer = deque(maxlen=config.PREBUFFER_FRAMES)
        recorded_speech_frames_buffer = bytearray()
        is_currently_speaking = False
        speech_start_timestamp_ms = None
        last_voice_activity_timestamp_ms = 0
        listening_state_flag = None

        consecutive_voice_frames_count = 0
        last_audio_volume_dbfs = None
        total_speech_frames_count = 0
        consecutive_silence_frames_count = 0
        silence_gap_duration_ms = 0

        with audio_input_stream:
            while True:
                current_time_ms = int(time.time() * 1000)

                if self.state.is_input_blocked():
                    if listening_state_flag is not False:
                        self.callbacks['indicate_idle_state']()
                        listening_state_flag = False
                    is_currently_speaking = False
                    recorded_speech_frames_buffer = bytearray()
                    speech_start_timestamp_ms = None
                    consecutive_voice_frames_count = 0
                    last_audio_volume_dbfs = None
                    total_speech_frames_count = 0
                    consecutive_silence_frames_count = 0
                    silence_gap_duration_ms = 0
                    pre_speech_audio_buffer.clear()
                    time.sleep(0.01)
                    continue

                if current_time_ms < self.cooldown_until_timestamp_ms:
                    time.sleep(0.003)
                    continue

                if listening_state_flag is not True:
                    self.callbacks['indicate_listen_state']()
                    listening_state_flag = True

                try:
                    audio_chunk, _ = audio_input_stream.read(config.BLOCK_SIZE)
                except Exception as e:
                    log_message(f"[Watus][STT] read err: {e}")
                    time.sleep(0.01)
                    continue

                frame_bytes = audio_chunk.tobytes()
                pre_speech_audio_buffer.append(frame_bytes)
                is_speech_frame = self._vad_is_speech(frame_bytes)

                # --- twardy start: kilka ramek powyżej progu dBFS ---
                if not is_currently_speaking:
                    if is_speech_frame:
                        current_samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        current_dbfs = float(self._calculate_rms_dbfs(current_samples))
                        if current_dbfs > config.START_MIN_DBFS:
                            consecutive_voice_frames_count += 1
                        else:
                            consecutive_voice_frames_count = 0
                        if consecutive_voice_frames_count >= config.START_MIN_FRAMES:
                            is_currently_speaking = True
                            recorded_speech_frames_buffer = bytearray()
                            if pre_speech_audio_buffer:
                                recorded_speech_frames_buffer.extend(b''.join(pre_speech_audio_buffer))
                            speech_start_timestamp_ms = current_time_ms - (len(pre_speech_audio_buffer) * frame_duration_ms)
                            last_voice_activity_timestamp_ms = current_time_ms
                            total_speech_frames_count = 0
                            consecutive_silence_frames_count = 0
                            silence_gap_duration_ms = 0
                            last_audio_volume_dbfs = None
                            consecutive_voice_frames_count = 0
                    else:
                        consecutive_voice_frames_count = 0
                    time.sleep(0.0005)
                    continue

                # --- w turze mowy ---
                if is_speech_frame:
                    recorded_speech_frames_buffer.extend(frame_bytes)
                    last_voice_activity_timestamp_ms = current_time_ms
                    consecutive_silence_frames_count = 0
                    silence_gap_duration_ms = 0
                    total_speech_frames_count += 1

                    current_samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    current_dbfs = float(self._calculate_rms_dbfs(current_samples))
                    if last_audio_volume_dbfs is None: last_audio_volume_dbfs = current_dbfs

                    if config.END_AT_DBFS_DROP > 0:
                        if total_speech_frames_count >= min_speech_frames_threshold and (
                                current_time_ms - (speech_start_timestamp_ms or current_time_ms)) >= config.MIN_MS_BEFORE_ENDPOINT:
                            if (last_audio_volume_dbfs - current_dbfs) >= config.END_AT_DBFS_DROP:
                                is_currently_speaking = False
                                speech_duration_ms = last_voice_activity_timestamp_ms - (speech_start_timestamp_ms or last_voice_activity_timestamp_ms)
                                self._process_recorded_speech_segment(recorded_speech_frames_buffer, speech_start_timestamp_ms, last_voice_activity_timestamp_ms, speech_duration_ms)
                                listening_state_flag = None
                                recorded_speech_frames_buffer = bytearray()
                                speech_start_timestamp_ms = None
                                total_speech_frames_count = 0
                                consecutive_silence_frames_count = 0
                                last_audio_volume_dbfs = None
                                silence_gap_duration_ms = 0
                                self.cooldown_until_timestamp_ms = current_time_ms + self.emit_cooldown_ms
                                continue
                    else:
                        last_audio_volume_dbfs = current_dbfs

                else:
                    # brak VAD -> liczymy ciszę i tolerowany GAP
                    consecutive_silence_frames_count += 1
                    silence_gap_duration_ms += frame_duration_ms
                    # toleruj krótką przerwę w środku wypowiedzi
                    if silence_gap_duration_ms < config.GAP_TOL_MS:
                        recorded_speech_frames_buffer.extend(frame_bytes)  # dodajemy ciszę do bufora na wszelki wypadek
                        continue

                    if consecutive_silence_frames_count >= silence_frames_threshold and (current_time_ms - (speech_start_timestamp_ms or current_time_ms)) >= config.MIN_MS_BEFORE_ENDPOINT:
                        is_currently_speaking = False
                        speech_duration_ms = last_voice_activity_timestamp_ms - (speech_start_timestamp_ms or last_voice_activity_timestamp_ms)
                        if speech_duration_ms >= config.VAD_MIN_MS and len(recorded_speech_frames_buffer) > 0:
                            self._process_recorded_speech_segment(recorded_speech_frames_buffer, speech_start_timestamp_ms, last_voice_activity_timestamp_ms, speech_duration_ms)
                            self.cooldown_until_timestamp_ms = current_time_ms + self.emit_cooldown_ms
                        listening_state_flag = None
                        recorded_speech_frames_buffer = bytearray()
                        speech_start_timestamp_ms = None
                        total_speech_frames_count = 0
                        consecutive_silence_frames_count = 0
                        last_audio_volume_dbfs = None
                        silence_gap_duration_ms = 0

                # twardy limit
                if is_currently_speaking and speech_start_timestamp_ms and (current_time_ms - speech_start_timestamp_ms) >= config.MAX_UTT_MS:
                    is_currently_speaking = False
                    speech_duration_ms = last_voice_activity_timestamp_ms - (speech_start_timestamp_ms or last_voice_activity_timestamp_ms)
                    if speech_duration_ms >= config.VAD_MIN_MS and len(recorded_speech_frames_buffer) > 0:
                        self._process_recorded_speech_segment(recorded_speech_frames_buffer, speech_start_timestamp_ms, last_voice_activity_timestamp_ms, speech_duration_ms)
                        self.cooldown_until_timestamp_ms = current_time_ms + self.emit_cooldown_ms
                    listening_state_flag = None
                    recorded_speech_frames_buffer = bytearray()
                    speech_start_timestamp_ms = None
                    total_speech_frames_count = 0
                    consecutive_silence_frames_count = 0
                    last_audio_volume_dbfs = None
                    silence_gap_duration_ms = 0

                time.sleep(0.0005)

    def _process_recorded_speech_segment(self, recorded_speech_frames_buffer: bytearray, speech_start_timestamp_ms: int, last_voice_activity_timestamp_ms: int, speech_duration_ms: int):
        """
        Przetwarza nagrany segment mowy: transkrypcja, weryfikacja, wysłanie do busa.
        """
        self.callbacks['indicate_think_state']()
        audio_samples_float32 = np.frombuffer(recorded_speech_frames_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        audio_volume_dbfs = float(self._calculate_rms_dbfs(audio_samples_float32))

        log_message(f"[DEBUG] _finalize: frames={len(recorded_speech_frames_buffer)} dur={speech_duration_ms}ms dbfs={audio_volume_dbfs:.2f} (min={config.ASR_MIN_DBFS})")

        if audio_volume_dbfs < config.ASR_MIN_DBFS:
            log_message(f"[DEBUG] Signal too quiet ({audio_volume_dbfs:.2f} < {config.ASR_MIN_DBFS}), ignoring.")
            return

        # 1. Transkrypcja
        transcribed_text = self._transcribe_audio_segment(audio_samples_float32).strip()
        log_message(f"[DEBUG] Transcribed text: '{transcribed_text}'")

        if not transcribed_text:
            log_message("[DEBUG] Empty transcription, ignoring.")
            return

        # 2. Logika Lidera oparta na słowie-klucz
        verification_result = {}
        is_leader_detected = False
        is_wake_word_detected = check_if_text_contains_wake_word(transcribed_text)
        log_message(f"[DEBUG] is_wake_word={is_wake_word_detected} enrolled={getattr(self.speaker_verifier, 'enrolled', False)}")

        if getattr(self.speaker_verifier, "enabled", False):
            if is_wake_word_detected:
                log_message("[Watus][SPK] Wykryto słowo-klucz. Rejestrowanie nowego lidera.")
                self.speaker_verifier.enroll_voice_samples(audio_samples_float32, config.SAMPLE_RATE)
                verification_result = self.speaker_verifier.verify_speaker_identity(audio_samples_float32, config.SAMPLE_RATE, audio_volume_dbfs)
                is_leader_detected = True
            elif self.speaker_verifier.enrolled:
                verification_result = self.speaker_verifier.verify_speaker_identity(audio_samples_float32, config.SAMPLE_RATE, audio_volume_dbfs)
                is_leader_detected = bool(verification_result.get("is_leader", False))
                log_message(f"[DEBUG] Verification result: {verification_result}")
            else:
                log_message(f"[Watus][SPK] Brak lidera i słowa-klucz. Ignoruję: '{transcribed_text}'")
                return
        else:
            # Jeśli weryfikacja jest wyłączona, każda wypowiedź jest od "lidera"
            is_leader_detected = not config.SPEAKER_REQUIRE_MATCH

        # 3. Przygotowanie i wysłanie danych
        ts_start = (speech_start_timestamp_ms or last_voice_activity_timestamp_ms) / 1000.0
        ts_end = last_voice_activity_timestamp_ms / 1000.0
        turn_id = int(last_voice_activity_timestamp_ms)

        dialog_line_object = {
            "type": "leader_utterance" if is_leader_detected else "unknown_utterance",
            "session_id": self.state.session_id,
            "group_id": f"{'leader' if is_leader_detected else 'unknown'}_{turn_id}",
            "speaker_id": "leader" if is_leader_detected else "unknown",
            "is_leader": is_leader_detected,
            "turn_ids": [turn_id],
            "text_full": transcribed_text,
            "category": "wypowiedź",
            "reply_hint": is_leader_detected,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "dbfs": audio_volume_dbfs,
            "verify": verification_result,
            "emit_reason": "endpoint",
            "ts": time.time()
        }
        append_line_to_dialog_history(dialog_line_object, config.DIALOG_PATH)

        if is_leader_detected:
            log_message(f"[Watus][PUB] dialog.leader → group={dialog_line_object['group_id']} spk_score={verification_result.get('score')}")
            self.state.set_awaiting_reply_flag(True)
            self.bus.publish_leader_utterance(dialog_line_object)
            self.state.block_input_until_reply_received()
            self.state.tts_pending_until_timestamp = time.time() + 0.6
        else:
            log_message(f"[Watus][SKIP] unknown (score={verification_result.get('score', 0):.2f}) zapisany, nie wysyłam ZMQ")
