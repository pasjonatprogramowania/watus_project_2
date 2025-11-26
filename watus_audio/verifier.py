import time
import numpy as np
from .common import log_message
from . import config

class _NoopVerifier:
    """
    Pusta implementacja weryfikatora, używana gdy weryfikacja jest wyłączona lub brak zależności.
    """
    enabled = True

    def __init__(self): self._enrolled_embedding = None

    @property
    def enrolled(self): return False

    def enroll_voice_samples(self, audio_samples, sample_rate_hz): pass

    def verify_speaker_identity(self, audio_samples, sample_rate_hz, audio_volume_dbfs): return {"enabled": False}


def create_speaker_verifier():
    """
    Tworzy i zwraca instancję weryfikatora mówcy (Speaker Verification).
    Jeśli biblioteki (torch, speechbrain) nie są dostępne, zwraca wersję Noop.
    """
    if not config.SPEAKER_VERIFY: return _NoopVerifier()
    try:
        import torch  # noqa
        from speechbrain.pretrained import EncoderClassifier  # noqa
    except Exception as e:
        log_message(f"[Watus][SPK] OFF (brak zależności): {e}")
        return _NoopVerifier()

    class _SbVerifier:
        """
        Weryfikator mówcy oparty na SpeechBrain (ECAPA-TDNN).
        """
        enabled = True

        def __init__(self):
            import torch
            self.threshold = config.SPEAKER_THRESHOLD
            self.sticky_threshold = config.SPEAKER_STICKY_THRESHOLD
            self.back_threshold = config.SPEAKER_BACK_THRESHOLD
            self.grace_period = config.SPEAKER_GRACE
            self.sticky_seconds = config.SPEAKER_STICKY_SEC
            self._speaker_encoder_classifier = None
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._enrolled_embedding_vector = None
            self._enroll_timestamp = 0.0

        @property
        def enrolled(self):
            """Czy wzorzec głosu lidera jest zarejestrowany?"""
            return self._enrolled_embedding_vector is not None

        def _ensure_model_loaded(self):
            """Ładuje model SpeechBrain jeśli nie jest jeszcze załadowany."""
            from speechbrain.pretrained import EncoderClassifier
            if self._speaker_encoder_classifier is None:
                self._speaker_encoder_classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self._device},
                    savedir="models/ecapa",
                )

        @staticmethod
        def _resample_to_16k(audio_samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
            """Konwertuje próbki audio do 16kHz (wymagane przez model)."""
            if sample_rate_hz == 16000: return audio_samples.astype(np.float32)
            ratio = 16000.0 / sample_rate_hz
            num_output_samples = int(round(len(audio_samples) * ratio))
            output_indices = np.linspace(0, len(audio_samples) - 1, num=num_output_samples, dtype=np.float32)
            input_indices = np.arange(len(audio_samples), dtype=np.float32)
            return np.interp(output_indices, input_indices, audio_samples).astype(np.float32)

        def _compute_embedding(self, audio_samples: np.ndarray, sample_rate_hz: int):
            """Oblicza wektor cech (embedding) dla podanych próbek audio."""
            import torch
            self._ensure_model_loaded()
            wav_resampled = self._resample_to_16k(audio_samples, sample_rate_hz)
            tensor_input = torch.tensor(wav_resampled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding_vector = self._speaker_encoder_classifier.encode_batch(tensor_input).squeeze(0).squeeze(0)
            return embedding_vector.detach().cpu().numpy().astype(np.float32)

        def enroll_voice_samples(self, audio_samples: np.ndarray, sample_rate_hz: int):
            """
            Rejestruje próbki głosu jako wzorzec lidera.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio (float32).
                sample_rate_hz (int): Częstotliwość próbkowania.
            """
            try:
                embedding_vector = self._compute_embedding(audio_samples, sample_rate_hz)
                self._enrolled_embedding_vector = embedding_vector
                self._enroll_timestamp = time.time()
                log_message(f"[Watus][SPK] Enrolled new leader voice.")
            except Exception as e:
                log_message(f"[Watus][SPK] enroll err: {e}")

        def verify_speaker_identity(self, audio_samples: np.ndarray, sample_rate_hz: int, audio_volume_dbfs: float) -> dict:
            """
            Weryfikuje, czy podane próbki audio należą do zarejestrowanego lidera.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio.
                sample_rate_hz (int): Częstotliwość próbkowania.
                audio_volume_dbfs (float): Głośność próbki (dBFS).
                
            Zwraca:
                dict: Wynik weryfikacji (score, is_leader).
            """
            if self._enrolled_embedding_vector is None:
                return {"enabled": True, "enrolled": False}
            import torch, torch.nn.functional as F
            current_embedding = self._compute_embedding(audio_samples, sample_rate_hz)
            similarity_score = float(F.cosine_similarity(
                torch.tensor(current_embedding, dtype=torch.float32).flatten(),
                torch.tensor(self._enrolled_embedding_vector, dtype=torch.float32).flatten(), dim=0, eps=1e-8
            ).detach().cpu().item())
            
            current_time = time.time()
            age_seconds = current_time - self._enroll_timestamp
            is_leader = False
            
            adjusted_threshold = (
                        self.sticky_threshold - self.grace_period) if audio_volume_dbfs > -22.0 else self.sticky_threshold  # emocje → głośniej → trochę łagodniej
            
            if age_seconds <= self.sticky_seconds and similarity_score >= adjusted_threshold:
                is_leader = True
            elif similarity_score >= self.threshold:
                is_leader = True
            elif similarity_score >= self.back_threshold and age_seconds <= self.sticky_seconds:
                is_leader = True
            
            return {"enabled": True, "enrolled": True, "score": similarity_score, "is_leader": bool(is_leader), "sticky_age_s": age_seconds}

    return _SbVerifier()
