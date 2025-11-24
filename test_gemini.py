#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Gemini TTS Synthesis
"""

import base64
import mimetypes
import os
import re
import struct
import time
import tempfile
import io
import soundfile as sf
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to: {file_name}")


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Parses bits per sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def test_gemini_voice(voice_name="Callirrhoe"):
    """Test Gemini TTS synthesis with different voices"""

    # Load environment variables
    load_dotenv(dotenv_path=Path('.env'), override=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print('❌ GEMINI_API_KEY nie ustawiony w .env')
        return False

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-tts")

    try:
        client = genai.Client(api_key=api_key)
        print(f'✅ Klient Gemini utworzony (voice: {voice_name})')
    except Exception as e:
        print(f'❌ Błąd tworzenia klienta Gemini: {e}')
        return False

    # Test syntezy
    test_text = f'Test polskiego głosu - głos {voice_name}'

    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=test_text)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        print(f'📝 Generuję TTS: "{test_text}"...')

        t0 = time.time()
        audio_collected = False

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue

            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                audio_data = inline_data.data
                mime_type = inline_data.mime_type
                audio_collected = True
                break

        if not audio_collected:
            print('❌ Nie odebrano audio data z Gemini')
            return False

        gen_time = time.time() - t0

        # Convert to WAV format if needed
        if mime_type and mime_type.startswith("audio/"):
            file_extension = mimetypes.guess_extension(mime_type)
            if file_extension == ".wav":
                data_buffer = audio_data
            else:
                data_buffer = convert_to_wav(audio_data, mime_type)
                file_extension = ".wav"
        else:
            print('❌ Nieznany mime_type audio')
            return False

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(data_buffer)
            file_name = tmp.name

        # Load and check file
        if os.path.exists(file_name) and os.path.getsize(file_name) > 1000:
            # Try to load with soundfile
            try:
                audio_array, sample_rate = sf.read(file_name, dtype='float32')
                print(f'✅ TTS Gemini działa! Plik: {file_name}')
                print(f'   Tekst: "{test_text}" ({len(audio_array)} samples @ {sample_rate} Hz)')
                print('.3f')
                print(f'   Głos: {voice_name}')
                print('   Możesz odtworzyć plik audio, aby sprawdzić jakość')
                return True
            except Exception as e:
                print(f'❌ Błąd ładowania audio: {e}')
                return False
        else:
            print('❌ Plik audio jest pusty lub zbyt mały')
            os.unlink(file_name) if os.path.exists(file_name) else None
            return False

    except Exception as e:
        print(f'❌ Błąd syntezy Gemini: {e}')
        return False


if __name__ == "__main__":
    print("=== Test Gemini TTS ===")

    # Test with current voice from .env
    load_dotenv(dotenv_path=Path('.env'), override=True)
    current_voice = os.environ.get("GEMINI_VOICE", "Callirrhoe")

    print(f"Testowanie głosu z .env: {current_voice}")
    if test_gemini_voice(current_voice):
        print("✅ Głos działa dobrze!")
    else:
        print("❌ Problem z obecnym głosem")

    print("\nTestowanie alternatywnych głosów:")
    voices_to_test = ["Zephyr", "Charon", "Kore", "Fenrir", "Aoede"]

    for voice in voices_to_test:
        if voice != current_voice:  # Don't test current voice twice
            print(f"\n--- Testowanie {voice} ---")
            success = test_gemini_voice(voice)
            if success:
                print(f"✅ {voice} działa!")
                break  # Stop at first working voice

    print("\n=== Podsumowanie ===")
    print("Aby zmienić głos, zaktualizuj GEMINI_VOICE w pliku .env")
