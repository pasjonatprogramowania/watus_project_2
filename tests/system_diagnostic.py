#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostyka systemu WATUS - sprawdza wszystkie komponenty
"""

import os
import sys
import traceback
from pathlib import Path

def check_python_version():
    """Sprawdź wersję Python"""
    print(f" Python: {sys.version}")
    if sys.version_info < (3, 11):
        print("  Wymaga Python 3.11+")
        return False
    return True

def check_audio_system():
    """Sprawdź system audio"""
    print("\n SYSTEM AUDIO:")
    try:
        import sounddevice as sd
        print(f" sounddevice: {sd.__version__}")
        
        devices = sd.query_devices()
        print(f" Urządzeń audio znaleziono: {len(devices)}")
        
        if devices:
            print(" Dostępne urządzenia:")
            for i, d in enumerate(devices):
                input_ch = d.get('max_input_channels', 0)
                output_ch = d.get('max_output_channels', 0)
                print(f"  [{i}] {d['name']} (IN:{input_ch} OUT:{output_ch})")
        else:
            print(" BRAK URZĄDZEŃ AUDIO")
            print(" W środowiskach serwerowych bez fizycznych urządzeń audio to normalne")
        
        return True
    except ImportError as e:
        print(f" sounddevice: {e}")
        return False
    except Exception as e:
        print(f" Błąd audio: {e}")
        return False

def check_ai_models():
    """Sprawdź modele AI"""
    print("\n MODELE AI:")
    
    # PyTorch
    try:
        import torch
        print(f" PyTorch: {torch.__version__}")
        print(f"CUDA dostępna: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU devices: {torch.cuda.device_count()}")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print(" PyTorch nie zainstalowany")
        return False
    
    # Faster-Whisper
    try:
        import faster_whisper
        print(f" Faster-Whisper: {faster_whisper.__version__}")
    except ImportError:
        print(" Faster-Whisper nie zainstalowany")
        return False
    
    # SpeechBrain
    try:
        import speechbrain
        print(f" SpeechBrain: {speechbrain.__version__}")
        # Test ECAPA
        try:
            from speechbrain.pretrained import EncoderClassifier
            print(" ECAPA model dostępny")
        except Exception as e:
            print(f"  ECAPA błąd: {e}")
    except ImportError:
        print(" SpeechBrain nie zainstalowany")
        return False
    
    return True

def check_zmq():
    """Sprawdź komunikację ZMQ"""
    print("\n📡 KOMUNIKACJA ZMQ:")
    try:
        import zmq
        print(f" PyZMQ: {zmq.zmq_version()}")
        
        # Test basic socket creation
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        print(" ZMQ PUB socket: OK")
        
        socket = ctx.socket(zmq.SUB)
        print(" ZMQ SUB socket: OK")
        ctx.destroy()
        
        return True
    except ImportError:
        print(" PyZMQ nie zainstalowany")
        return False
    except Exception as e:
        print(f" ZMQ błąd: {e}")
        return False

def check_config():
    """Sprawdź konfigurację"""
    print("\n  KONFIGURACJA:")
    
    env_file = Path(".env")
    if env_file.exists():
        print(" Plik .env istnieje")
        
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
        # Sprawdź kluczowe zmienne
        critical_vars = [
            'ZMQ_PUB_ADDR', 'ZMQ_SUB_ADDR',
            'WHISPER_MODEL', 'WHISPER_DEVICE', 'WHISPER_COMPUTE_TYPE',
            'WATUS_SR', 'WATUS_BLOCKSIZE'
        ]
        
        for var in critical_vars:
            value = os.environ.get(var)
            if value:
                print(f" {var}: {value}")
            else:
                print(f"  {var}: nie ustawiona")
        
        # Sprawdź urządzenia audio
        input_dev = os.environ.get('WATUS_INPUT_DEVICE')
        output_dev = os.environ.get('WATUS_OUTPUT_DEVICE')
        if not input_dev or not output_dev:
            print(" Urządzenia audio nie skonfigurowane - będą autodetekowane")
        
        # Sprawdź Piper
        piper_vars = ['PIPER_BIN', 'PIPER_MODEL', 'PIPER_CONFIG']
        piper_missing = []
        for var in piper_vars:
            if not os.environ.get(var):
                piper_missing.append(var)
        
        if piper_missing:
            print(f"  Piper nie skonfigurowany: {', '.join(piper_missing)}")
        else:
            print(" Piper skonfigurowany")
            
    else:
        print(" Brak pliku .env")
        return False
    
    return True

def check_dependencies():
    """Sprawdź wszystkie zależności"""
    print("\n ZALEŻNOŚCI:")
    
    packages = [
        'numpy', 'scipy', 'webrtcvad', 'requests',
        'python-dotenv', 'transformers', 'tokenizers',
        'onnxruntime', 'protobuf'
    ]
    
    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f" {pkg}")
        except ImportError:
            print(f" {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"  Brakuje pakietów: {', '.join(missing)}")
        return False
    
    return True

def test_watus_imports():
    """Test importów modułów watus"""
    print("\n TEST IMPORTÓW WATUS:")
    
    try:
        from led_controller import LEDController
        print(" led_controller")
    except Exception as e:
        print(f" led_controller: {e}")
        return False
    
    try:
        # Test bez faktycznego uruchomienia
        import watus
        print(" watus.py import: OK")
        return True
    except Exception as e:
        print(f" watus.py: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Główna diagnostyka"""
    print(" DIAGNOSTYKA SYSTEMU WATUS")
    print("=" * 50)
    
    results = {
        'python': check_python_version(),
        'audio': check_audio_system(),
        'ai': check_ai_models(),
        'zmq': check_zmq(),
        'config': check_config(),
        'deps': check_dependencies(),
        'imports': test_watus_imports()
    }
    
    print("\n" + "=" * 50)
    print(" PODSUMOWANIE:")
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        status_text = " DZIAŁA" if status else " BŁĄD"
        print(f"{name.upper():12} {status_text}")
    
    print(f"\n Status ogólny: {working}/{total} komponentów działa")
    
    if working == total:
        print(" SYSTEM GOTOWY DO URUCHOMIENIA!")
        print("\n Uruchom: python3 watus.py")
    else:
        print("  Wymaga naprawy błędów przed uruchomieniem")
        
    return working == total

if __name__ == "__main__":
    main()
