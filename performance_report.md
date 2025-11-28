# Raport Analizy Wydajności i Migracji do Go

## 1. Analiza Obecnego Stanu (Python)

Twoja aplikacja Watus składa się z kilku kluczowych komponentów:
1.  **Pętla Audio (Input/Output):** Obsługiwana przez `sounddevice` (wrapper na C PortAudio).
2.  **VAD (Detekcja głosu):** Prawdopodobnie `webrtcvad` (wrapper na C).
3.  **STT (Whisper):** `faster-whisper` (wrapper na C++ CTranslate2).
4.  **TTS (Piper/XTTS):** Wrappery na biblioteki C++/ONNX/PyTorch.
5.  **Logika (LLM/State):** Czysty Python.
6.  **Komunikacja:** ZeroMQ (bardzo szybkie, asynchroniczne).

### Gdzie są wąskie gardła?
*   **Python GIL (Global Interpreter Lock):** To główny problem. Mimo że biblioteki AI (Whisper, Torch) zwalniają GIL podczas obliczeń, to sama pętla obsługi audio, logiki i przesyłania danych w Pythonie może powodować mikro-opóźnienia (jitter), szczególnie gdy system jest obciążony.
*   **Zarządzanie pamięcią:** Python ma narzut na zarządzanie obiektami, co przy przetwarzaniu tysięcy ramek audio na sekundę może być zauważalne (choć `numpy` to łagodzi).

## 2. Czy przepisanie na Go ma sens?

### Co zyskałbyś w Go?
*   **Prawdziwa współbieżność:** Goroutines są lekkie i działają równolegle na wielu rdzeniach bez blokady GIL. Idealne do obsługi audio (nagrywanie) i logiki (wysyłanie ZMQ) w tym samym czasie.
*   **Stabilność i Latencja:** Go (jako język kompilowany z Garbage Collectorem nastawionym na niskie opóźnienia) zapewniłby bardziej przewidywalny czas reakcji pętli audio. Mniej "czknięć" dźwięku.
*   **Mniejsze zużycie RAM:** Go zazwyczaj zużywa mniej pamięci niż Python dla samej logiki "sklejania" systemów.
*   **Jeden plik wykonywalny:** Łatwiejsze wdrażanie (brak venv, pip install itd.), choć modele nadal muszą być na dysku.

### Czego NIE zyskasz (lub stracisz)?
*   **Szybkość AI (STT/TTS):** Tutaj zysk będzie **ZEROWY**. Python używa tych samych bibliotek C++/CUDA co Go. Whisper w Go (np. przez bindingi do `whisper.cpp`) działa tak samo szybko jak `faster-whisper` w Pythonie.
*   **Ekosystem:** Python to król AI. XTTS, najnowsze modele LLM, eksperymentalne funkcje – to wszystko trafia najpierw do Pythona. Przepisanie obsługi XTTS czy Gemini na Go wymagałoby szukania nieoficjalnych bibliotek lub pisania własnych wrapperów HTTP/C++.

## 3. Rekomendacja: Architektura Hybrydowa

Nie przepisuj wszystkiego. To ogrom pracy z małym zyskiem dla części AI. Zamiast tego, rozważ wydzielenie **warstwy czasu rzeczywistego**.

### Proponowany podział:

#### A. Usługa "Ucho i Usta" (Go) - *Zalecane do przepisania*
To mały, super-szybki program w Go, który:
1.  Nasłuchuje mikrofonu (PortAudio).
2.  Robi VAD (WebRTC) i Wake Word (Porcupine/OpenWakeWord).
3.  Buforuje audio.
4.  Gdy wykryje mowę -> wysyła surowe ramki audio przez ZMQ do Pythona.
5.  Odbiera audio z Pythona (TTS) przez ZMQ i odtwarza je na głośnikach.

**Zalety:**
*   Audio nigdy się nie zatnie, nawet jak Python będzie "myślał" nad LLM.
*   Błyskawiczna reakcja na przerwanie mówienia (VAD).

#### B. Usługa "Mózg" (Python) - *Zostaw jak jest*
To obecny kod, ale odciążony od pętli audio:
1.  Odbiera gotowe pakiety "User Speech" od Go.
2.  Robi STT (Whisper).
3.  Pyta LLM.
4.  Robi TTS (XTTS/Piper).
5.  Wysyła audio do Go.

## Podsumowanie

Jeśli Twoim celem jest **maksymalna płynność i responsywność**, przepisanie samej warstwy audio/VAD na Go ma duży sens. Pozwoli to uniknąć problemów z GIL i zapewni stabilny strumień danych.

Przepisywanie logiki AI (STT/TTS/LLM) na Go **nie ma sensu** ekonomicznego ani wydajnościowego w tym projekcie.

### Decyzja
Czy chcesz, abym przygotował plan takiej migracji (wydzielenie warstwy audio do Go), czy na razie zostajemy przy optymalizacji Pythona?
