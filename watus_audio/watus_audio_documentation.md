# Dokumentacja Modułu `watus_audio`

## 1. Przegląd Systemu (System Overview)

`watus_audio` to zaawansowany moduł asystenta głosowego, który integruje rozpoznawanie mowy (STT), syntezę mowy (TTS), weryfikację mówcy oraz analizę obrazu z kamery. System jest zaprojektowany jako architektura wieloprocesowa, komunikująca się za pomocą lekkiej szyny komunikatów ZeroMQ (ZMQ).

### Główny Workflow

1.  **Nasłuchiwanie (Watus Process)**:
    *   Mikrofon zbiera dźwięk w pętli.
    *   **VAD (Voice Activity Detection)** wykrywa segmenty mowy.
    *   Po wykryciu mowy, segment jest transkrybowany na tekst przez model **Whisper**.
    *   **Weryfikator Mówcy** sprawdza, czy głos należy do "Lidera" (autoryzowanego użytkownika) lub czy padło słowo wybudzające (np. "Hej Watusiu").
    *   Jeśli wypowiedź jest zaakceptowana, jest publikowana na szynę ZMQ (kanał `dialog.leader`).

2.  **Przetwarzanie i Logika (Reporter Process)**:
    *   Proces Reportera nasłuchuje na kanale `dialog.leader`.
    *   Pobiera aktualny kontekst wizualny z kamery (analiza obrazu z pliku JSONL).
    *   Buduje "Meldunek" zawierający: tekst użytkownika, opis tego co widzi kamera, historię dialogu i metadane.
    *   Wysyła zapytanie do zewnętrznego modelu **LLM** (Large Language Model) przez HTTP.
    *   Odbiera odpowiedź tekstową od LLM.
    *   Publikuje odpowiedź na szynę ZMQ (kanał `tts.speak`).

3.  **Odpowiedź Głosowa (Watus Process)**:
    *   Proces Watus nasłuchuje na kanale `tts.speak`.
    *   Odbiera tekst do wypowiedzenia.
    *   Syntezuje mowę używając wybranego silnika TTS (Piper, Gemini lub XTTS).
    *   Odtwarza dźwięk przez głośniki.

---

## 2. Architektura i Komunikacja

System składa się z dwóch głównych punktów wejścia (entry points), które zazwyczaj działają jako osobne procesy:

1.  **`watus_main.py`** (Proces Audio/Interfejsu): Odpowiada za "uszy" i "usta" systemu.
2.  **`reporter_main.py`** (Proces Logiki/Mózgu): Odpowiada za "mózg" i integrację zmysłów (wzrok + tekst).

Komunikacja odbywa się przez **ZeroMQ (PUB/SUB)**:
*   **PUB `dialog.leader`**: Watus -> Reporter (Wypowiedź użytkownika)
*   **PUB `tts.speak`**: Reporter -> Watus (Tekst do wypowiedzenia przez system)
*   **PUB `watus.state`**: Watus -> UI/Inne (Stan systemu: listening, processing, speaking)

---

## 3. Szczegółowa Analiza Plików

Poniżej znajduje się dokładny opis każdego pliku i funkcji w module `watus_audio`.

### 3.1. Pliki Rdzenia (Core)

#### `config.py`
**Cel**: Centralna konfiguracja systemu. Ładuje zmienne środowiskowe z pliku `.env` i definiuje stałe.
*   **Kluczowe zmienne**:
    *   `TTS_PROVIDER`: Wybór silnika TTS (gemini, piper, xtts).
    *   `STT_PROVIDER`: Wybór silnika STT (local - Whisper).
    *   `WHISPER_MODEL_NAME`: Ścieżka lub nazwa modelu Whisper.
    *   `WAKE_WORDS`: Lista słów wybudzających.
    *   `PUB_ADDR`, `SUB_ADDR`: Adresy portów ZMQ.
    *   `LLM_HTTP_URL`: Adres API do LLM.

#### `common.py`
**Cel**: Funkcje pomocnicze używane w całym projekcie.
*   **Funkcje**:
    *   `log_message(message)`: Wypisuje logi na konsolę z wymuszonym flushowaniem (ważne przy pracy jako serwis/docker).
    *   `append_line_to_dialog_history(dialog_object, file_path)`: Dopisuje obiekt dialogu do pliku JSONL (historia rozmowy).
    *   `write_object_to_jsonl_file(file_path, data_object)`: Uniwersalna funkcja do zapisu danych do plików JSONL (np. meldunki, odpowiedzi).

#### `state.py`
**Cel**: Zarządzanie stanem procesu Watus (synchronizacja wątków).
*   **Klasa `SystemState`**:
    *   `__init__`: Inicjalizuje flagi (`_tts_active_flag`, `_awaiting_reply_flag`) i blokady.
    *   `set_tts_active_flag(is_active)`: Ustawia, czy system aktualnie mówi.
    *   `set_awaiting_reply_flag(is_awaiting)`: Ustawia, czy system czeka na odpowiedź od LLM.
    *   `block_input_until_reply_received()`: Ustawia timer blokujący mikrofon na krótki czas oczekiwania.
    *   `is_input_blocked()`: Zwraca `True`, jeśli mikrofon powinien być wyłączony (bo system mówi lub myśli).

#### `bus.py`
**Cel**: Obsługa komunikacji ZeroMQ.
*   **Klasa `ZMQMessageBus`**:
    *   `__init__`: Tworzy gniazda PUB i SUB. Uruchamia wątek nasłuchujący (`_subscriber_loop`).
    *   `publish_leader_utterance(message_payload)`: Wysyła wiadomość o wypowiedzi użytkownika (`dialog.leader`).
    *   `publish_system_state(state_name, data)`: Wysyła informację o stanie systemu (`watus.state`).
    *   `_subscriber_loop()`: Wątek w tle odbierający wiadomości `tts.speak` i wrzucający je do kolejki.
    *   `get_next_tts_message(timeout)`: Pobiera wiadomość z kolejki dla wątku TTS.
    *   `_cleanup_resources()`: Bezpiecznie zamyka gniazda przy wyjściu.

#### `__init__.py`
**Cel**: Plik inicjalizujący pakiet, eksportuje kluczowe moduły dla łatwiejszego importu.

### 3.2. Moduły Funkcjonalne (Audio & AI)

#### `audio_utils.py`
**Cel**: Narzędzia do obsługi urządzeń audio (mikrofon/głośniki).
*   **Funkcje**:
    *   `print_available_audio_devices()`: Wypisuje listę dostępnych urządzeń audio.
    *   `get_default_input_device_index()`: Wybiera mikrofon na podstawie konfiguracji lub domyślnych ustawień systemu.
    *   `get_default_output_device_index()`: Wybiera głośniki.
    *   `_find_device_index_by_name_fragment(...)`: Pomocnicze wyszukiwanie urządzenia po nazwie.

#### `stt.py`
**Cel**: Silnik rozpoznawania mowy (Speech-to-Text) i pętla nagrywania.
*   **Funkcja `check_if_text_contains_wake_word(text)`**: Sprawdza, czy w tekście jest "Hej Watusiu" itp.
*   **Klasa `SpeechToTextProcessingEngine`**:
    *   `__init__`: Ładuje model Whisper i VAD.
    *   `_initialize_local_whisper_model()`: Inicjalizuje `FasterWhisper`.
    *   `_vad_is_speech(frame_bytes)`: Używa `webrtcvad` do sprawdzenia czy ramka audio to mowa.
    *   `_transcribe_audio_segment(audio_samples)`: Zamienia nagrane audio na tekst.
    *   `start_listening_loop(input_device_index)`: **Główna pętla**. Czyta audio z mikrofonu, buforuje ramki, wykrywa początek i koniec mowy (VAD + analiza głośności dBFS). Gdy wykryje koniec wypowiedzi, wywołuje `_process_recorded_speech_segment`.
    *   `_process_recorded_speech_segment(...)`:
        1.  Transkrybuje audio.
        2.  Weryfikuje mówcę (czy to Lider?).
        3.  Jeśli Lider lub wykryto Wake Word -> publikuje wiadomość na ZMQ (`dialog.leader`) i blokuje nasłuchiwanie (czeka na odpowiedź).

#### `tts.py`
**Cel**: Silnik syntezy mowy (Text-to-Speech).
*   **Funkcje**:
    *   `synthesize_speech_and_play(text, device_index)`: Główna funkcja, wybiera odpowiedni backend (Piper, Gemini, XTTS) na podstawie configu.
    *   `synthesize_speech_piper(...)`: Obsługa Piper TTS (lokalny, szybki). Może używać Python API lub binarnego pliku wykonywalnego.
    *   `synthesize_speech_gemini(...)`: Obsługa Google Gemini TTS (chmura, wysoka jakość). Generuje audio strumieniowo.
    *   `synthesize_speech_xtts(...)`: Obsługa Coqui XTTS (lokalny, klonowanie głosu). Wymaga GPU dla dobrej wydajności.
    *   `_add_wav_header(...)`: Dodaje nagłówek WAV do surowych danych audio (dla Gemini).

#### `verifier.py`
**Cel**: Weryfikacja tożsamości mówcy (Speaker Verification).
*   **Klasa `_SbVerifier` (SpeechBrain)**:
    *   Używa modelu `speechbrain/spkrec-ecapa-voxceleb`.
    *   `enroll_voice_samples(...)`: Zapisuje wzorzec głosu lidera (embedding). Dzieje się to automatycznie po wykryciu Wake Word.
    *   `verify_speaker_identity(...)`: Porównuje nowy fragment mowy z zapisanym wzorcem. Zwraca `score` (podobieństwo) i decyzję `is_leader`.
    *   Obsługuje mechanizm "sticky": jeśli raz zostaniesz rozpoznany, system jest bardziej wyrozumiały przez pewien czas (`SPEAKER_STICKY_SEC`).

#### `led.py`
**Cel**: Sterowanie wskaźnikami wizualnymi (LED).
*   Obecnie zawiera klasę `LEDStatusController` z pustymi metodami (dummy), gotową do implementacji sterowania diodami (np. na Raspberry Pi).

### 3.3. Moduły Reportera (Logika & Integracja)

#### `reporter_camera.py`
**Cel**: Obsługa danych wizualnych.
*   **Działanie**: Nie łączy się bezpośrednio z kamerą. Zamiast tego śledzi (tail) plik JSONL (`camera.jsonl`), do którego inny proces (np. `camera_runner.py`) zapisuje wyniki detekcji obiektów.
*   **Funkcje**:
    *   `start_camera_tail_loop(file_path)`: Wątek czytający nowe linie z pliku JSONL w czasie rzeczywistym.
    *   `_summarize_single_frame(...)`: Tworzy krótki opis tekstowy klatki (np. "osoba(90%), kubek(50%)").
    *   `_summarize_time_window(...)`: Agreguje statystyki z ostatnich kilku sekund (np. najczęściej widoczne obiekty).
    *   `get_current_camera_summary()`: Zwraca kompletny obraz sytuacji wizualnej dla LLM.

#### `reporter_llm.py`
**Cel**: Komunikacja z modelem językowym.
*   **Funkcje**:
    *   `send_query_to_llm(content_text)`: Wysyła zapytanie POST do skonfigurowanego adresu URL.
    *   Obsługuje błędy sieciowe, timeouty i loguje czasy odpowiedzi.
    *   Zapisuje pary zapytanie-odpowiedź do pliku `responses.jsonl`.

#### `reporter_main.py`
**Cel**: Główny proces logiki biznesowej.
*   **Działanie**:
    *   Inicjalizuje ZMQ (SUB `dialog.leader`, PUB `tts.speak`).
    *   Uruchamia wątek kamery (`start_camera_tail_loop`).
    *   Uruchamia serwer FastAPI (dla health checków).
    *   **Pętla Główna (`start_main_loop`)**:
        1.  Odbiera wiadomość `dialog.leader`.
        2.  Buduje **Meldunek** (`build_report_payload`): łączy tekst użytkownika z opisem wizualnym z kamery i metadanymi.
        3.  Wysyła meldunek do LLM.
        4.  Odbiera odpowiedź i wysyła ją do Watusa (`tts.speak`).
        5.  Jeśli LLM nie odpowiada, wysyła komunikat błędu głosowego.

### 3.4. Entry Points

#### `watus_main.py`
**Cel**: Punkt wejścia dla procesu Audio.
*   **Działanie**:
    1.  Inicjalizuje konfigurację i urządzenia audio.
    2.  Tworzy `SystemState` i `ZMQMessageBus`.
    3.  Uruchamia wątek `tts_worker_thread` (do mówienia).
    4.  Inicjalizuje `SpeechToTextProcessingEngine`.
    5.  Uruchamia główną pętlę nasłuchiwania `stt_engine.start_listening_loop`.

---

## 4. Podsumowanie Przepływu Danych

1.  **Użytkownik mówi**: "Co widzisz?"
2.  **Watus (STT)**:
    *   Nagrywa audio -> Whisper -> Tekst: "Co widzisz?"
    *   Weryfikuje głos -> OK.
    *   Wysyła ZMQ `dialog.leader`: `{"text": "Co widzisz?", "is_leader": true, ...}`
3.  **Reporter**:
    *   Odbiera ZMQ.
    *   Pobiera stan kamery: "osoba, laptop".
    *   Tworzy prompt: `[VISION: osoba, laptop] USER: Co widzisz?`
    *   Wysyła do LLM.
4.  **LLM**:
    *   Generuje odpowiedź: "Widzę osobę siedzącą przed laptopem."
5.  **Reporter**:
    *   Wysyła ZMQ `tts.speak`: `{"text": "Widzę osobę siedzącą przed laptopem."}`
6.  **Watus (TTS)**:
    *   Odbiera ZMQ.
    *   Blokuje mikrofon.
    *   Generuje audio (TTS) -> Odtwarza dźwięk.
    *   Odblokowuje mikrofon.
