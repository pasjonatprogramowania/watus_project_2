import time
import json
import threading
import zmq
import uvicorn
from fastapi import FastAPI
from typing import Set, Dict, Any
from . import config
from .common import log_message, write_object_to_jsonl_file
from .reporter_camera import start_camera_tail_loop, get_current_camera_summary, _camera_last_state
from .reporter_llm import send_query_to_llm, parse_retry_hint_from_error

# ===== ZMQ =====
zmq_context = zmq.Context.instance()
subscriber_socket = None
publisher_socket = None

def setup_zmq_sockets():
    """Inicjalizuje gniazda ZMQ dla Reportera."""
    global subscriber_socket, publisher_socket
    subscriber_socket = zmq_context.socket(zmq.SUB)
    subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
    subscriber_socket.connect(config.PUB_ADDR)

    publisher_socket = zmq_context.socket(zmq.PUB)
    publisher_socket.setsockopt(zmq.SNDHWM, 100)
    publisher_socket.setsockopt(zmq.LINGER, 0)
    publisher_socket.bind(config.SUB_ADDR)

# ===== HTTP API =====
app = FastAPI()

# ===== Scenariusz (watch) =====
_scenario_lock = threading.Lock()
_active_scenario_id = "default"

def _read_active_scenario_from_file(path: str) -> str:
    """Czyta ID aktywnego scenariusza z pliku."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            last_line = None
            for line in f:
                last_line = line
            if not last_line:
                return _active_scenario_id
            scenario_object = json.loads(last_line.strip())
            scenario_identifier = str(scenario_object.get("id") or "").strip()
            return scenario_identifier if scenario_identifier else _active_scenario_id
    except Exception:
        return _active_scenario_id

def start_scenario_watch_loop(path: str, poll_interval_seconds: float = 1.0):
    """
    Monitoruje plik scenariusza i aktualizuje stan w razie zmian.
    """
    global _active_scenario_id
    previous_scenario_id = None
    while True:
        current_scenario_id = _read_active_scenario_from_file(path)
        if current_scenario_id and current_scenario_id != previous_scenario_id:
            with _scenario_lock:
                _active_scenario_id = current_scenario_id
            log_message(f"[Reporter] Aktywny scenariusz → {current_scenario_id}")
            previous_scenario_id = current_scenario_id
        time.sleep(poll_interval_seconds)

def get_active_scenario_id() -> str:
    """Zwraca ID aktualnie aktywnego scenariusza."""
    with _scenario_lock:
        return _active_scenario_id

@app.on_event("startup")
def _on_startup():
    log_message(f"[Reporter] SUB dialog.leader  @ {config.PUB_ADDR}")
    log_message(f"[Reporter] PUB tts.speak      @ {config.SUB_ADDR}")
    log_message(f"[Reporter] LLM_HTTP_URL       = {config.LLM_HTTP_URL or '(BRAK)'}  timeout={config.HTTP_TIMEOUT:.1f}s")
    log_message(f"[Reporter] CAMERA_JSONL       = {config.CAMERA_JSONL or '(OFF)'}")
    log_message(f"[Reporter] SCENARIO_ACTIVE    = {config.SCENARIO_ACTIVE_PATH}")

@app.get("/health")
def health_check_endpoint():
    """Endpoint sprawdzający stan zdrowia usługi."""
    return {
        "ok": True,
        "ts": time.time(),
        "llm_url": config.LLM_HTTP_URL,
        "scenario": get_active_scenario_id(),
        "camera_tail_active": _camera_last_state.get("ts", 0.0) > 0.0
    }

# ===== Duplikaty =====
_seen_turn_ids: Set[int] = set()
_SEEN_LIMIT = 10000

def _is_duplicate_turn(dialog_turn_identifiers) -> bool:
    """Sprawdza, czy dana tura była już przetwarzana (deduplikacja)."""
    if not dialog_turn_identifiers: return False
    try: turn_id = int(dialog_turn_identifiers[0])
    except Exception: return False
    if turn_id in _seen_turn_ids: return True
    _seen_turn_ids.add(turn_id)
    if len(_seen_turn_ids) > _SEEN_LIMIT:
        for _ in range(len(_seen_turn_ids)//2):
            try: _seen_turn_ids.pop()
            except KeyError: break
    return False

# ===== Meldunek =====
def build_report_payload(received_message_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buduje pełny obiekt meldunku (raportu) zawierający dane audio, wideo i kontekst.
    """
    question_text = (received_message_payload.get("text_full") or "").strip()
    session_id = received_message_payload.get("session_id")
    group_id   = received_message_payload.get("group_id")
    ts_start   = float(received_message_payload.get("ts_start") or 0.0)
    ts_end     = float(received_message_payload.get("ts_end") or 0.0)
    dbfs       = received_message_payload.get("dbfs")
    verify_data = received_message_payload.get("verify") or {}
    current_time = time.time()
    scenario_id = get_active_scenario_id()

    # kamera – ostatnia klatka + okno
    last_record, last_summary_text, time_window_summary = get_current_camera_summary()

    # krótkie stringi w opisie (zostawiamy szybki POST tylko ze stringiem)
    camera_last_string = f"LAST={last_summary_text}" if last_record else "LAST=none"
    if time_window_summary:
        top_objects_str = ", ".join([f"{t['name']}×{t['count']}" for t in (time_window_summary.get('top_objects') or [])]) or "none"
        if isinstance(time_window_summary.get("avg_brightness"), (int,float)):
            camera_window_string = f"WIN{int(config.CAM_WINDOW_SEC*1000)}ms: {top_objects_str} | avg_bri={time_window_summary['avg_brightness']:.2f}"
        else:
            camera_window_string = f"WIN{int(config.CAM_WINDOW_SEC*1000)}ms: {top_objects_str}"
    else:
        camera_window_string = "WIN=none"

    description_string = (
        f"[SYS_TIME={current_time:.3f}] [SCENARIO={scenario_id}] [CAMERA={config.CAMERA_NAME}] "
        f"[SESSION={session_id}] [GROUP={group_id}] "
        f"[SPEECH={ts_start:.3f}-{ts_end:.3f}s ~{dbfs:.1f}dBFS] "
        f"[LEADER_SCORE={verify_data.get('score')}] "
        f"[VISION {camera_last_string} | {camera_window_string}] "
        f"USER: {question_text}"
    )

    # pełny meldunek do logów/debug (do JSONL) – nie wysyłamy go do LLM (zachowujemy kompatybilność)
    report_payload_object = {
        "ts_system": current_time,
        "scenario": scenario_id,
        "camera": {"name": config.CAMERA_NAME, "jsonl_path": config.CAMERA_JSONL or None},
        "question_text": question_text,
        "opis": description_string,
        "dialog_meta": {
            "session_id": session_id,
            "group_id": group_id,
            "turn_ids": received_message_payload.get("turn_ids"),
            "ts_start": ts_start,
            "ts_end": ts_end,
            "dbfs": dbfs,
            "verify": verify_data,
        },
        "vision": {
            "last": last_record,
            "window_summary": time_window_summary
        }
    }
    return report_payload_object

def print_report_summary(report_payload_object: Dict[str, Any]):
    """Wypisuje skrót raportu na konsolę."""
    print(
        "\n[Reporter][MELDUNEK]"
        f"\n- ts_system : {report_payload_object['ts_system']:.3f}"
        f"\n- scenariusz: {report_payload_object['scenario']}"
        f"\n- kamera    : {report_payload_object['camera']['name']}"
        f"\n- opis→LLM  : {report_payload_object['opis']}\n",
        flush=True
    )

# ===== Pętla główna =====
def start_main_loop():
    """Główna pętla przetwarzania wiadomości ZMQ."""
    time.sleep(0.2)
    while True:
        try:
            topic, payload = subscriber_socket.recv_multipart()
            if topic != b"dialog.leader":
                continue

            try:
                received_message_payload = json.loads(payload.decode("utf-8"))
            except Exception:
                continue

            dialog_turn_identifiers = received_message_payload.get("turn_ids") or []
            group_id = received_message_payload.get("group_id")

            if _is_duplicate_turn(dialog_turn_identifiers):
                log_message(f"[Reporter][RECV] dup turn_id={dialog_turn_identifiers[0]} – pomijam")
                continue

            report_payload_object = build_report_payload(received_message_payload)
            print_report_summary(report_payload_object)
            write_object_to_jsonl_file(config.MELD_FILE, report_payload_object)

            content_text = report_payload_object["opis"]  # tylko string – kompatybilność i szybkość

            # 1. próba
            llm_answer_text, status_code, error_text, latency_ms = send_query_to_llm(content_text)

            # ewentualny retry (rzadko potrzebny – POST i tak jest szybki)
            wait_hint = parse_retry_hint_from_error(error_text or "") if not llm_answer_text and status_code == 500 else None
            retried = False
            if not llm_answer_text:
                if wait_hint is not None:
                    log_message(f"[Reporter][HTTP] backend 500/429 – czekam {wait_hint:.1f}s i retry")
                    time.sleep(wait_hint); retried = True
                    llm_answer_text, status_code, error_text, _ = send_query_to_llm(content_text)
                elif status_code in {408, 429, 502, 503, 504, None}:
                    import random
                    backoff = 0.40 + random.random() * 0.50
                    log_message(f"[Reporter][HTTP] retryable status={status_code} – backoff {backoff:.2f}s")
                    time.sleep(backoff); retried = True
                    llm_answer_text, status_code, error_text, _ = send_query_to_llm(content_text)

            if not llm_answer_text:
                msg_text = "Przepraszam, serwer odpowiedzi jest chwilowo przeciążony. Spróbuj proszę za moment."
                log_message(f"[Reporter][DROP] brak odpowiedzi po {'retry' if retried else '1. próbie'} (status={status_code})")
                publisher_socket.send_multipart([b"tts.speak", json.dumps(
                    {"text": msg_text, "reply_to": group_id, "turn_ids": dialog_turn_identifiers},
                    ensure_ascii=False).encode("utf-8")])
                continue

            log_message(f"[Reporter][LLM] answer len={len(llm_answer_text)}")
            publisher_socket.send_multipart([b"tts.speak", json.dumps(
                {"text": llm_answer_text, "reply_to": group_id, "turn_ids": dialog_turn_identifiers},
                ensure_ascii=False).encode("utf-8")])
            log_message(f"[Reporter][PUB] tts.speak → reply_to={group_id} len={len(llm_answer_text)}")

        except Exception as e:
            log_message(f"[Reporter] loop exception: {e}")
            time.sleep(0.15)

def main():
    """Funkcja startowa procesu Reporter."""
    setup_zmq_sockets()
    if config.CAMERA_JSONL:
        threading.Thread(target=start_camera_tail_loop, args=(config.CAMERA_JSONL,), daemon=True).start()
    if config.SCENARIO_ACTIVE_PATH:
        threading.Thread(target=start_scenario_watch_loop, args=(config.SCENARIO_ACTIVE_PATH,), daemon=True).start()
    thr = threading.Thread(target=start_main_loop, daemon=True)
    thr.start()
    uvicorn.run(app, host="127.0.0.1", port=8781, log_level="info")
