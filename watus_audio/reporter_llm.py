import time
import json
import re
import requests
from typing import Optional, Tuple
from . import config
from .common import log_message, write_object_to_jsonl_file

_RETRY_IN_REGEX = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_retry_hint_from_error(error_response_body: str) -> Optional[float]:
    """Parsuje sugestię czasu oczekiwania (retry-after) z treści błędu."""
    if not error_response_body: return None
    if ("RESOURCE_EXHAUSTED" in error_response_body or " 429 " in error_response_body or "\"code\": 429" in error_response_body):
        match = _RETRY_IN_REGEX.search(error_response_body)
        if match:
            try: return max(1.0, min(float(match.group(1)), 60.0))
            except Exception: return 5.0
        return 5.0
    return None

def send_query_to_llm(content_text: str) -> Tuple[Optional[str], Optional[int], Optional[str], float]:
    """
    Wysyła zapytanie do serwera LLM (HTTP POST).
    
    Argumenty:
        content_text (str): Treść zapytania (tekst).
        
    Zwraca:
        Tuple: (odpowiedź, kod_statusu, błąd, opóźnienie_ms)
    """
    if not config.LLM_HTTP_URL:
        return None, None, "LLM_HTTP_URL is empty", 0.0

    request_body = json.dumps({"content": content_text}, ensure_ascii=True).encode("utf-8")
    request_headers = {"Content-Type": "application/json; charset=utf-8"}

    request_start_time = time.time()
    try:
        log_message(f"[Reporter][HTTP→] POST {config.LLM_HTTP_URL} len={len(content_text)}")
        http_response = requests.post(config.LLM_HTTP_URL, data=request_body, headers=request_headers, timeout=config.HTTP_TIMEOUT)
        latency_milliseconds = (time.time() - request_start_time) * 1000.0
        log_message(f"[Reporter][HTTP←] {http_response.status_code} ({latency_milliseconds:.1f} ms)")

        if 200 <= http_response.status_code < 300:
            try:
                response_data = http_response.json()
            except Exception:
                response_data = {"raw_text": http_response.text}
            llm_answer_text = (response_data.get("answer") or response_data.get("msg") or response_data.get("text") or "").strip()
            write_object_to_jsonl_file(config.RESP_FILE, {
                "ts": time.time(),
                "request": content_text,
                "raw_response": response_data,
                "answer": llm_answer_text,
                "latency_ms": latency_milliseconds
            })
            return (llm_answer_text if llm_answer_text else json.dumps(response_data, ensure_ascii=False)), http_response.status_code, None, latency_milliseconds

        error_response_body = http_response.text
        log_message(f"[Reporter][HTTP!] status={http_response.status_code} body={error_response_body[:400]}")
        return None, http_response.status_code, error_response_body, latency_milliseconds

    except requests.Timeout as e:
        return None, 408, f"timeout: {e}", (time.time() - request_start_time) * 1000.0
    except requests.RequestException as e:
        return None, None, f"request_exception: {e}", (time.time() - request_start_time) * 1000.0
