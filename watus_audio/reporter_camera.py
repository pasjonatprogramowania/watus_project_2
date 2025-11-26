import time
import json
import threading
from collections import deque, Counter
from typing import Dict, Any
from . import config
from .common import log_message

_camera_lock = threading.Lock()
_camera_last_state: Dict[str, Any] = {"ts": 0.0, "record": None}
_camera_buffer: deque[Dict[str, Any]] = deque()

def _summarize_single_frame(camera_frame_object: Dict[str, Any]) -> str:
    """Tworzy krótki opis tekstowy pojedynczej klatki z kamery."""
    detected_items_list = camera_frame_object.get("objects") or camera_frame_object.get("detections") or []
    if not detected_items_list:
        brightness_value = camera_frame_object.get("brightness")
        brightness_string = f" | bright={brightness_value:.2f}" if isinstance(brightness_value, (int,float)) else ""
        return f"brak detekcji{brightness_string}".strip()
    
    detected_items_list = sorted(detected_items_list, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
    top_items = [f"{(it.get('name') or '?')}({int(round(100*float(it.get('conf',0.0))))}%)" for it in detected_items_list[:3]]
    num_items = len(detected_items_list)
    extra_items_str = f"+{num_items-3}" if num_items > 3 else ""
    
    brightness_value = camera_frame_object.get("brightness")
    brightness_string = f" | bright={brightness_value:.2f}" if isinstance(brightness_value, (int,float)) else ""
    
    return f"{', '.join(top_items)} {extra_items_str}{brightness_string}".strip()

def _summarize_time_window(frame_buffer: deque) -> Dict[str, Any]:
    """Tworzy podsumowanie statystyczne z bufora klatek (okno czasowe)."""
    # liczymy najczęstsze obiekty oraz średnią jasność
    object_counts = Counter()
    brightness_sum = 0.0
    brightness_count = 0
    last_timestamp = 0.0
    
    for camera_record in frame_buffer:
        items = camera_record.get("objects") or []
        for it in items:
            name = str(it.get("name") or "?")
            object_counts[name] += 1
        
        brightness_value = camera_record.get("brightness")
        if isinstance(brightness_value, (int, float)):
            brightness_sum += float(brightness_value)
            brightness_count += 1
        
        last_timestamp = max(last_timestamp, float(camera_record.get("ts") or 0.0))
    
    top_objects = [{"name": n, "count": c} for n, c in object_counts.most_common(3)]
    average_brightness = (brightness_sum/brightness_count) if brightness_count > 0 else None
    
    return {
        "since": config.CAM_WINDOW_SEC,
        "top_objects": top_objects,
        "avg_brightness": average_brightness,
        "last_ts": last_timestamp,
    }

def start_camera_tail_loop(file_path: str):
    """
    Uruchamia pętlę śledzenia pliku JSONL z danymi z kamery (tail).
    Aktualizuje globalny stan kamery.
    """
    if not file_path:
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.05); continue
                try:
                    camera_frame_object = json.loads(line.strip())
                    timestamp = float(camera_frame_object.get("ts") or time.time())
                    with _camera_lock:
                        _camera_last_state["ts"] = timestamp
                        _camera_last_state["record"] = camera_frame_object
                        _camera_buffer.append(camera_frame_object)
                        # trim do okna czasowego
                        cutoff_time = timestamp - config.CAM_WINDOW_SEC
                        while _camera_buffer and float(_camera_buffer[0].get("ts", 0.0)) < cutoff_time:
                            _camera_buffer.popleft()
                except Exception:
                    pass
    except Exception as e:
        log_message(f"[Reporter][CAM] tail err: {e}")

def get_current_camera_summary():
    """
    Zwraca aktualny stan kamery: ostatnią klatkę, jej opis i podsumowanie okna czasowego.
    """
    with _camera_lock:
        last_record = _camera_last_state.get("record")
        last_summary_text = _summarize_single_frame(last_record) if last_record else "brak danych"
        time_window_summary = _summarize_time_window(_camera_buffer) if _camera_buffer else None
    
    return last_record, last_summary_text, time_window_summary
