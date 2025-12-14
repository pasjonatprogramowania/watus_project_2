#!/usr/bin/env python3
"""
Skrypt uruchamiający interfejs BellaBot z WebSocket Bridge
Uruchamia:
1. WebSocket Bridge (ZMQ <-> WebSocket)
2. Prosty serwer HTTP dla interfejsu HTML
3. Otwiera przeglądarkę z interfejsem
"""

import asyncio
import os
import sys
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

# Dodaj ścieżkę do modułów
sys.path.insert(0, str(Path(__file__).parent.parent))

from bellabot_interface.bellabot_websocket_bridge import BellabotWebSocketBridge


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Handler bez logów do konsoli"""
    def log_message(self, format, *args):
        pass  # Suppress logging


def run_http_server(port: int, directory: str):
    """Uruchamia prosty serwer HTTP w tle"""
    os.chdir(directory)
    handler = QuietHTTPHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"[HTTP] Serving at http://127.0.0.1:{port}")
        httpd.serve_forever()


async def main():
    """Główna funkcja"""
    http_port = 8081
    ws_port = 8080
    interface_dir = Path(__file__).parent
    
    print("=" * 50)
    print("  Watus BellaBot Interface")
    print("=" * 50)
    print()
    
    # Uruchom serwer HTTP w osobnym wątku
    http_thread = threading.Thread(
        target=run_http_server,
        args=(http_port, str(interface_dir)),
        daemon=True
    )
    http_thread.start()
    print(f"[HTTP] Interface dir: {interface_dir}")
    
    # Otwórz przeglądarkę
    interface_url = f"http://127.0.0.1:{http_port}/bellabot_interface.html"
    print(f"[Browser] Opening: {interface_url}")
    webbrowser.open(interface_url)
    
    # Uruchom WebSocket Bridge (używa config.PUB_ADDR automatycznie)
    bridge = BellabotWebSocketBridge(ws_port=ws_port)
    
    try:
        await bridge.start_server()
    except KeyboardInterrupt:
        print("\n[Interface] Shutting down...")
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    print()
    print("Starting Watus BellaBot Interface...")
    print("Press Ctrl+C to stop")
    print()
    asyncio.run(main())
