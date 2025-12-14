#!/usr/bin/env python3
"""
WebSocket Bridge dla BellaBot Interface
Łączy ZMQ komunikację z watus_project_2 z WebSocket dla interfejsu BellaBot
"""

import asyncio
import json
import time
import sys
import platform
from pathlib import Path

# Fix dla Windows - musi być PRZED importem zmq.asyncio
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dodaj ścieżkę do modułów watus_audio
sys.path.insert(0, str(Path(__file__).parent.parent))

import zmq
import zmq.asyncio
from websockets.server import serve
from websockets.exceptions import ConnectionClosed
from typing import Set, Dict, Any

# Import konfiguracji z watus_audio
from watus_audio import config

class BellabotWebSocketBridge:
    def __init__(self, ws_port: int = 8080):
        # Użyj portów z konfiguracji Watus
        self.zmq_watus_pub_addr = config.PUB_ADDR  # Port na którym Watus publikuje
        self.ws_port = ws_port

        # ZMQ Context
        self.zmq_context = zmq.asyncio.Context()
        self.subscriber_socket = None

        # WebSocket connections
        self.connected_clients: Set = set()

        # State tracking
        self.current_state = "neutral"
        self.last_state_change = time.time()

    async def setup_zmq(self):
        """Inicjalizuje gniazda ZMQ"""
        try:
            # Subscriber socket - for receiving states from Watus
            self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
            self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "watus.state")
            self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
            self.subscriber_socket.connect(self.zmq_watus_pub_addr)
            print(f"[Bridge] ZMQ SUB connected to: {self.zmq_watus_pub_addr}")
            print(f"[Bridge] Subscribed topics: watus.state, dialog.leader")

        except Exception as e:
            print(f"[Bridge] ZMQ setup error: {e}")
            raise

    async def zmq_subscriber_loop(self):
        """Pętla nasłuchująca ZMQ i publikująca do WebSocket"""
        print("[Bridge] ZMQ subscriber loop started, waiting for messages...")
        while True:
            try:
                topic, message_payload = await self.subscriber_socket.recv_multipart()
                decoded_message = json.loads(message_payload.decode("utf-8"))
                
                topic_str = topic.decode("utf-8")
                print(f"[Bridge] Received ZMQ: {topic_str} -> {decoded_message.get('state', '')}")

                if topic == b"watus.state":
                    await self.handle_watus_state(decoded_message)
                elif topic == b"dialog.leader":
                    await self.handle_dialog_leader(decoded_message)

            except Exception as e:
                print(f"[Bridge] ZMQ subscriber error: {e}")
                await asyncio.sleep(0.5)

    async def handle_watus_state(self, message: Dict[str, Any]):
        """Obsługuje zmiany stanu Watus i przekazuje do BellaBot"""
        state_name = message.get("state", "")
        timestamp = message.get("timestamp", time.time())

        # Mapowanie stanów Watus na stany BellaBot
        bellabot_state_mapping = {
            "listening": "listening",
            "processing": "thinking", 
            "speaking": "excited",
            "idle": "neutral"
        }

        bellabot_state = bellabot_state_mapping.get(state_name, "neutral")
        print(f"[Bridge] State: {state_name} -> {bellabot_state} (clients: {len(self.connected_clients)})")

        # Aktualizuj lokalny stan
        self.current_state = bellabot_state
        self.last_state_change = timestamp

        # Publikuj do wszystkich podłączonych klientów WebSocket
        await self.broadcast_to_clients({
            "type": "state",
            "state": bellabot_state,
            "timestamp": timestamp,
            "source": "watus"
        })

    async def handle_dialog_leader(self, message: Dict[str, Any]):
        """Obsługuje komunikaty lidera dialogu"""
        await self.broadcast_to_clients({
            "type": "dialog",
            "message": message,
            "timestamp": time.time()
        })

    async def broadcast_to_clients(self, data: Dict[str, Any]):
        """Wysyła wiadomość do wszystkich podłączonych klientów"""
        if not self.connected_clients:
            return

        message = json.dumps(data, ensure_ascii=False)
        disconnected_clients = set()

        for client in self.connected_clients:
            try:
                await client.send(message)
            except ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                print(f"[Bridge] WebSocket send error: {e}")
                disconnected_clients.add(client)

        self.connected_clients -= disconnected_clients

    async def handle_client(self, websocket):
        """Obsługuje nowe połączenie WebSocket"""
        print(f"[Bridge] Client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)

        try:
            # Wyślij aktualny stan do nowego klienta
            await websocket.send(json.dumps({
                "type": "state",
                "state": self.current_state,
                "timestamp": self.last_state_change,
                "source": "initial"
            }, ensure_ascii=False))

            # Nasłuchuj wiadomości od klienta
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": time.time()
                        }, ensure_ascii=False))
                except:
                    pass
        except ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)

    async def start_server(self):
        """Uruchamia WebSocket server"""
        await self.setup_zmq()

        # Uruchom pętlę ZMQ w tle
        asyncio.create_task(self.zmq_subscriber_loop())

        # Uruchom WebSocket server
        print(f"[Bridge] WebSocket: ws://127.0.0.1:{self.ws_port}")
        print(f"[Bridge] Listening for Watus on: {self.zmq_watus_pub_addr}")
        print("[Bridge] Ready!")
        print()
        
        async with serve(self.handle_client, "127.0.0.1", self.ws_port):
            await asyncio.Future()

    def cleanup(self):
        """Sprzątanie zasobów"""
        try:
            if self.subscriber_socket:
                self.subscriber_socket.close()
            self.zmq_context.term()
        except:
            pass

async def main():
    """Główna funkcja"""
    bridge = BellabotWebSocketBridge(ws_port=8080)
    try:
        await bridge.start_server()
    except KeyboardInterrupt:
        print("\n[Bridge] Shutting down...")
    finally:
        bridge.cleanup()

if __name__ == "__main__":
    print("=== Bellabot WebSocket Bridge ===")
    asyncio.run(main())
