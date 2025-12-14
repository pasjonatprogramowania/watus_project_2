#!/usr/bin/env python3
"""
WebSocket Bridge dla BellaBot Interface
Łączy ZMQ komunikację z watus_project_2 z WebSocket dla interfejsu BellaBot
"""

import asyncio
import json
import time
import zmq
import zmq.asyncio
from websockets.server import serve
from websockets.exceptions import ConnectionClosed
from typing import Set, Dict, Any

class BellabotWebSocketBridge:
    def __init__(self, 
                 zmq_pub_addr: str = "tcp://127.0.0.1:7780",
                 zmq_sub_addr: str = "tcp://127.0.0.1:7781",
                 ws_port: int = 8080):
        self.zmq_pub_addr = zmq_pub_addr
        self.zmq_sub_addr = zmq_sub_addr
        self.ws_port = ws_port

        # ZMQ Context
        self.zmq_context = zmq.asyncio.Context()
        self.publisher_socket = None
        self.subscriber_socket = None

        # WebSocket connections
        self.connected_clients: Set = set()

        # State tracking
        self.current_state = "neutral"
        self.last_state_change = time.time()

    async def setup_zmq(self):
        """Inicjalizuje gniazda ZMQ"""
        try:
            # Publisher socket - for sending commands to Watus
            self.publisher_socket = self.zmq_context.socket(zmq.PUB)
            self.publisher_socket.setsockopt(zmq.SNDHWM, 100)
            self.publisher_socket.bind(self.zmq_pub_addr)
            print(f"[Bridge] ZMQ PUB: {self.zmq_pub_addr}")

            # Subscriber socket - for receiving states from Watus
            self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
            self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "watus.state")
            self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
            self.subscriber_socket.connect(self.zmq_sub_addr)
            print(f"[Bridge] ZMQ SUB: {self.zmq_sub_addr}")

        except Exception as e:
            print(f"[Bridge] ZMQ setup error: {e}")

    async def zmq_subscriber_loop(self):
        """Pętla nasłuchująca ZMQ i publikująca do WebSocket"""
        while True:
            try:
                topic, message_payload = await self.subscriber_socket.recv_multipart()
                decoded_message = json.loads(message_payload.decode("utf-8"))

                if topic == b"watus.state":
                    await self.handle_watus_state(decoded_message)
                elif topic == b"dialog.leader":
                    await self.handle_dialog_leader(decoded_message)

            except Exception as e:
                print(f"[Bridge] ZMQ subscriber error: {e}")
                await asyncio.sleep(0.1)

    async def handle_watus_state(self, message: Dict[str, Any]):
        """Obsługuje zmiany stanu Watus i przekazuje do BellaBot"""
        state_name = message.get("state", "")
        timestamp = message.get("timestamp", time.time())

        print(f"[Bridge] Watus state: {state_name}")

        # Mapowanie stanów Watus na stany BellaBot
        bellabot_state_mapping = {
            "listening": "listening",
            "processing": "thinking", 
            "speaking": "excited",
            "idle": "neutral"
        }

        bellabot_state = bellabot_state_mapping.get(state_name, "neutral")

        # Aktualizuj lokalny stan
        if bellabot_state != self.current_state:
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
        print(f"[Bridge] Dialog leader message received")

        # Przekaż informacje o dialogu do interfejsu
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

        # Usuń rozłączonych klientów
        self.connected_clients -= disconnected_clients

    async def handle_client(self, websocket):
        """Obsługuje nowe połączenie WebSocket"""
        print(f"[Bridge] New client connected: {websocket.remote_address}")
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
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    print(f"[Bridge] Invalid JSON from client: {message}")
                except Exception as e:
                    print(f"[Bridge] Client message error: {e}")
        except ConnectionClosed:
            pass
        except Exception as e:
            print(f"[Bridge] Client handler error: {e}")
        finally:
            print(f"[Bridge] Client disconnected: {websocket.remote_address}")
            self.connected_clients.discard(websocket)

    async def handle_client_message(self, websocket, data: Dict[str, Any]):
        """Obsługuje wiadomości od klientów WebSocket"""
        message_type = data.get("type")

        if message_type == "state_change":
            # Klient chce zmienić stan BellaBot
            new_state = data.get("state")
            if new_state and self.publisher_socket:
                # Publikuj przez ZMQ dla innych systemów
                state_message = {
                    "state": new_state,
                    "timestamp": time.time(),
                    "source": "bellabot_interface"
                }
                await self.publisher_socket.send_multipart([
                    b"bellabot.state",
                    json.dumps(state_message, ensure_ascii=False).encode("utf-8")
                ])
                print(f"[Bridge] Published bellabot state: {new_state}")

        elif message_type == "ping":
            # Odpowiedz na ping
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }, ensure_ascii=False))

    async def start_server(self):
        """Uruchamia WebSocket server"""
        await self.setup_zmq()

        # Uruchom pętlę ZMQ w tle
        asyncio.create_task(self.zmq_subscriber_loop())

        # Uruchom WebSocket server
        print(f"[Bridge] Starting WebSocket server on port {self.ws_port}")
        async with serve(self.handle_client, "127.0.0.1", self.ws_port):
            print(f"[Bridge] Bellabot WebSocket Bridge ready!")
            print(f"[Bridge] WebSocket: ws://127.0.0.1:{self.ws_port}")
            print(f"[Bridge] ZMQ: {self.zmq_pub_addr} <-> {self.zmq_sub_addr}")

            await asyncio.Future()  # Uruchom forever

    def cleanup(self):
        """Sprzątanie zasobów"""
        try:
            if self.publisher_socket:
                self.publisher_socket.close()
            if self.subscriber_socket:
                self.subscriber_socket.close()
            self.zmq_context.term()
        except Exception as e:
            print(f"[Bridge] Cleanup error: {e}")

async def main():
    """Główna funkcja"""
    bridge = BellabotWebSocketBridge(
        zmq_pub_addr="tcp://127.0.0.1:7782",  # Bridge publishes on different port
        zmq_sub_addr="tcp://127.0.0.1:7780",  # Subscribe to Watus PUB port
        ws_port=8080
    )

    try:
        await bridge.start_server()
    except KeyboardInterrupt:
        print("\n[Bridge] Shutting down...")
    finally:
        bridge.cleanup()

if __name__ == "__main__":
    print("=== Bellabot WebSocket Bridge ===")
    print("Connecting ZMQ (watus_project_2) to WebSocket (BellaBot Interface)")
    print()
    asyncio.run(main())
