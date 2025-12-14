#!/usr/bin/env python3
"""
Uruchamia interfejs BellaBot z WebSocket Bridge.
Synchronizuje stany systemu Watus z animowanym interfejsem robota.
"""

from bellabot_interface.run_interface import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
