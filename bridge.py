"""
bridge.py — Connects vision/main.py state_queue to FastAPI server via HTTP POST.
Run this alongside vision/main.py when the server is separate.
"""

import asyncio
import aiohttp
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)
API_INGEST = "http://localhost:8000/api/ingest"


async def bridge_loop():
    """Reads from vision state_queue and POSTs to the API server."""
    from vision.main import state_queue, run as vision_run
    import threading

    # Start vision engine in a daemon thread
    t = threading.Thread(
        target=vision_run,
        kwargs={"camera_idx": 0, "sensitivity": "medium", "no_display": False},
        daemon=True
    )
    t.start()
    logger.info("Vision engine thread started.")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                state = await asyncio.wait_for(state_queue.get(), timeout=2.0)
                try:
                    async with session.post(API_INGEST, json=state, timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status != 200:
                            logger.warning(f"Ingest returned {r.status}")
                except aiohttp.ClientError as e:
                    logger.warning(f"API unreachable: {e}")
            except asyncio.TimeoutError:
                pass  # No new state — continue
            except Exception as e:
                logger.error(f"Bridge error: {e}")
                await asyncio.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(bridge_loop())
