"""
core/tts.py — Offline text-to-speech using pyttsx3.

Uses a single background worker thread with a queue to prevent
overlapping audio and resource contention.
"""

import logging
import threading
from queue import Queue
import pyttsx3

logger = logging.getLogger(__name__)

_engine = pyttsx3.init()
_engine.setProperty("rate", 150)

_queue: Queue[str | None] = Queue()


def _worker():
    while True:
        text = _queue.get()
        if text is None:
            break
        try:
            _engine.say(text)
            _engine.runAndWait()
        finally:
            _queue.task_done()


_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()


def speak(text: str) -> None:
    """Enqueue *text* for speech. Non-blocking, thread-safe."""
    if not text or not text.strip():
        return
    logger.info("TTS enqueue: %s", text.strip())
    _queue.put(text.strip())


def shutdown() -> None:
    """Signal the worker to exit (call on app close)."""
    logger.info("TTS shutdown requested")
    _queue.put(None)
