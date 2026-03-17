"""Cancellation support for long-running knowledge-build tasks."""

import threading

_cancel_events: dict[int, threading.Event] = {}
_lock = threading.Lock()


class CancelledError(Exception):
    """Raised when a knowledge-build task is cancelled by the user."""


def get_cancel_event(kb_id: int) -> threading.Event:
    """Get or create a cancellation event for a knowledge base build."""
    with _lock:
        if kb_id not in _cancel_events:
            _cancel_events[kb_id] = threading.Event()
        return _cancel_events[kb_id]


def clear_cancel_event(kb_id: int) -> None:
    """Remove the cancellation event after a task finishes."""
    with _lock:
        _cancel_events.pop(kb_id, None)


def check_cancelled(kb_id: int) -> None:
    """Raise ``CancelledError`` if the build for *kb_id* was cancelled."""
    with _lock:
        ev = _cancel_events.get(kb_id)
    if ev and ev.is_set():
        raise CancelledError(f"知识库 kb_id={kb_id} 的构建任务已被用户取消")
