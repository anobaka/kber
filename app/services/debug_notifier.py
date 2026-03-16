"""Debug notification helper – sends progress messages to debug-enabled chats."""

import logging
import threading
from typing import Any, Callable

from sqlalchemy import select

from app.db.models import ChatKbBinding, ChatRepoBinding, ChatSettings
from app.db.session import get_session

logger = logging.getLogger(__name__)

# Lazy reference to avoid circular imports; set by bot startup.
_send_fn: Callable[[str, str], str | None] | None = None
_update_fn: Callable[[str, str], bool] | None = None
_send_progress_fn: Callable[[str, str], str | None] | None = None

# Per-chat last progress message_id for in-place updates.
# Key: (context_type, context_id, chat_id) → message_id
_progress_msg_ids: dict[tuple[str, Any, str], str] = {}
_progress_lock = threading.Lock()


def set_send_fn(fn: Callable[[str, str], str | None]) -> None:
    """Register the bot's send_message function (called once at startup)."""
    global _send_fn
    _send_fn = fn


def set_update_fn(fn: Callable[[str, str], bool]) -> None:
    """Register the bot's update_message function (called once at startup)."""
    global _update_fn
    _update_fn = fn


def set_send_progress_fn(fn: Callable[[str, str], str | None]) -> None:
    """Register the bot's send_progress_card function (called once at startup)."""
    global _send_progress_fn
    _send_progress_fn = fn


def _send(chat_id: str, message: str) -> str | None:
    if _send_fn is None:
        return None
    try:
        return _send_fn(chat_id, f"🔧 {message}")
    except Exception:
        logger.debug("Debug notify failed for %s", chat_id, exc_info=True)
        return None


def _send_or_update(
    chat_id: str,
    message: str,
    context_type: str,
    context_id: Any,
) -> None:
    """Send a progress card or update the previous one in-place.

    Feishu only supports editing card (interactive) messages, so progress
    notifications are sent as cards via *_send_progress_fn*.
    """
    key = (context_type, context_id, chat_id)
    full_msg = f"🔧 {message}"

    with _progress_lock:
        prev_id = _progress_msg_ids.get(key)

    # Try updating the previous card message first
    if prev_id and _update_fn:
        try:
            if _update_fn(prev_id, full_msg):
                return
        except Exception:
            logger.debug("Update failed, will send new message", exc_info=True)

    # Fallback: send a new progress card (must be card type for future edits)
    send = _send_progress_fn or _send_fn
    if send is None:
        return
    try:
        msg_id = send(chat_id, full_msg)
        if msg_id:
            with _progress_lock:
                _progress_msg_ids[key] = msg_id
    except Exception:
        logger.debug("Debug notify failed for %s", chat_id, exc_info=True)


def clear_progress_msg(context_type: str, context_id: Any) -> None:
    """Clear tracked progress message IDs for a context (e.g. after completion)."""
    with _progress_lock:
        keys_to_remove = [
            k for k in _progress_msg_ids if k[0] == context_type and k[1] == context_id
        ]
        for k in keys_to_remove:
            del _progress_msg_ids[k]


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------

def is_debug(chat_id: str) -> bool:
    """Check if a specific chat has debug mode enabled."""
    with get_session() as session:
        row = session.execute(
            select(ChatSettings.debug_mode).where(ChatSettings.chat_id == chat_id)
        ).scalar_one_or_none()
        return bool(row)


def get_debug_chat_ids_for_kb(kb_id: int) -> list[str]:
    """Return chat_ids that are bound to this KB **and** have debug mode on."""
    with get_session() as session:
        chat_ids = session.execute(
            select(ChatKbBinding.chat_id).where(
                ChatKbBinding.kb_id == kb_id,
                ChatKbBinding.deleted_at.is_(None),
            )
        ).scalars().all()

        if not chat_ids:
            return []

        debug_ids = session.execute(
            select(ChatSettings.chat_id).where(
                ChatSettings.chat_id.in_(chat_ids),
                ChatSettings.debug_mode.is_(True),
            )
        ).scalars().all()

        return list(debug_ids)


def get_debug_chat_ids_for_repo(repo_id: int) -> list[str]:
    """Return chat_ids that are bound to this repo **and** have debug mode on."""
    with get_session() as session:
        chat_ids = session.execute(
            select(ChatRepoBinding.chat_id).where(
                ChatRepoBinding.repo_id == repo_id,
                ChatRepoBinding.deleted_at.is_(None),
            )
        ).scalars().all()

        if not chat_ids:
            return []

        debug_ids = session.execute(
            select(ChatSettings.chat_id).where(
                ChatSettings.chat_id.in_(chat_ids),
                ChatSettings.debug_mode.is_(True),
            )
        ).scalars().all()

        return list(debug_ids)


# ------------------------------------------------------------------
# Broadcast helpers
# ------------------------------------------------------------------

def notify_kb(kb_id: int, message: str) -> None:
    """Send a debug message to all debug-enabled chats bound to a KB."""
    for chat_id in get_debug_chat_ids_for_kb(kb_id):
        _send(chat_id, message)


def notify_repo(repo_id: int, message: str, *, progress: bool = False) -> None:
    """Send a debug message to all debug-enabled chats bound to a repo.

    If *progress* is True, the message is updated in-place (edit previous msg).
    """
    for chat_id in get_debug_chat_ids_for_repo(repo_id):
        if progress:
            _send_or_update(chat_id, message, "repo", repo_id)
        else:
            _send(chat_id, message)


def notify_chat(chat_id: str, message: str) -> None:
    """Send a debug message to a specific chat (only if debug mode is on)."""
    if is_debug(chat_id):
        _send(chat_id, message)
