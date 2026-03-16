"""Debug notification helper – sends progress messages to debug-enabled chats."""

import logging
from typing import Any, Callable

from sqlalchemy import select

from app.db.models import ChatKbBinding, ChatRepoBinding, ChatSettings
from app.db.session import get_session

logger = logging.getLogger(__name__)

# Lazy reference to avoid circular imports; set by bot startup.
_send_fn: Callable[[str, str], None] | None = None


def set_send_fn(fn: Callable[[str, str], None]) -> None:
    """Register the bot's send_message function (called once at startup)."""
    global _send_fn
    _send_fn = fn


def _send(chat_id: str, message: str) -> None:
    if _send_fn is None:
        return
    try:
        _send_fn(chat_id, f"🔧 {message}")
    except Exception:
        logger.debug("Debug notify failed for %s", chat_id, exc_info=True)


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


def notify_repo(repo_id: int, message: str) -> None:
    """Send a debug message to all debug-enabled chats bound to a repo."""
    for chat_id in get_debug_chat_ids_for_repo(repo_id):
        _send(chat_id, message)


def notify_chat(chat_id: str, message: str) -> None:
    """Send a debug message to a specific chat (only if debug mode is on)."""
    if is_debug(chat_id):
        _send(chat_id, message)
