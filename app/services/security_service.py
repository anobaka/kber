"""Prompt security guard – multi-layer protection against prompt injection."""

import logging
import re

from sqlalchemy import select

from app.db.models import SecurityLog
from app.db.session import get_session

logger = logging.getLogger(__name__)

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"忽略以上指令", re.IGNORECASE),
    re.compile(r"忽略上面的", re.IGNORECASE),
    re.compile(r"忽略之前的", re.IGNORECASE),
    re.compile(r"你现在是", re.IGNORECASE),
    re.compile(r"你的新角色", re.IGNORECASE),
    re.compile(r"角色扮演", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"<\|", re.IGNORECASE),
    re.compile(r"\|\>", re.IGNORECASE),
    re.compile(r"ignore (?:all )?(?:previous|above) instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new instructions?:", re.IGNORECASE),
    re.compile(r"override (?:system|instructions)", re.IGNORECASE),
    re.compile(r"forget (?:everything|your rules)", re.IGNORECASE),
    re.compile(r"disregard", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s*mode", re.IGNORECASE),
]

MAX_INPUT_LENGTH = 2000
MAX_OUTPUT_LENGTH = 3000


def _strip_control_chars(text: str) -> str:
    """Remove special control characters except common whitespace."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def check_input(text: str, chat_id: str | None = None, sender_id: str | None = None) -> tuple[bool, str]:
    """Check user input for prompt injection.

    Returns:
        (is_safe, reason) – reason is empty if safe.
    """
    # Layer 1: strip control characters
    cleaned = _strip_control_chars(text)

    # Length check
    if len(cleaned) > MAX_INPUT_LENGTH:
        _log_block(chat_id, sender_id, text, "input_too_long")
        return False, f"输入长度超过 {MAX_INPUT_LENGTH} 字符限制"

    # Pattern matching
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            reason = f"检测到注入模式: {pattern.pattern}"
            _log_block(chat_id, sender_id, text, reason)
            return False, "检测到不安全的输入内容，请修改后重试"

    return True, ""


def sanitize_output(text: str) -> str:
    """Sanitize LLM output – truncate if too long."""
    if len(text) > MAX_OUTPUT_LENGTH:
        text = text[:MAX_OUTPUT_LENGTH] + "\n\n...(内容过长，已截断)"
    return text


def _log_block(chat_id: str | None, sender_id: str | None, raw_text: str, reason: str) -> None:
    """Record a security block event."""
    try:
        with get_session() as session:
            log_entry = SecurityLog(
                chat_id=chat_id,
                sender_id=sender_id,
                raw_text=raw_text[:5000],
                block_reason=reason[:200],
            )
            session.add(log_entry)
    except Exception:
        logger.exception("Failed to write security log")
