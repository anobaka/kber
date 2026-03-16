"""Scheduled tasks – message summarization, history compensation, code updates."""

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from app.config import config
from app.db.models import (
    ChatKbBinding,
    ChatMessage,
    ChatRepoBinding,
    CodeRepo,
    KnowledgeBase,
)
from app.db.session import get_session

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def start_scheduler() -> None:
    """Register all periodic tasks and start the scheduler."""
    scheduler.add_job(
        run_message_summarization,
        "interval",
        minutes=config.SUMMARIZE_INTERVAL_MINUTES,
        id="message_summarization",
        replace_existing=True,
    )
    scheduler.add_job(
        run_history_compensation,
        "interval",
        minutes=config.HISTORY_COMPENSATE_INTERVAL_MINUTES,
        id="history_compensation",
        replace_existing=True,
    )
    scheduler.add_job(
        run_code_update_check,
        "interval",
        minutes=config.CODE_UPDATE_INTERVAL_MINUTES,
        id="code_update_check",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: summarize every %dm, history every %dm, code every %dm",
        config.SUMMARIZE_INTERVAL_MINUTES,
        config.HISTORY_COMPENSATE_INTERVAL_MINUTES,
        config.CODE_UPDATE_INTERVAL_MINUTES,
    )


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


# ------------------------------------------------------------------
# Task: Message summarization (every 5 min)
# ------------------------------------------------------------------

def run_message_summarization() -> None:
    """Run summarization for all knowledge bases that have unprocessed messages."""
    from app.services.message_analyzer import knowledge_deflator, message_analyzer

    with get_session() as session:
        kb_ids = session.execute(
            select(KnowledgeBase.id).where(
                KnowledgeBase.deleted_at.is_(None),
                KnowledgeBase.kb_type.in_(["chat", "manual"]),
            )
        ).scalars().all()

    for kb_id in kb_ids:
        try:
            stats = message_analyzer.run_for_kb(kb_id)
            if any(v > 0 for v in stats.values()):
                logger.info("Summarized kb_%d: %s", kb_id, stats)
                # Run deflation after summarization
                deflation_stats = knowledge_deflator.run(kb_id)
                if any(v > 0 for v in deflation_stats.values()):
                    logger.info("Deflated kb_%d: %s", kb_id, deflation_stats)
        except Exception:
            logger.exception("Summarization failed for kb_id=%d", kb_id)


# ------------------------------------------------------------------
# Task: History message compensation (every 30 min)
# ------------------------------------------------------------------

def run_history_compensation() -> None:
    """Pull recent messages from all bound chats to compensate for missed events."""
    with get_session() as session:
        chat_ids = session.execute(
            select(ChatKbBinding.chat_id).where(
                ChatKbBinding.deleted_at.is_(None),
            ).distinct()
        ).scalars().all()

    for chat_id in chat_ids:
        try:
            compensate_history_for_chat(chat_id)
        except Exception:
            logger.exception("History compensation failed for %s", chat_id)


def compensate_history_for_chat(chat_id: str) -> None:
    """Fetch and store recent history messages for a specific chat."""
    from app.bot.handler import feishu_bot
    from app.services.debug_notifier import notify_chat

    notify_chat(chat_id, "开始拉取历史消息补偿...")

    messages = feishu_bot.fetch_history_messages(chat_id)
    if not messages:
        return

    stored = 0
    with get_session() as session:
        for msg in messages:
            if msg.get("msg_type") != "text" or not msg.get("content"):
                continue

            # Dedup
            exists = session.execute(
                select(ChatMessage.id).where(
                    ChatMessage.message_id == msg["message_id"]
                )
            ).scalar_one_or_none()

            if exists:
                continue

            session.add(ChatMessage(
                chat_id=chat_id,
                message_id=msg["message_id"],
                sender_id=msg.get("sender_id"),
                content=msg["content"],
                msg_type="text",
                parent_id=msg.get("parent_id"),
            ))
            stored += 1

    if stored:
        logger.info("Compensated %d messages for chat %s", stored, chat_id)
        notify_chat(chat_id, f"历史消息补偿完成，新增 {stored} 条消息")


# ------------------------------------------------------------------
# Task: Code repository update check (every 30 min)
# ------------------------------------------------------------------

def run_code_update_check() -> None:
    """Check all code repositories for remote updates and run incremental analysis."""
    from app.services.repo_analyzer import repo_analyzer

    with get_session() as session:
        repos = session.execute(
            select(CodeRepo).where(CodeRepo.deleted_at.is_(None))
        ).scalars().all()

    for repo in repos:
        try:
            # check_and_update → analyze_repo, which uses notify_repo internally
            stats = repo_analyzer.check_and_update(repo.id)
            if stats.get("knowledge_generated", 0) > 0:
                logger.info("Updated repo %d: %s", repo.id, stats)
        except Exception:
            logger.exception("Code update check failed for repo_id=%d", repo.id)
