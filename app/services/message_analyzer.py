"""Chat message analysis – topic grouping, status detection, and knowledge extraction."""

import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, select, update

from app.config import config
from app.db.models import (
    ChatKbBinding,
    ChatMessage,
    KnowledgeBase,
    ManualKnowledge,
    SummarizeErrorLog,
    SummarizeTaskLog,
)
from app.db.session import get_session
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.milvus_service import milvus_service

logger = logging.getLogger(__name__)

# Short replies / noise to filter out
STOP_WORDS = {
    "好的", "收到", "ok", "OK", "Ok", "好", "嗯", "嗯嗯", "哦", "行",
    "+1", "👍", "🙏", "谢谢", "感谢", "了解", "明白", "知道了", "对",
    "是的", "没问题", "可以", "哈哈", "哈哈哈", "😂", "🤣",
}

# Max pending rounds before forcing summarization
MAX_PENDING_ROUNDS = 3


class MessageAnalyzer:
    """Orchestrates the message-to-knowledge pipeline."""

    def run_for_kb(self, kb_id: int) -> dict[str, int]:
        """Run a single summarization cycle for one knowledge base.

        Returns dict with counts: {new, updated, deleted, skipped}.
        """
        task_log_id = self._start_task_log(kb_id, "scheduled")
        stats: dict[str, int] = {"new": 0, "updated": 0, "deleted": 0, "skipped": 0}

        try:
            # Load unprocessed messages
            messages = self._load_unprocessed_messages(kb_id)
            manual_entries = self._load_unprocessed_manual(kb_id)

            if not messages and not manual_entries:
                self._finish_task_log(task_log_id, "success", stats)
                return stats

            # Process manual knowledge immediately
            for mk in manual_entries:
                mk_stats = self._process_manual_knowledge(kb_id, mk)
                for k in stats:
                    stats[k] += mk_stats.get(k, 0)

            if not messages:
                self._finish_task_log(task_log_id, "success", stats)
                return stats

            # Filter noise
            meaningful = [m for m in messages if m["content"].strip() not in STOP_WORDS and len(m["content"].strip()) > 2]
            if not meaningful:
                self._mark_messages_processed([m["message_id"] for m in messages])
                self._finish_task_log(task_log_id, "success", stats)
                return stats

            # Format messages for LLM
            messages_text = self._format_messages(meaningful)

            # Topic grouping
            topic_groups = self._group_topics(meaningful, messages_text)

            # Process each topic group
            processed_msg_ids: list[str] = []
            for group in topic_groups:
                group_stats = self._process_topic_group(kb_id, group)
                for k in stats:
                    stats[k] += group_stats.get(k, 0)
                if group.get("should_process"):
                    processed_msg_ids.extend(group["message_ids"])

            # Mark noise messages as processed too
            noise_ids = [m["message_id"] for m in messages if m["content"].strip() in STOP_WORDS or len(m["content"].strip()) <= 2]
            processed_msg_ids.extend(noise_ids)

            if processed_msg_ids:
                self._mark_messages_processed(processed_msg_ids)

            self._finish_task_log(task_log_id, "success", stats)

        except Exception as e:
            logger.exception("Summarization failed for kb_id=%d", kb_id)
            self._finish_task_log(task_log_id, "failed", stats, str(e))
            raise

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_unprocessed_messages(self, kb_id: int) -> list[dict[str, Any]]:
        """Load unprocessed chat messages for a knowledge base."""
        with get_session() as session:
            # Find chat_ids bound to this kb
            bindings = session.execute(
                select(ChatKbBinding.chat_id).where(
                    ChatKbBinding.kb_id == kb_id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalars().all()

            if not bindings:
                return []

            rows = session.execute(
                select(ChatMessage).where(
                    ChatMessage.chat_id.in_(bindings),
                    ChatMessage.processed.is_(False),
                    ChatMessage.msg_type == "text",
                ).order_by(ChatMessage.created_at).limit(500)
            ).scalars().all()

            return [
                {
                    "message_id": r.message_id,
                    "chat_id": r.chat_id,
                    "sender_id": r.sender_id,
                    "content": r.content or "",
                    "created_at": r.created_at,
                    "parent_id": r.parent_id,
                    "topic_group_id": r.topic_group_id,
                    "pending_count": r.pending_count,
                }
                for r in rows
            ]

    def _load_unprocessed_manual(self, kb_id: int) -> list[dict[str, Any]]:
        with get_session() as session:
            rows = session.execute(
                select(ManualKnowledge).where(
                    ManualKnowledge.kb_id == kb_id,
                    ManualKnowledge.processed.is_(False),
                )
            ).scalars().all()
            return [
                {"id": r.id, "content": r.content, "sender_id": r.sender_id, "chat_id": r.chat_id}
                for r in rows
            ]

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for m in messages:
            ts = m["created_at"].strftime("%Y-%m-%d %H:%M") if isinstance(m["created_at"], datetime) else str(m["created_at"])
            lines.append(f"[{ts}] (ID:{m['message_id']}) {m['sender_id']}: {m['content']}")
        return "\n".join(lines)

    def _group_topics(self, messages: list[dict[str, Any]], messages_text: str) -> list[dict[str, Any]]:
        """Use LLM to group messages into topics, then attach message data."""
        try:
            raw = llm_service.group_topics(messages_text)
        except Exception as e:
            logger.error("Topic grouping LLM call failed: %s", e)
            # Fallback: treat all messages as one group
            return [{
                "group_id": "1",
                "summary": "未分组消息",
                "message_ids": [m["message_id"] for m in messages],
                "messages": messages,
            }]

        groups = self._parse_topic_groups(raw, messages)
        return groups

    def _parse_topic_groups(self, raw_output: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse LLM topic grouping output."""
        msg_map = {m["message_id"]: m for m in messages}
        groups: list[dict[str, Any]] = []

        # Parse structured output
        current_group: dict[str, Any] | None = None
        for line in raw_output.split("\n"):
            line = line.strip()
            if "【话题组ID】" in line:
                if current_group:
                    groups.append(current_group)
                gid = re.sub(r".*【话题组ID】[：:]\s*", "", line).strip()
                current_group = {"group_id": gid, "summary": "", "message_ids": [], "messages": []}
            elif "【话题摘要】" in line and current_group:
                current_group["summary"] = re.sub(r".*【话题摘要】[：:]\s*", "", line).strip()
            elif "【消息ID列表】" in line and current_group:
                ids_str = re.sub(r".*【消息ID列表】[：:]\s*", "", line).strip()
                ids = [mid.strip() for mid in ids_str.split(",") if mid.strip()]
                current_group["message_ids"] = ids
                current_group["messages"] = [msg_map[mid] for mid in ids if mid in msg_map]

        if current_group:
            groups.append(current_group)

        # Ensure all messages are assigned to at least one group
        assigned = {mid for g in groups for mid in g["message_ids"]}
        unassigned = [m for m in messages if m["message_id"] not in assigned]
        if unassigned:
            groups.append({
                "group_id": str(len(groups) + 1),
                "summary": "其他消息",
                "message_ids": [m["message_id"] for m in unassigned],
                "messages": unassigned,
            })

        return groups

    def _process_topic_group(self, kb_id: int, group: dict[str, Any]) -> dict[str, int]:
        """Judge status of topic group and summarize if ready."""
        stats: dict[str, int] = {"new": 0, "updated": 0, "deleted": 0, "skipped": 0}
        msgs = group.get("messages", [])
        if not msgs:
            group["should_process"] = True
            return stats

        # Check if forced due to pending_count
        max_pending = max((m.get("pending_count", 0) for m in msgs), default=0)
        force_summarize = max_pending >= MAX_PENDING_ROUNDS

        # Judge discussion status
        discussion_text = self._format_messages(msgs)
        last_msg_time = msgs[-1]["created_at"]
        if isinstance(last_msg_time, datetime):
            minutes_ago = int((datetime.utcnow() - last_msg_time).total_seconds() / 60)
        else:
            minutes_ago = 999

        if force_summarize:
            status = "已沉寂"
            conclusion = "（超过3轮未结论，强制归纳）"
        else:
            try:
                status_raw = llm_service.judge_topic_status(
                    discussion_text,
                    str(last_msg_time),
                    minutes_ago,
                )
                status, conclusion = self._parse_status(status_raw)
            except Exception as e:
                logger.error("Status judgment failed: %s", e)
                # Default to "进行中" on error
                status = "进行中"
                conclusion = ""

        if status == "进行中":
            group["should_process"] = False
            self._increment_pending(group["message_ids"])
            stats["skipped"] = len(msgs)
            return stats

        # Summarize this group
        group["should_process"] = True
        try:
            # Get existing knowledge for context
            existing = self._get_existing_knowledge_context(kb_id)
            result = llm_service.summarize_topic(existing, discussion_text, status, conclusion)

            if "无新增知识" in result:
                return stats

            entries = self._parse_knowledge_entries(result)
            for entry in entries:
                op = entry.get("operation", "新增")
                if op == "新增":
                    self._insert_knowledge(kb_id, entry)
                    stats["new"] += 1
                elif op == "更新":
                    self._update_knowledge(kb_id, entry)
                    stats["updated"] += 1
                elif op == "删除":
                    self._delete_knowledge(kb_id, entry)
                    stats["deleted"] += 1

        except Exception as e:
            logger.exception("Knowledge extraction failed for group %s", group.get("group_id"))
            self._log_error(kb_id, discussion_text, "", "extraction_error", str(e))

        return stats

    def _parse_status(self, raw: str) -> tuple[str, str]:
        """Parse status judgment output. Returns (status, conclusion)."""
        status = "进行中"
        conclusion = ""
        for line in raw.split("\n"):
            line = line.strip()
            if "讨论状态" in line or "状态" in line:
                if "已结论" in line:
                    status = "已结论"
                elif "已沉寂" in line:
                    status = "已沉寂"
                elif "进行中" in line:
                    status = "进行中"
            if "结论" in line and "提取" not in line and "如果" not in line:
                conclusion = re.sub(r".*[：:]\s*", "", line).strip()
        return status, conclusion

    def _parse_knowledge_entries(self, raw: str) -> list[dict[str, str]]:
        """Parse structured knowledge entries from LLM output."""
        entries: list[dict[str, str]] = []
        current: dict[str, str] | None = None

        for line in raw.split("\n"):
            line = line.strip()
            if "【主题】" in line:
                if current and current.get("topic"):
                    entries.append(current)
                current = {}
                current["topic"] = re.sub(r".*【主题】[：:]\s*", "", line).strip()
            elif current is not None:
                if "【内容】" in line:
                    current["content"] = re.sub(r".*【内容】[：:]\s*", "", line).strip()
                elif "【确定性】" in line:
                    val = re.sub(r".*【确定性】[：:]\s*", "", line).strip()
                    certainty_map = {"确定": "confirmed", "有分歧": "disputed", "待确认": "unverified"}
                    current["certainty"] = certainty_map.get(val, "unverified")
                elif "【来源】" in line:
                    current["source_detail"] = re.sub(r".*【来源】[：:]\s*", "", line).strip()
                elif "【操作】" in line:
                    current["operation"] = re.sub(r".*【操作】[：:]\s*", "", line).strip()

        if current and current.get("topic"):
            entries.append(current)
        return entries

    def _get_existing_knowledge_context(self, kb_id: int, top_k: int = 20) -> str:
        """Retrieve existing knowledge summaries from Milvus for context."""
        try:
            entries = milvus_service.get_all_entries(kb_id, limit=top_k)
            if not entries:
                return "（暂无已有知识）"
            lines: list[str] = []
            for e in entries:
                lines.append(f"- [{e.get('topic', '')}] {e.get('content', '')}")
            return "\n".join(lines)
        except Exception:
            return "（暂无已有知识）"

    def _insert_knowledge(self, kb_id: int, entry: dict[str, str]) -> None:
        """Insert a new knowledge entry into Milvus."""
        content = entry.get("content", "")
        topic = entry.get("topic", "")
        if not content:
            return

        text_for_embedding = f"{topic}: {content}"
        vector = embedding_service.embed(text_for_embedding)

        milvus_service.insert_knowledge(
            kb_id=kb_id,
            vectors=[vector],
            topics=[topic[:200]],
            contents=[content[:5000]],
            sources=["chat"],
            source_details=[entry.get("source_detail", "")[:500]],
            certainties=[entry.get("certainty", "unverified")],
        )

    def _update_knowledge(self, kb_id: int, entry: dict[str, str]) -> None:
        """Update an existing knowledge entry by finding similar and replacing."""
        topic = entry.get("topic", "")
        content = entry.get("content", "")
        if not content:
            return

        text_for_embedding = f"{topic}: {content}"
        vector = embedding_service.embed(text_for_embedding)

        # Search for the most similar existing entry
        hits = milvus_service.search(kb_id, vector, top_k=3)
        # Delete the closest match if similarity > 0.8
        ids_to_delete = [h["id"] for h in hits if h["score"] > 0.8]
        if ids_to_delete:
            milvus_service.delete_by_ids(kb_id, ids_to_delete[:1])

        # Insert updated version
        milvus_service.insert_knowledge(
            kb_id=kb_id,
            vectors=[vector],
            topics=[topic[:200]],
            contents=[content[:5000]],
            sources=["chat"],
            source_details=[entry.get("source_detail", "")[:500]],
            certainties=[entry.get("certainty", "unverified")],
        )

    def _delete_knowledge(self, kb_id: int, entry: dict[str, str]) -> None:
        """Delete a knowledge entry by finding similar entries."""
        topic = entry.get("topic", "")
        content = entry.get("content", "")
        text_for_embedding = f"{topic}: {content}"
        try:
            vector = embedding_service.embed(text_for_embedding)
            hits = milvus_service.search(kb_id, vector, top_k=3)
            ids_to_delete = [h["id"] for h in hits if h["score"] > 0.85]
            if ids_to_delete:
                milvus_service.delete_by_ids(kb_id, ids_to_delete)
        except Exception as e:
            logger.warning("Failed to delete knowledge: %s", e)

    def _process_manual_knowledge(self, kb_id: int, mk: dict[str, Any]) -> dict[str, int]:
        """Process a manually added knowledge entry."""
        stats: dict[str, int] = {"new": 0, "updated": 0, "deleted": 0, "skipped": 0}
        content = mk["content"]
        vector = embedding_service.embed(content)

        milvus_service.insert_knowledge(
            kb_id=kb_id,
            vectors=[vector],
            topics=[content[:100]],
            contents=[content[:5000]],
            sources=["manual"],
            source_details=[f"手动添加 by {mk.get('sender_id', 'unknown')}"],
            certainties=["confirmed"],
        )
        stats["new"] = 1

        # Mark as processed
        with get_session() as session:
            session.execute(
                update(ManualKnowledge)
                .where(ManualKnowledge.id == mk["id"])
                .values(processed=True)
            )

        return stats

    def _mark_messages_processed(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        with get_session() as session:
            session.execute(
                update(ChatMessage)
                .where(ChatMessage.message_id.in_(message_ids))
                .values(processed=True)
            )

    def _increment_pending(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        with get_session() as session:
            session.execute(
                update(ChatMessage)
                .where(ChatMessage.message_id.in_(message_ids))
                .values(pending_count=ChatMessage.pending_count + 1)
            )

    def _start_task_log(self, kb_id: int, task_type: str) -> int:
        with get_session() as session:
            log = SummarizeTaskLog(
                kb_id=kb_id,
                task_type=task_type,
                status="running",
                started_at=datetime.utcnow(),
            )
            session.add(log)
            session.flush()
            return log.id

    def _finish_task_log(
        self,
        log_id: int,
        status: str,
        stats: dict[str, int],
        error_msg: str | None = None,
    ) -> None:
        with get_session() as session:
            session.execute(
                update(SummarizeTaskLog)
                .where(SummarizeTaskLog.id == log_id)
                .values(
                    status=status,
                    new_knowledge_count=stats.get("new", 0),
                    updated_knowledge_count=stats.get("updated", 0),
                    deleted_knowledge_count=stats.get("deleted", 0),
                    error_message=error_msg,
                    finished_at=datetime.utcnow(),
                )
            )

    def _log_error(self, kb_id: int, input_text: str, raw_output: str, error_type: str, error_message: str) -> None:
        try:
            with get_session() as session:
                session.add(SummarizeErrorLog(
                    kb_id=kb_id,
                    input_text=input_text[:5000],
                    raw_output=raw_output[:5000],
                    error_type=error_type,
                    error_message=error_message,
                ))
        except Exception:
            logger.exception("Failed to log error")


class KnowledgeDeflator:
    """Knowledge de-bloating: merge similar, age out cold entries, enforce capacity."""

    def run(self, kb_id: int) -> dict[str, int]:
        stats = {"merged": 0, "aged_out": 0}

        # Merge similar
        stats["merged"] = self._merge_similar(kb_id)

        # Age out cold knowledge
        stats["aged_out"] = self._age_out(kb_id)

        # Capacity check
        count = milvus_service.get_collection_count(kb_id)
        if count > config.KB_HARD_LIMIT:
            logger.warning("kb_%d exceeds hard limit (%d/%d)", kb_id, count, config.KB_HARD_LIMIT)

        return stats

    def _merge_similar(self, kb_id: int) -> int:
        """Scan for similar knowledge entries and merge them."""
        try:
            entries = milvus_service.get_all_entries(kb_id, limit=500)
        except Exception:
            return 0

        if len(entries) < 2:
            return 0

        # Embed all entries to find pairs with high similarity
        texts = [f"{e.get('topic', '')}: {e.get('content', '')}" for e in entries]
        try:
            vectors = embedding_service.embed_batch(texts)
        except Exception:
            return 0

        merged = 0
        merged_ids: set[int] = set()
        threshold = config.SIMILARITY_MERGE_THRESHOLD

        for i in range(len(vectors)):
            if entries[i].get("id") in merged_ids:
                continue
            for j in range(i + 1, len(vectors)):
                if entries[j].get("id") in merged_ids:
                    continue
                sim = self._cosine_similarity(vectors[i], vectors[j])
                if sim > threshold:
                    # Merge j into i
                    try:
                        merged_text = llm_service.merge_knowledge(
                            entries[i].get("content", ""),
                            entries[j].get("content", ""),
                        )
                        # Delete both old, insert merged
                        milvus_service.delete_by_ids(kb_id, [entries[j]["id"]])
                        merged_ids.add(entries[j]["id"])
                        merged += 1
                    except Exception as e:
                        logger.warning("Merge failed: %s", e)

        return merged

    def _age_out(self, kb_id: int) -> int:
        """Remove cold knowledge older than threshold."""
        cutoff = int(time.time()) - config.KNOWLEDGE_COLD_DAYS * 86400
        try:
            expr = f"last_referenced_at < {cutoff} and last_updated_at < {cutoff}"
            entries = milvus_service.get_all_entries(kb_id, limit=200)
            cold_ids = [
                e["id"] for e in entries
                if e.get("last_referenced_at", 0) < cutoff and e.get("last_updated_at", 0) < cutoff
            ]
            if cold_ids:
                milvus_service.delete_by_ids(kb_id, cold_ids)
                logger.info("Aged out %d cold entries from kb_%d", len(cold_ids), kb_id)
            return len(cold_ids)
        except Exception:
            return 0

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


message_analyzer = MessageAnalyzer()
knowledge_deflator = KnowledgeDeflator()
