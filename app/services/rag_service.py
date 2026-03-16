"""RAG (Retrieval-Augmented Generation) service for knowledge-based Q&A."""

import logging
from typing import Any

from sqlalchemy import select

from app.db.models import ChatKbBinding, ChatRepoBinding, CodeRepo, KnowledgeBase
from app.db.session import get_session
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.milvus_service import milvus_service
from app.services.security_service import check_input, sanitize_output

logger = logging.getLogger(__name__)


class RAGService:
    """Handles RAG-based Q&A pipeline."""

    def answer(self, chat_id: str, question: str, sender_id: str | None = None) -> str:
        """Full RAG pipeline: security check → retrieve → generate answer."""
        # Security check
        is_safe, reason = check_input(question, chat_id=chat_id, sender_id=sender_id)
        if not is_safe:
            return f"⚠️ {reason}"

        # Get all kb_ids bound to this chat
        kb_ids = self._get_bound_kb_ids(chat_id)
        if not kb_ids:
            return "⚠️ 本群尚未绑定知识库，请先发送「绑定知识库 {名称}」进行绑定。"

        # Embed the question
        try:
            query_vector = embedding_service.embed(question)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return "⚠️ 系统内部错误，请稍后再试。"

        # Search across all bound knowledge bases
        all_hits: list[dict[str, Any]] = []
        for kb_id in kb_ids:
            try:
                hits = milvus_service.search(kb_id, query_vector, top_k=10)
                all_hits.extend(hits)
            except Exception as e:
                logger.warning("Search failed for kb_%d: %s", kb_id, e)

        if not all_hits:
            return "抱歉，知识库中暂未找到与您问题相关的信息。请尝试换个方式提问，或联系管理员添加相关知识。"

        # Sort by score (cosine similarity, higher is better) and apply certainty weighting
        for hit in all_hits:
            base_score = hit.get("score", 0)
            certainty = hit.get("certainty", "unverified")
            weight = {"confirmed": 1.0, "disputed": 0.8, "unverified": 0.6}.get(certainty, 0.6)
            hit["weighted_score"] = base_score * weight

        all_hits.sort(key=lambda h: h["weighted_score"], reverse=True)
        top_hits = all_hits[:15]

        # Update last_referenced_at (best effort)
        for hit in top_hits:
            try:
                milvus_service.update_referenced_at(hit.get("kb_id", 0), [hit["id"]])
            except Exception:
                pass

        # Build context
        context = self._build_context(top_hits)

        # Generate answer
        try:
            answer = llm_service.rag_answer(context, question)
            answer = sanitize_output(answer)
        except Exception as e:
            logger.error("RAG answer generation failed: %s", e)
            return "⚠️ 生成回答时出错，请稍后再试。"

        return answer

    def _get_bound_kb_ids(self, chat_id: str) -> list[int]:
        """Get all knowledge base IDs bound to a chat (including code KBs)."""
        kb_ids: list[int] = []
        with get_session() as session:
            # Direct KB bindings
            rows = session.execute(
                select(ChatKbBinding.kb_id).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalars().all()
            kb_ids.extend(rows)

            # Code repo bindings → their associated KBs
            repo_bindings = session.execute(
                select(ChatRepoBinding.repo_id).where(
                    ChatRepoBinding.chat_id == chat_id,
                    ChatRepoBinding.deleted_at.is_(None),
                )
            ).scalars().all()

            if repo_bindings:
                code_kb_ids = session.execute(
                    select(CodeRepo.kb_id).where(
                        CodeRepo.id.in_(repo_bindings),
                        CodeRepo.deleted_at.is_(None),
                        CodeRepo.kb_id.is_not(None),
                    )
                ).scalars().all()
                kb_ids.extend([kid for kid in code_kb_ids if kid is not None])

        return list(set(kb_ids))

    def _build_context(self, hits: list[dict[str, Any]]) -> str:
        """Build RAG context from search hits."""
        parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            certainty_note = ""
            if hit.get("certainty") == "disputed":
                certainty_note = "⚠️ 此信息存在不同观点"
            elif hit.get("certainty") == "unverified":
                certainty_note = "ℹ️ 此信息待确认"

            part = f"""### 参考资料 {i}
{hit.get('content', '')}
来源：{hit.get('source_detail', hit.get('source', 'unknown'))}
{certainty_note}"""
            parts.append(part)

        return "\n\n".join(parts)


rag_service = RAGService()
