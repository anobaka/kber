"""RAG (Retrieval-Augmented Generation) service for knowledge-based Q&A."""

import logging
import threading
from collections import deque
from typing import Any

from sqlalchemy import select

from app.db.models import ChatKbBinding, ChatRepoBinding, CodeRepo, KnowledgeBase
from app.db.session import get_session
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.milvus_service import milvus_service
from app.services.rerank_service import rerank_service
from app.services.security_service import check_input, sanitize_output

logger = logging.getLogger(__name__)

# Per-chat conversation history for multi-turn context.
# Key: chat_id → deque of (question, answer) tuples (max 5 rounds).
_chat_history: dict[str, deque[tuple[str, str]]] = {}
_history_lock = threading.Lock()

# Chats that requested a one-time context clear.
_clear_next: set[str] = set()

_MAX_HISTORY = 5


class RAGService:
    """Handles RAG-based Q&A pipeline."""

    def clear_context(self, chat_id: str) -> None:
        """Clear conversation history for the next @mention in a chat."""
        with _history_lock:
            _clear_next.add(chat_id)

    def answer(self, chat_id: str, question: str, sender_id: str | None = None) -> str:
        """Full RAG pipeline: security check → retrieve → generate answer."""
        # Security check
        is_safe, reason = check_input(question, chat_id=chat_id, sender_id=sender_id)
        if not is_safe:
            return f"⚠️ {reason}"

        # Get all kb_ids bound to this chat
        kb_ids = self._get_bound_kb_ids(chat_id)
        logger.info("Chat %s bound kb_ids: %s", chat_id, kb_ids)
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
                logger.info("KB %d returned %d hits", kb_id, len(hits))
                all_hits.extend(hits)
            except Exception as e:
                logger.warning("Search failed for kb_%d: %s", kb_id, e)

        if not all_hits:
            return "抱歉，知识库中暂未找到与您问题相关的信息。请尝试换个方式提问，或联系管理员添加相关知识。"

        # 保存原始排序结果（用于对比）
        original_hits = all_hits.copy()
        for hit in original_hits:
            base_score = hit.get("score", 0)
            certainty = hit.get("certainty", "unverified")
            weight = {"confirmed": 1.0, "disputed": 0.8, "unverified": 0.6}.get(certainty, 0.6)
            hit["weighted_score"] = base_score * weight
        original_hits.sort(key=lambda h: h.get("weighted_score", 0), reverse=True)

        # Rerank: 使用重排序模型对检索结果进行精排
        try:
            # 提取文档内容用于重排序
            documents = [hit.get("content", "") for hit in all_hits]
            rerank_results = rerank_service.rerank(question, documents, top_n=15, return_documents=False)
            
            # 按重排序结果重新排列
            reranked_hits = []
            for result in rerank_results:
                index = result.get("index", 0)
                if 0 <= index < len(all_hits):
                    hit = all_hits[index].copy()
                    hit["rerank_score"] = result.get("relevance_score", 0)
                    reranked_hits.append(hit)
            
            if reranked_hits:
                all_hits = reranked_hits
                
                # 对比重排序和原始排序的差异
                self._log_rerank_comparison(question, original_hits[:15], reranked_hits[:15])
            else:
                all_hits = original_hits
        except Exception as e:
            logger.warning("Rerank failed, using original order: %s", e)
            all_hits = original_hits

        top_hits = all_hits[:15]

        # Update last_referenced_at (best effort)
        for hit in top_hits:
            try:
                milvus_service.update_referenced_at(hit.get("kb_id", 0), [hit["id"]])
            except Exception:
                pass

        # Build context
        context = self._build_context(top_hits)

        # Build conversation history (skip if cleared)
        history_text = ""
        with _history_lock:
            should_clear = chat_id in _clear_next
            if should_clear:
                _clear_next.discard(chat_id)
                _chat_history.pop(chat_id, None)
            else:
                history = _chat_history.get(chat_id)
                if history:
                    parts = []
                    for q, a in history:
                        parts.append(f"用户：{q}\n助手：{a}")
                    history_text = "\n\n".join(parts)

        # Generate answer
        try:
            answer = llm_service.rag_answer(context, question, history=history_text)
            answer = sanitize_output(answer)
            logger.info("回答：%s", answer)
        except Exception as e:
            logger.error("RAG answer generation failed: %s", e)
            return "⚠️ 生成回答时出错，请稍后再试。"

        # Store in conversation history
        with _history_lock:
            if chat_id not in _chat_history:
                _chat_history[chat_id] = deque(maxlen=_MAX_HISTORY)
            _chat_history[chat_id].append((question, answer))

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

        # 统计开发人员信息（用于回答"主要开发人员"类问题）
        author_stats: dict[str, dict] = {}
        for hit in hits:
            if hit.get("commit_author"):
                author = hit["commit_author"]
                if author not in author_stats:
                    author_stats[author] = {"count": 0, "files": set()}
                author_stats[author]["count"] += 1
                if hit.get("file_path"):
                    author_stats[author]["files"].add(hit["file_path"])

        # 如果有多个代码块，添加开发人员统计
        if len(hits) > 1 and author_stats:
            # 按涉及代码块数量和文件数量排序
            sorted_authors = sorted(
                author_stats.items(),
                key=lambda x: (x[1]["count"], len(x[1]["files"])),
                reverse=True
            )
            dev_summary = "### 项目开发人员统计\n"
            dev_summary += f"共检索到 {len(hits)} 个代码块，涉及 {len(author_stats)} 位开发人员：\n\n"
            for i, (author, stats) in enumerate(sorted_authors[:5], 1):
                file_count = len(stats["files"])
                block_count = stats["count"]
                dev_summary += f"{i}. **{author}**：参与 {file_count} 个文件，{block_count} 个代码块\n"
            if len(sorted_authors) > 5:
                dev_summary += f"\n... 及其他 {len(sorted_authors) - 5} 位开发人员\n"
            dev_summary += "\n---\n"
            parts.append(dev_summary)

        for i, hit in enumerate(hits, 1):
            certainty_note = ""
            if hit.get("certainty") == "disputed":
                certainty_note = "⚠️ 此信息存在不同观点"
            elif hit.get("certainty") == "unverified":
                certainty_note = "ℹ️ 此信息待确认"

            # 构建开发者信息（独立段落）
            dev_info = ""
            if hit.get("commit_author"):
                dev_info += f"\n**最后修改**：{hit['commit_author']}"
                if hit.get("commit_date"):
                    dev_info += f" @ {hit['commit_date'][:10] if len(hit['commit_date']) > 10 else hit['commit_date']}"

            # 添加贡献者信息
            if hit.get("contributors"):
                try:
                    import json
                    contributors = json.loads(hit["contributors"])
                    if contributors:
                        dev_info += "\n**贡献者**："
                        for j, c in enumerate(contributors[:2], 1):
                            dev_info += f"\n  - {c.get('author', '未知')}：{c.get('commits', 0)} 次提交"
                        if len(contributors) > 2:
                            dev_info += f"\n  - ... 及其他 {len(contributors) - 2} 位"
                except Exception:
                    pass

            part = f"""### 参考资料 {i}
{hit.get('content', '')}
来源：{hit.get('source_detail', hit.get('source', 'unknown'))}{dev_info}
{certainty_note}"""
            parts.append(part)

        return "\n\n".join(parts)

    def _log_rerank_comparison(self, question: str, original_hits: list[dict], reranked_hits: list[dict]) -> None:
        """记录重排序和原始排序的对比日志"""
        top_n = 15
        logger.info("=" * 60)
        logger.info("Rerank 对比分析 - 问题: %s", question[:50] + "..." if len(question) > 50 else question)
        logger.info("-" * 60)
        
        # 原始排序 Top N
        logger.info("【原始排序 Top %d】", top_n)
        for i, hit in enumerate(original_hits[:top_n], 1):
            source = hit.get("source_detail", hit.get("source", "unknown"))
            topic = hit.get("topic", "unknown")[:30]
            score = hit.get("weighted_score", hit.get("score", 0))
            logger.info("  %d. %s - %s (score=%.4f)", i, source, topic, score)
        
        # 重排序 Top N
        logger.info("【重排序 Top %d】", top_n)
        for i, hit in enumerate(reranked_hits[:top_n], 1):
            source = hit.get("source_detail", hit.get("source", "unknown"))
            topic = hit.get("topic", "unknown")[:30]
            rerank_score = hit.get("rerank_score", 0)
            logger.info("  %d. %s - %s (rerank_score=%.4f)", i, source, topic, rerank_score)
        
        # 分析排序变化
        original_top_ids = [h.get("id") for h in original_hits[:top_n]]
        reranked_top_ids = [h.get("id") for h in reranked_hits[:top_n]]
        
        # 计算重叠率
        overlap = len(set(original_top_ids) & set(reranked_top_ids))
        overlap_rate = overlap / top_n * 100 if original_top_ids else 0
        
        # 计算排名变化
        rank_changes = []
        for i, hit in enumerate(reranked_hits[:top_n]):
            hit_id = hit.get("id")
            if hit_id in original_top_ids:
                original_rank = original_top_ids.index(hit_id) + 1
                new_rank = i + 1
                change = original_rank - new_rank
                rank_changes.append(change)
        
        avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0
        
        logger.info("-" * 60)
        logger.info("【对比统计】")
        logger.info("  Top %d 重叠率: %.0f%% (%d/%d)", top_n, overlap_rate, overlap, top_n)
        logger.info("  平均排名变化: %.2f (正值=提升, 负值=下降)", avg_rank_change)
        logger.info("=" * 60)


rag_service = RAGService()
