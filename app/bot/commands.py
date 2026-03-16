"""Command router – parses user commands and dispatches to handlers."""

import logging
import re
import threading
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update

from app.db.models import (
    AdminUser,
    ChatKbBinding,
    ChatMessage,
    ChatRepoBinding,
    CodeRepo,
    KnowledgeBase,
    ManualKnowledge,
    SummarizeTaskLog,
)
from app.db.session import get_session
from app.services.milvus_service import milvus_service
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

GIT_URL_PATTERN = re.compile(
    r"^(git@[\w.\-]+:[\w.\-/]+\.git|https?://[\w.\-/]+\.git|https?://[\w.\-/]+)$"
)


class CommandRouter:
    """Routes incoming user commands to the appropriate handler."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot

    def handle(self, chat_id: str, message_id: str, sender_id: str, text: str) -> None:
        """Parse command prefix and dispatch."""
        text = text.strip()

        if text.startswith("绑定知识库"):
            self._bind_kb(chat_id, sender_id, text[len("绑定知识库"):].strip())
        elif text.startswith("解绑知识库"):
            self._unbind_kb(chat_id, sender_id, text[len("解绑知识库"):].strip())
        elif text.startswith("绑定代码库"):
            self._bind_repo(chat_id, sender_id, text[len("绑定代码库"):].strip())
        elif text.startswith("解绑代码库"):
            self._unbind_repo(chat_id, sender_id, text[len("解绑代码库"):].strip())
        elif text.startswith("添加知识"):
            self._add_knowledge(chat_id, sender_id, text[len("添加知识"):].strip())
        elif text == "立即总结":
            self._force_summarize(chat_id, sender_id)
        elif text == "查询知识库":
            self._query_kb_status(chat_id, sender_id)
        elif text in ("帮助", "help"):
            self._show_help(chat_id)
        else:
            # Free-form question → RAG
            self._rag_query(chat_id, sender_id, text)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _bind_kb(self, chat_id: str, sender_id: str, kb_name: str) -> None:
        if not kb_name:
            self.bot.send_message(chat_id, "⚠️ 知识库名称不能为空，请使用格式：绑定知识库 {名称}")
            return

        with get_session() as session:
            # Find or create KB
            kb = session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.name == kb_name,
                    KnowledgeBase.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not kb:
                kb = KnowledgeBase(
                    name=kb_name,
                    kb_type="chat",
                    milvus_collection=None,  # Will be set when collection is created
                )
                session.add(kb)
                session.flush()

            # Check existing binding
            existing = session.execute(
                select(ChatKbBinding).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.kb_id == kb.id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if existing:
                self.bot.send_message(chat_id, f"ℹ️ 本群已绑定到知识库「{kb_name}」。")
                return

            session.add(ChatKbBinding(chat_id=chat_id, kb_id=kb.id))

            # Ensure Milvus collection exists
            milvus_service.ensure_collection(kb.id)

            if not kb.milvus_collection:
                kb.milvus_collection = f"kb_{kb.id}"

        self.bot.send_message(chat_id, f"✅ 已将本群绑定到知识库「{kb_name}」，历史消息将被纳入知识库。")

        # Trigger async history fetch
        self._async_history_compensate(chat_id)

    def _unbind_kb(self, chat_id: str, sender_id: str, kb_name: str) -> None:
        if not kb_name:
            self.bot.send_message(chat_id, "⚠️ 知识库名称不能为空，请使用格式：解绑知识库 {名称}")
            return

        with get_session() as session:
            kb = session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.name == kb_name,
                    KnowledgeBase.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not kb:
                self.bot.send_message(chat_id, f"⚠️ 知识库「{kb_name}」不存在。")
                return

            binding = session.execute(
                select(ChatKbBinding).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.kb_id == kb.id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not binding:
                self.bot.send_message(chat_id, f"⚠️ 本群未绑定知识库「{kb_name}」。")
                return

            binding.deleted_at = datetime.utcnow()

        self.bot.send_message(chat_id, f"✅ 已解绑知识库「{kb_name}」，后续消息将不再纳入该知识库。")

    def _bind_repo(self, chat_id: str, sender_id: str, git_url: str) -> None:
        if not git_url:
            self.bot.send_message(chat_id, "⚠️ 代码库地址不能为空，请使用格式：绑定代码库 {git地址}")
            return

        if not GIT_URL_PATTERN.match(git_url):
            self.bot.send_message(chat_id, "⚠️ Git 地址格式不正确，请使用 git@... 或 https://... 格式。")
            return

        with get_session() as session:
            # Find or create repo
            repo = session.execute(
                select(CodeRepo).where(
                    CodeRepo.git_url == git_url,
                    CodeRepo.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not repo:
                # Create a code-type knowledge base for this repo
                repo_name = git_url.split("/")[-1].replace(".git", "")
                kb = KnowledgeBase(
                    name=repo_name,
                    kb_type="code",
                    description=f"Code knowledge from {git_url}",
                )
                session.add(kb)
                session.flush()

                milvus_service.ensure_collection(kb.id)
                kb.milvus_collection = f"kb_{kb.id}"

                repo = CodeRepo(
                    git_url=git_url,
                    kb_id=kb.id,
                )
                session.add(repo)
                session.flush()

            # Check existing binding
            existing = session.execute(
                select(ChatRepoBinding).where(
                    ChatRepoBinding.chat_id == chat_id,
                    ChatRepoBinding.repo_id == repo.id,
                    ChatRepoBinding.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if existing:
                self.bot.send_message(chat_id, f"ℹ️ 本群已绑定该代码库。")
                return

            session.add(ChatRepoBinding(chat_id=chat_id, repo_id=repo.id))
            repo_id = repo.id

        self.bot.send_message(chat_id, "✅ 已绑定代码库，正在分析代码库，请稍候...")

        # Trigger async full analysis
        self._async_repo_analysis(repo_id, chat_id)

    def _unbind_repo(self, chat_id: str, sender_id: str, git_url: str) -> None:
        if not git_url:
            self.bot.send_message(chat_id, "⚠️ 代码库地址不能为空。")
            return

        with get_session() as session:
            repo = session.execute(
                select(CodeRepo).where(
                    CodeRepo.git_url == git_url,
                    CodeRepo.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not repo:
                self.bot.send_message(chat_id, "⚠️ 未找到该代码库。")
                return

            binding = session.execute(
                select(ChatRepoBinding).where(
                    ChatRepoBinding.chat_id == chat_id,
                    ChatRepoBinding.repo_id == repo.id,
                    ChatRepoBinding.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

            if not binding:
                self.bot.send_message(chat_id, "⚠️ 本群未绑定该代码库。")
                return

            binding.deleted_at = datetime.utcnow()

        self.bot.send_message(chat_id, "✅ 已解绑代码库。")

    def _add_knowledge(self, chat_id: str, sender_id: str, content: str) -> None:
        if not content:
            self.bot.send_message(chat_id, "⚠️ 知识内容不能为空，请使用格式：添加知识 {内容}")
            return

        if len(content) > 5000:
            self.bot.send_message(chat_id, "⚠️ 知识内容过长，请控制在 5000 字符以内。")
            return

        with get_session() as session:
            # Get non-code KB bindings
            bindings = session.execute(
                select(ChatKbBinding.kb_id).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalars().all()

            if not bindings:
                self.bot.send_message(chat_id, "⚠️ 本群尚未绑定知识库，请先发送「绑定知识库 {名称}」进行绑定。")
                return

            # Filter out code-type KBs
            kb_ids = session.execute(
                select(KnowledgeBase.id).where(
                    KnowledgeBase.id.in_(bindings),
                    KnowledgeBase.kb_type != "code",
                    KnowledgeBase.deleted_at.is_(None),
                )
            ).scalars().all()

            if not kb_ids:
                self.bot.send_message(chat_id, "⚠️ 本群绑定的知识库均为代码库，无法手动添加知识。请先绑定一个聊天知识库。")
                return

            for kb_id in kb_ids:
                session.add(ManualKnowledge(
                    kb_id=kb_id,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    content=content,
                ))

        self.bot.send_message(chat_id, "✅ 知识已添加，正在归纳整合中。")

        # Trigger immediate summarization
        for kb_id in kb_ids:
            self._async_summarize(kb_id)

    def _force_summarize(self, chat_id: str, sender_id: str) -> None:
        if not self._is_admin(sender_id):
            self.bot.send_message(chat_id, "⚠️ 你没有执行此命令的权限，请联系管理员。")
            return

        with get_session() as session:
            kb_ids = session.execute(
                select(ChatKbBinding.kb_id).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.deleted_at.is_(None),
                )
            ).scalars().all()

            # Also get code repos
            repo_ids = session.execute(
                select(ChatRepoBinding.repo_id).where(
                    ChatRepoBinding.chat_id == chat_id,
                    ChatRepoBinding.deleted_at.is_(None),
                )
            ).scalars().all()

        if not kb_ids and not repo_ids:
            self.bot.send_message(chat_id, "⚠️ 本群尚未绑定任何知识库或代码库。")
            return

        total = len(set(kb_ids)) + len(set(repo_ids))
        self.bot.send_message(chat_id, f"🔄 已触发知识库归纳任务，涉及 {total} 个知识库/代码库，请稍候...")

        for kb_id in set(kb_ids):
            self._async_summarize(kb_id)
        for repo_id in set(repo_ids):
            self._async_repo_analysis(repo_id, chat_id)

    def _query_kb_status(self, chat_id: str, sender_id: str) -> None:
        if not self._is_admin(sender_id):
            self.bot.send_message(chat_id, "⚠️ 你没有执行此命令的权限，请联系管理员。")
            return

        with get_session() as session:
            kbs = session.execute(
                select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
            ).scalars().all()

            if not kbs:
                self.bot.send_message(chat_id, "ℹ️ 当前没有任何知识库。")
                return

            lines = [f"📚 知识库列表（共 {len(kbs)} 个）\n"]

            for i, kb in enumerate(kbs, 1):
                # Count bindings
                binding_count = session.execute(
                    select(func.count()).select_from(ChatKbBinding).where(
                        ChatKbBinding.kb_id == kb.id,
                        ChatKbBinding.deleted_at.is_(None),
                    )
                ).scalar() or 0

                # Count Milvus entries
                try:
                    entry_count = milvus_service.get_collection_count(kb.id)
                except Exception:
                    entry_count = 0

                # Last summarize time
                last_log = session.execute(
                    select(SummarizeTaskLog).where(
                        SummarizeTaskLog.kb_id == kb.id,
                        SummarizeTaskLog.status == "success",
                    ).order_by(SummarizeTaskLog.finished_at.desc()).limit(1)
                ).scalar_one_or_none()

                last_time = "从未"
                if last_log and last_log.finished_at:
                    delta = datetime.utcnow() - last_log.finished_at
                    if delta.total_seconds() < 3600:
                        last_time = f"{int(delta.total_seconds() / 60)} 分钟前"
                    elif delta.total_seconds() < 86400:
                        last_time = f"{int(delta.total_seconds() / 3600)} 小时前"
                    else:
                        last_time = f"{int(delta.days)} 天前"

                # Code repo info
                code_info = ""
                if kb.kb_type == "code":
                    code_repo = session.execute(
                        select(CodeRepo).where(
                            CodeRepo.kb_id == kb.id,
                            CodeRepo.deleted_at.is_(None),
                        )
                    ).scalar_one_or_none()
                    if code_repo:
                        code_info = f"\n   代码库：{code_repo.git_url}"

                lines.append(
                    f"{i}. {kb.name}（类型：{kb.kb_type}）{code_info}\n"
                    f"   绑定群：{binding_count} | 知识条目：{entry_count:,} | 最近归纳：{last_time}"
                )

            self.bot.send_card(chat_id, "知识库状态", "\n\n".join(lines))

    def _show_help(self, chat_id: str) -> None:
        help_text = """📖 **可用命令：**

**绑定知识库** {名称}　— 将本群聊天记录纳入指定知识库
**解绑知识库** {名称}　— 解除本群与知识库的绑定
**绑定代码库** {git地址}　— 关联代码库并自动分析
**解绑代码库** {git地址}　— 解除代码库关联
**添加知识** {内容}　— 手动向知识库添加一条知识
**帮助**　— 显示本帮助信息

🔒 **管理员命令：**
**立即总结**　— 立即触发知识归纳任务
**查询知识库**　— 查看所有知识库状态

💬 直接提问即可查询知识库，例如：「这个接口怎么调用？」"""
        self.bot.send_card(chat_id, "帮助", help_text)

    def _rag_query(self, chat_id: str, sender_id: str, question: str) -> None:
        if not question:
            return
        answer = rag_service.answer(chat_id, question, sender_id=sender_id)
        self.bot.send_message(chat_id, answer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_admin(sender_id: str) -> bool:
        with get_session() as session:
            admin = session.execute(
                select(AdminUser).where(AdminUser.sender_id == sender_id)
            ).scalar_one_or_none()
            return admin is not None

    @staticmethod
    def chat_has_kb_binding(chat_id: str) -> bool:
        with get_session() as session:
            binding = session.execute(
                select(ChatKbBinding.id).where(
                    ChatKbBinding.chat_id == chat_id,
                    ChatKbBinding.deleted_at.is_(None),
                ).limit(1)
            ).scalar_one_or_none()
            return binding is not None

    def _async_history_compensate(self, chat_id: str) -> None:
        """Trigger async history message fetch."""
        def _run() -> None:
            try:
                from app.scheduler.tasks import compensate_history_for_chat
                compensate_history_for_chat(chat_id)
            except Exception:
                logger.exception("History compensate failed for %s", chat_id)

        threading.Thread(target=_run, daemon=True).start()

    def _async_summarize(self, kb_id: int) -> None:
        """Trigger async summarization for a KB."""
        def _run() -> None:
            try:
                from app.services.message_analyzer import message_analyzer
                message_analyzer.run_for_kb(kb_id)
            except Exception:
                logger.exception("Summarize failed for kb_id=%d", kb_id)

        threading.Thread(target=_run, daemon=True).start()

    def _async_repo_analysis(self, repo_id: int, chat_id: str) -> None:
        """Trigger async repo analysis."""
        def _run() -> None:
            try:
                from app.services.repo_analyzer import repo_analyzer
                repo_analyzer.analyze_repo(
                    repo_id,
                    chat_id=chat_id,
                    progress_callback=lambda cid, msg: self.bot.send_message(cid, msg),
                )
            except Exception:
                logger.exception("Repo analysis failed for repo_id=%d", repo_id)

        threading.Thread(target=_run, daemon=True).start()
