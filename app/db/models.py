"""SQLAlchemy ORM models matching schema.md DDL definitions."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Index, Integer, String, Text,
    UniqueConstraint, SmallInteger,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    kb_type: Mapped[str] = mapped_column(
        Enum("chat", "code", "manual", name="kb_type_enum"),
        nullable=False, default="chat",
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    milvus_collection: Mapped[Optional[str]] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class ChatKbBinding(Base):
    __tablename__ = "chat_kb_binding"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(100), nullable=False)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        UniqueConstraint("chat_id", "kb_id", "deleted_at", name="uk_chat_kb"),
    )


class CodeRepo(Base):
    __tablename__ = "code_repo"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    git_url: Mapped[str] = mapped_column(String(500), nullable=False)
    kb_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    default_branch: Mapped[str] = mapped_column(String(100), default="")
    last_commit_hash: Mapped[Optional[str]] = mapped_column(String(64))
    last_analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class ChatRepoBinding(Base):
    __tablename__ = "chat_repo_binding"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(100), nullable=False)
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        UniqueConstraint("chat_id", "repo_id", "deleted_at", name="uk_chat_repo"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(100), nullable=False)
    message_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String(100))
    sender_id: Mapped[Optional[str]] = mapped_column(String(100))
    user_id: Mapped[Optional[str]] = mapped_column(String(100), comment="Feishu user_id (employee number)")
    content: Mapped[Optional[str]] = mapped_column(Text)
    msg_type: Mapped[str] = mapped_column(String(20), default="text")
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    topic_group_id: Mapped[Optional[str]] = mapped_column(String(100))
    pending_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_chat_processed", "chat_id", "processed"),
        Index("idx_topic_group", "topic_group_id"),
        Index("idx_created", "created_at"),
    )


class ManualKnowledge(Base):
    __tablename__ = "manual_knowledge"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chat_id: Mapped[Optional[str]] = mapped_column(String(100))
    sender_id: Mapped[Optional[str]] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class ChatSettings(Base):
    __tablename__ = "chat_settings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    debug_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )


class AdminUser(Base):
    __tablename__ = "admin_user"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sender_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class SecurityLog(Base):
    __tablename__ = "security_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[Optional[str]] = mapped_column(String(100))
    sender_id: Mapped[Optional[str]] = mapped_column(String(100))
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    block_reason: Mapped[Optional[str]] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class CodeBlock(Base):
    __tablename__ = "code_block"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    block_type: Mapped[str] = mapped_column(String(20), nullable=False)
    block_name: Mapped[Optional[str]] = mapped_column(String(200))
    parent_class: Mapped[Optional[str]] = mapped_column(String(200))
    start_line: Mapped[Optional[int]] = mapped_column(Integer)
    end_line: Mapped[Optional[int]] = mapped_column(Integer)
    signature: Mapped[Optional[str]] = mapped_column(Text)
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64), comment="SHA-256 of code content, used to detect changes",
    )
    commit_hash: Mapped[Optional[str]] = mapped_column(String(64))
    description: Mapped[Optional[str]] = mapped_column(
        Text, comment="LLM-generated knowledge text, persisted before Milvus write",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        comment="pending / generated / success / failed",
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
        comment="Number of times this block has been retried",
    )
    milvus_id: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    __table_args__ = (
        Index("idx_repo_file", "repo_id", "file_path"),
        Index("idx_commit", "commit_hash"),
        Index("idx_repo_status", "repo_id", "status"),
    )


class SummarizeTaskLog(Base):
    __tablename__ = "summarize_task_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_type: Mapped[str] = mapped_column(
        Enum("scheduled", "manual", "code", name="task_type_enum"), nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Enum("running", "success", "failed", name="task_status_enum"), nullable=False,
    )
    message_count: Mapped[Optional[int]] = mapped_column(Integer)
    new_knowledge_count: Mapped[Optional[int]] = mapped_column(Integer)
    updated_knowledge_count: Mapped[Optional[int]] = mapped_column(Integer)
    deleted_knowledge_count: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class SummarizeErrorLog(Base):
    __tablename__ = "summarize_error_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    input_text: Mapped[Optional[str]] = mapped_column(Text)
    raw_output: Mapped[Optional[str]] = mapped_column(Text)
    error_type: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
