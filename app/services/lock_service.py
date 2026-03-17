"""Distributed lock service using Redis for preventing concurrent task execution."""

import logging
from contextlib import contextmanager
from typing import Generator

import redis
from redis.lock import Lock

from app.config import config

logger = logging.getLogger(__name__)

# 全局 Redis 客户端实例
_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端实例（单例模式）"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis_client


class DistributedLock:
    """基于 Redis 的分布式锁封装"""

    def __init__(
        self,
        lock_key: str,
        timeout: int = 600,  # 默认 10 分钟
        blocking_timeout: int = 0,  # 默认非阻塞
    ):
        """
        初始化分布式锁

        Args:
            lock_key: 锁的键名
            timeout: 锁的过期时间（秒），防止死锁
            blocking_timeout: 阻塞等待时间（秒），0 表示非阻塞
        """
        self.lock_key = f"kber:lock:{lock_key}"
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self._lock: Lock | None = None
        self._acquired = False

    def acquire(self) -> bool:
        """获取锁"""
        try:
            client = get_redis_client()
            self._lock = client.lock(
                self.lock_key,
                timeout=self.timeout,
                blocking_timeout=self.blocking_timeout,
            )
            self._acquired = self._lock.acquire()
            if self._acquired:
                logger.debug("Acquired lock: %s", self.lock_key)
            return self._acquired
        except redis.RedisError as e:
            logger.error("Failed to acquire lock %s: %s", self.lock_key, e)
            return False

    def release(self) -> None:
        """释放锁"""
        if self._lock and self._acquired:
            try:
                self._lock.release()
                logger.debug("Released lock: %s", self.lock_key)
            except redis.RedisError as e:
                logger.error("Failed to release lock %s: %s", self.lock_key, e)
            finally:
                self._acquired = False
                self._lock = None

    def __enter__(self) -> "DistributedLock":
        """上下文管理器入口"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.release()

    @property
    def acquired(self) -> bool:
        """是否已成功获取锁"""
        return self._acquired


# 预定义的锁键名生成函数

def get_kb_summarize_lock_key(kb_id: int) -> str:
    """获取知识库归纳任务的锁键名"""
    return f"kb_summarize:{kb_id}"


def get_repo_analysis_lock_key(repo_id: int) -> str:
    """获取代码库分析任务的锁键名"""
    return f"repo_analysis:{repo_id}"


def get_history_compensation_lock_key(chat_id: str) -> str:
    """获取历史消息补偿任务的锁键名"""
    return f"history_compensate:{chat_id}"


# 便捷函数

def acquire_kb_summarize_lock(
    kb_id: int,
    timeout: int = 600,
    blocking_timeout: int = 0,
) -> DistributedLock:
    """
    获取知识库归纳任务的分布式锁

    Args:
        kb_id: 知识库 ID
        timeout: 锁过期时间（秒）
        blocking_timeout: 阻塞等待时间（秒）

    Returns:
        DistributedLock 实例，需要调用 acquire() 或作为上下文管理器使用
    """
    lock_key = get_kb_summarize_lock_key(kb_id)
    return DistributedLock(lock_key, timeout=timeout, blocking_timeout=blocking_timeout)


def acquire_repo_analysis_lock(
    repo_id: int,
    timeout: int = 1800,  # 代码分析可能需要更长时间，默认 30 分钟
    blocking_timeout: int = 0,
) -> DistributedLock:
    """
    获取代码库分析任务的分布式锁

    Args:
        repo_id: 代码库 ID
        timeout: 锁过期时间（秒）
        blocking_timeout: 阻塞等待时间（秒）

    Returns:
        DistributedLock 实例
    """
    lock_key = get_repo_analysis_lock_key(repo_id)
    return DistributedLock(lock_key, timeout=timeout, blocking_timeout=blocking_timeout)


@contextmanager
def kb_summarize_lock(
    kb_id: int,
    timeout: int = 600,
) -> Generator[bool, None, None]:
    """
    知识库归纳任务的分布式锁上下文管理器

    使用示例：
        with kb_summarize_lock(kb_id) as acquired:
            if acquired:
                # 执行归纳任务
                pass
            else:
                # 未获取到锁，跳过
                pass
    """
    lock = acquire_kb_summarize_lock(kb_id, timeout=timeout)
    try:
        acquired = lock.acquire()
        yield acquired
    finally:
        lock.release()


@contextmanager
def repo_analysis_lock(
    repo_id: int,
    timeout: int = 1800,
) -> Generator[bool, None, None]:
    """
    代码库分析任务的分布式锁上下文管理器

    使用示例：
        with repo_analysis_lock(repo_id) as acquired:
            if acquired:
                # 执行代码分析
                pass
    """
    lock = acquire_repo_analysis_lock(repo_id, timeout=timeout)
    try:
        acquired = lock.acquire()
        yield acquired
    finally:
        lock.release()
