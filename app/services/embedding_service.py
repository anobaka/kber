"""Embedding service using OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from typing import Callable

from openai import OpenAI

from app.config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate text embeddings via Alibaba Cloud Bailian API."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=config.EMBEDDING_API_KEY or config.LLM_API_KEY,
                base_url=config.EMBEDDING_BASE_URL,
                timeout=config.LLM_TIMEOUT,
            )
        return self._client

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 10,
        progress_fn: "Callable[[int, int], None] | None" = None,
    ) -> list[list[float]]:
        """Embed a batch of texts, handling API limits.

        ``progress_fn(done, total)`` is called after each micro-batch
        completes so callers can report progress.
        """
        all_vectors: list[list[float]] = []
        total = len(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(
                        model=config.EMBEDDING_MODEL,
                        input=batch,
                    )
                    vectors = [item.embedding for item in resp.data]
                    all_vectors.extend(vectors)
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        logger.warning("Embedding call failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, e)
                        time.sleep(wait)
                    else:
                        raise
            if progress_fn:
                progress_fn(min(i + batch_size, total), total)
        return all_vectors


embedding_service = EmbeddingService()
