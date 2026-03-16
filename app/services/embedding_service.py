"""Embedding service using OpenAI-compatible API."""

import logging
import time

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
                timeout=30,
            )
        return self._client

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Embed a batch of texts, handling API limits."""
        all_vectors: list[list[float]] = []
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
        return all_vectors


embedding_service = EmbeddingService()
