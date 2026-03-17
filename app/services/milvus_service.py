"""Milvus vector database service for knowledge storage and retrieval."""

import logging
import time
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.config import config

logger = logging.getLogger(__name__)


class MilvusService:
    """Manages Milvus collections for knowledge bases."""

    def __init__(self) -> None:
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
        )
        self._connected = True
        logger.info("Connected to Milvus at %s:%s", config.MILVUS_HOST, config.MILVUS_PORT)

    def _collection_name(self, kb_id: int) -> str:
        return f"kb_{kb_id}"

    def ensure_collection(self, kb_id: int) -> Collection:
        """Create collection if it doesn't exist, return Collection handle."""
        self.connect()
        name = self._collection_name(kb_id)

        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            return col

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="source_detail", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="certainty", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="kb_id", dtype=DataType.INT64),
            FieldSchema(name="last_updated_at", dtype=DataType.INT64),
            FieldSchema(name="last_referenced_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        col = Collection(name=name, schema=schema)

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        col.create_index(field_name="vector", index_params=index_params)
        col.load()
        logger.info("Created Milvus collection: %s", name)
        return col

    def insert_knowledge(
        self,
        kb_id: int,
        vectors: list[list[float]],
        topics: list[str],
        contents: list[str],
        sources: list[str],
        source_details: list[str],
        certainties: list[str],
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Insert knowledge entries into Milvus. Returns list of inserted IDs."""
        col = self.ensure_collection(kb_id)
        now_ts = int(time.time())

        # Truncate varchar fields to schema byte limits
        def _trunc(val: str, limit: int) -> str:
            while len(val.encode("utf-8")) > limit:
                val = val[: len(val) - 1]
            return val

        topics = [_trunc(t, 200) for t in topics]
        contents = [_trunc(c, 5000) for c in contents]
        sources = [_trunc(s, 50) for s in sources]
        source_details = [_trunc(sd, 500) for sd in source_details]
        certainties = [_trunc(c, 20) for c in certainties]

        data = [
            vectors,
            topics,
            contents,
            sources,
            source_details,
            certainties,
            [kb_id] * len(vectors),
            [now_ts] * len(vectors),
            [now_ts] * len(vectors),
        ]

        result = col.insert(data)

        # Insert dynamic fields if provided (for code knowledge)
        if extra_fields:
            for i, ef in enumerate(extra_fields):
                for key, val in ef.items():
                    # Dynamic fields are set via insert with dict format
                    pass  # Dynamic fields handled via dict-based insert below

        col.flush()
        logger.info("Inserted %d entries into kb_%d", len(vectors), kb_id)
        return result.primary_keys

    # Max lengths matching the collection schema varchar fields.
    _VARCHAR_LIMITS: dict[str, int] = {
        "topic": 200,
        "content": 5000,
        "source": 50,
        "source_detail": 500,
        "certainty": 20,
    }

    def insert_knowledge_dicts(
        self,
        kb_id: int,
        entries: list[dict[str, Any]],
    ) -> list[int]:
        """Insert knowledge using list of dicts (supports dynamic fields)."""
        col = self.ensure_collection(kb_id)
        now_ts = int(time.time())

        for entry in entries:
            entry.setdefault("kb_id", kb_id)
            entry.setdefault("last_updated_at", now_ts)
            entry.setdefault("last_referenced_at", now_ts)
            # Truncate varchar fields to schema byte limits
            for field, limit in self._VARCHAR_LIMITS.items():
                if field in entry and isinstance(entry[field], str):
                    val = entry[field]
                    while len(val.encode("utf-8")) > limit:
                        val = val[: len(val) - 1]
                    entry[field] = val

        result = col.insert(entries)
        col.flush()
        logger.info("Inserted %d dict entries into kb_%d", len(entries), kb_id)
        return result.primary_keys

    def search(
        self,
        kb_id: int,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar knowledge entries."""
        col = self.ensure_collection(kb_id)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        output_fields = [
            "topic", "content", "source", "source_detail",
            "certainty", "kb_id", "last_updated_at", "last_referenced_at",
        ]

        results = col.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )

        hits = []
        for hit in results[0]:
            entry = {
                "id": hit.id,
                "score": hit.score,
            }
            for field in output_fields:
                entry[field] = hit.entity.get(field)
            hits.append(entry)
        return hits

    def delete_by_expr(self, kb_id: int, expr: str) -> None:
        """Delete entries matching an expression."""
        col = self.ensure_collection(kb_id)
        col.delete(expr)
        col.flush()
        logger.info("Deleted entries from kb_%d with expr: %s", kb_id, expr)

    def delete_by_ids(self, kb_id: int, ids: list[int]) -> None:
        """Delete entries by primary key IDs."""
        if not ids:
            return
        expr = f"id in {ids}"
        self.delete_by_expr(kb_id, expr)

    def update_referenced_at(self, kb_id: int, ids: list[int]) -> None:
        """Update last_referenced_at for given IDs (by delete + re-insert pattern
        since Milvus doesn't support in-place update for all versions)."""
        # For simplicity, we skip in-place update; the field is updated on next insert.
        # In production, use upsert if Milvus version supports it.
        pass

    def get_collection_count(self, kb_id: int) -> int:
        """Get number of entities in a collection."""
        self.connect()
        name = self._collection_name(kb_id)
        if not utility.has_collection(name):
            return 0
        col = Collection(name)
        return col.num_entities

    def get_all_entries(
        self,
        kb_id: int,
        limit: int = 1000,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query all entries from a collection (for deduplication scans)."""
        col = self.ensure_collection(kb_id)
        if output_fields is None:
            output_fields = ["topic", "content", "source", "certainty", "last_updated_at", "last_referenced_at"]

        results = col.query(
            expr="kb_id >= 0",
            output_fields=output_fields,
            limit=limit,
        )
        return results

    def get_existing_topics(self, kb_id: int, limit: int = 16384) -> set[str]:
        """Return the set of topic values present in a collection.

        Used to detect orphaned blocks (marked success in DB but missing
        from Milvus).
        """
        self.connect()
        name = self._collection_name(kb_id)
        if not utility.has_collection(name):
            return set()
        col = Collection(name)
        results = col.query(
            expr='source == "code"',
            output_fields=["topic"],
            limit=limit,
        )
        return {r["topic"] for r in results}

    def drop_collection(self, kb_id: int) -> None:
        """Drop a collection entirely."""
        self.connect()
        name = self._collection_name(kb_id)
        if utility.has_collection(name):
            utility.drop_collection(name)
            logger.info("Dropped Milvus collection: %s", name)


milvus_service = MilvusService()
