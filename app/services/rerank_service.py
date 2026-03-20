"""Rerank service using Alibaba Cloud DashScope API."""

from __future__ import annotations

import logging
import time

import requests

from app.config import config

logger = logging.getLogger(__name__)


class RerankService:
    """Rerank documents using Alibaba Cloud DashScope API."""

    def __init__(self) -> None:
        self._api_key: str | None = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = config.RERANK_API_KEY
        return self._api_key

    @property
    def base_url(self) -> str:
        return config.RERANK_BASE_URL

    def rerank(self, query: str, documents: list[str], top_n: int = 10, return_documents: bool = True) -> list[dict]:
        """
        对文档列表进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回前 N 个结果
            return_documents: 是否返回文档内容
        
        Returns:
            [{"index": 0, "relevance_score": 0.95, "document": "..."}, ...]
        """
        if not documents:
            return []
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": config.RERANK_MODEL,
                        "input": {
                            "query": query,
                            "documents": documents,
                        },
                        "parameters": {
                            "return_documents": return_documents,
                            "top_n": min(top_n, len(documents)),
                        },
                    },
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                
                # 解析结果
                results = []
                for item in result.get("output", {}).get("results", []):
                    doc = item.get("document", "")
                    # document 可能是 dict 格式 {'text': '...'} 或字符串
                    if isinstance(doc, dict):
                        doc = doc.get("text", "")
                    results.append({
                        "index": item.get("index", 0),
                        "relevance_score": item.get("relevance_score", 0),
                        "document": doc if return_documents else "",
                    })
                return results
            
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rerank call failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    logger.error("Rerank call failed after 3 attempts: %s", e)
                    raise
        
        return []

    def rerank_with_metadata(self, query: str, items: list[dict], top_n: int = 10, content_key: str = "content") -> list[dict]:
        """
        对带有元数据的文档列表进行重排序
        
        Args:
            query: 查询文本
            items: 待排序的文档列表，每个元素是 dict
            top_n: 返回前 N 个结果
            content_key: 文档内容在 dict 中的 key
        
        Returns:
            排序后的 items 列表，每个元素添加 relevance_score
        """
        if not items:
            return []
        
        # 提取文档内容
        documents = [item.get(content_key, "") for item in items]
        
        # 调用 rerank
        rerank_results = self.rerank(query, documents, top_n=top_n, return_documents=False)
        
        # 按重排序结果重新排列
        sorted_items = []
        for result in rerank_results:
            index = result.get("index", 0)
            if 0 <= index < len(items):
                item = items[index].copy()
                item["relevance_score"] = result.get("relevance_score", 0)
                sorted_items.append(item)
        
        return sorted_items


rerank_service = RerankService()