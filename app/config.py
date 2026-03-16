"""Application configuration loaded from environment variables."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Feishu ---
    FEISHU_APP_ID: str = os.getenv("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET: str = os.getenv("FEISHU_APP_SECRET", "")

    # --- MySQL ---
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "kber")

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    # --- Milvus ---
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))

    # --- LLM (Alibaba Cloud Bailian, OpenAI-compatible) ---
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "minimax-m2.5")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # --- Embedding ---
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    # --- Git / Repo ---
    # PAT per host: JSON mapping {"github.com": "ghp_xxx", "gitlab.myco.com": "glpat-xxx"}
    # Falls back to GIT_PAT as default for any host not in the map.
    GIT_PAT: str = os.getenv("GIT_PAT", "")
    GIT_PAT_MAP: dict[str, str] = json.loads(os.getenv("GIT_PAT_MAP", "{}"))
    REPOS_BASE_DIR: str = os.getenv("REPOS_BASE_DIR", "/data/repos")

    def get_pat_for_host(self, host: str) -> str:
        """Return the PAT for a given git host, falling back to GIT_PAT."""
        return self.GIT_PAT_MAP.get(host, self.GIT_PAT)

    # --- Redis (optional) ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    # --- Knowledge limits ---
    KB_SOFT_LIMIT: int = int(os.getenv("KB_SOFT_LIMIT", "10000"))
    KB_HARD_LIMIT: int = int(os.getenv("KB_HARD_LIMIT", "15000"))
    KNOWLEDGE_COLD_DAYS: int = int(os.getenv("KNOWLEDGE_COLD_DAYS", "90"))
    SIMILARITY_MERGE_THRESHOLD: float = float(os.getenv("SIMILARITY_MERGE_THRESHOLD", "0.92"))

    # --- Scheduler ---
    SUMMARIZE_INTERVAL_MINUTES: int = int(os.getenv("SUMMARIZE_INTERVAL_MINUTES", "5"))
    HISTORY_COMPENSATE_INTERVAL_MINUTES: int = int(os.getenv("HISTORY_COMPENSATE_INTERVAL_MINUTES", "30"))
    CODE_UPDATE_INTERVAL_MINUTES: int = int(os.getenv("CODE_UPDATE_INTERVAL_MINUTES", "30"))


config = Config()
