# 数据库表设计

## 1. 概述

本文档统一管理所有 MySQL 表和 Milvus Collection 的结构定义，作为各子文档的唯一 DDL 参考来源。

## 2. MySQL 表

### 2.1 知识库

```sql
CREATE TABLE knowledge_base (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    kb_type ENUM('chat', 'code', 'manual') NOT NULL DEFAULT 'chat' COMMENT '知识库类型',
    description TEXT,
    milvus_collection VARCHAR(200) COMMENT 'Milvus collection 名称',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at DATETIME DEFAULT NULL
);
```

### 2.2 群与知识库绑定关系

```sql
CREATE TABLE chat_kb_binding (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id VARCHAR(100) NOT NULL COMMENT '飞书 chat_id',
    kb_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at DATETIME DEFAULT NULL,
    UNIQUE KEY uk_chat_kb (chat_id, kb_id, deleted_at)
);
```

### 2.3 代码库

```sql
CREATE TABLE code_repo (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    git_url VARCHAR(500) NOT NULL,
    kb_id BIGINT COMMENT '关联的知识库ID',
    default_branch VARCHAR(100) DEFAULT 'main',
    last_commit_hash VARCHAR(64),
    last_analyzed_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at DATETIME DEFAULT NULL
);
```

### 2.4 群与代码库绑定关系

```sql
CREATE TABLE chat_repo_binding (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id VARCHAR(100) NOT NULL,
    repo_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at DATETIME DEFAULT NULL,
    UNIQUE KEY uk_chat_repo (chat_id, repo_id, deleted_at)
);
```

### 2.5 群聊消息存储

```sql
CREATE TABLE chat_message (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id VARCHAR(100) NOT NULL,
    message_id VARCHAR(100) NOT NULL UNIQUE,
    parent_id VARCHAR(100) DEFAULT NULL COMMENT '飞书回复消息的 parent_message_id，用于话题分组',
    sender_id VARCHAR(100),
    user_id VARCHAR(100) DEFAULT NULL COMMENT '飞书 user_id（工号）',
    content TEXT,
    msg_type VARCHAR(20) DEFAULT 'text',
    processed BOOLEAN DEFAULT FALSE COMMENT '是否已被归纳处理',
    topic_group_id VARCHAR(100) DEFAULT NULL COMMENT '所属话题组ID（由归纳任务分配）',
    pending_count TINYINT DEFAULT 0 COMMENT '连续被判定为"进行中"的次数，达到3次强制归纳',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chat_processed (chat_id, processed),
    INDEX idx_topic_group (topic_group_id),
    INDEX idx_created (created_at)
);
```

### 2.6 手动添加的知识

```sql
CREATE TABLE manual_knowledge (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    kb_id BIGINT NOT NULL,
    chat_id VARCHAR(100) COMMENT '来源群',
    sender_id VARCHAR(100) COMMENT '添加人',
    content TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.7 授权管理员

```sql
CREATE TABLE admin_user (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sender_id VARCHAR(100) NOT NULL UNIQUE COMMENT '飞书 user_id',
    name VARCHAR(100) COMMENT '备注名',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.8 安全拦截日志

```sql
CREATE TABLE security_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id VARCHAR(100),
    sender_id VARCHAR(100),
    raw_text TEXT,
    block_reason VARCHAR(200),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.9 代码块解析记录

```sql
CREATE TABLE code_block (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    repo_id BIGINT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    block_type VARCHAR(20) NOT NULL,
    block_name VARCHAR(200),
    parent_class VARCHAR(200),
    start_line INT,
    end_line INT,
    signature TEXT,
    content_hash VARCHAR(64) COMMENT '代码内容的 SHA-256，用于检测变更',
    commit_hash VARCHAR(64),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT 'pending / success / failed',
    error_message TEXT COMMENT '失败原因',
    milvus_id VARCHAR(100) COMMENT 'Milvus 中的向量 ID',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_repo_file (repo_id, file_path),
    INDEX idx_commit (commit_hash),
    INDEX idx_repo_status (repo_id, status)
);
```

### 2.10 归纳任务执行记录

```sql
CREATE TABLE summarize_task_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    kb_id BIGINT NOT NULL,
    task_type ENUM('scheduled', 'manual', 'code') NOT NULL,
    status ENUM('running', 'success', 'failed') NOT NULL,
    message_count INT COMMENT '处理的消息数',
    new_knowledge_count INT COMMENT '新增知识数',
    updated_knowledge_count INT COMMENT '更新知识数',
    deleted_knowledge_count INT COMMENT '删除知识数',
    error_message TEXT,
    started_at DATETIME NOT NULL,
    finished_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.11 LLM 调用错误日志

```sql
CREATE TABLE summarize_error_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    kb_id BIGINT NOT NULL,
    input_text TEXT COMMENT 'LLM 输入',
    raw_output TEXT COMMENT 'LLM 原始输出',
    error_type VARCHAR(100),
    error_message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.12 群设置

```sql
CREATE TABLE chat_settings (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id VARCHAR(100) NOT NULL UNIQUE COMMENT '飞书 chat_id',
    debug_mode BOOLEAN DEFAULT FALSE COMMENT '是否开启 Debug 模式',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## 3. Milvus Collection

### 3.1 知识库 Collection

每个知识库对应一个 Milvus Collection，命名规则：`kb_{kb_id}`

```python
collection_schema = {
    "fields": [
        {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True},
        {"name": "vector", "type": "FLOAT_VECTOR", "dim": "EMBEDDING_DIM"},  # 维度以实际 Embedding 模型为准，通过配置项 EMBEDDING_DIM 指定
        {"name": "topic", "type": "VARCHAR", "max_length": 200},
        {"name": "content", "type": "VARCHAR", "max_length": 5000},
        {"name": "source", "type": "VARCHAR", "max_length": 50},       # chat / manual / code
        {"name": "source_detail", "type": "VARCHAR", "max_length": 500},
        {"name": "certainty", "type": "VARCHAR", "max_length": 20},    # confirmed / disputed / unverified
        {"name": "kb_id", "type": "INT64"},
        {"name": "last_updated_at", "type": "INT64"},                  # Unix timestamp
        {"name": "last_referenced_at", "type": "INT64"},               # Unix timestamp
    ],
    "index": {
        "field": "vector",
        "type": "IVF_FLAT",       # 或 HNSW，根据数据量选择
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
}
```

**代码知识的额外 metadata**（使用 Milvus Dynamic Field）：

`block_type` 取值：
- `function` / `method` / `class` / `file` / `chunk` — 代码块级知识
- `module_summary` — 模块（目录）级摘要
- `repo_summary` — 仓库全局概览

```python
# 代码块级
{
    "file_path": "src/services/user_service.py",
    "language": "python",
    "block_type": "method",
    "block_name": "create_user",
    "commit_hash": "abc123f"
}

# 模块摘要级
{
    "file_path": "app/services",       # 模块目录路径
    "block_type": "module_summary",
    "block_name": "app/services"
}

# 仓库概览级
{
    "file_path": ".",
    "block_type": "repo_summary",
    "block_name": "overview"
}
```

## 4. 表关系总览

```
knowledge_base (1) ←── (N) chat_kb_binding (N) ──→ (1) chat [飞书群]
knowledge_base (1) ←── (1) code_repo
code_repo      (1) ←── (N) chat_repo_binding (N) ──→ (1) chat [飞书群]
knowledge_base (1) ←── (N) manual_knowledge
knowledge_base (1) ←── (N) summarize_task_log
knowledge_base (1) ←── (N) summarize_error_log
code_repo      (1) ←── (N) code_block
chat [飞书群]   (1) ←── (N) chat_message
chat [飞书群]   (1) ←── (1) chat_settings
```
