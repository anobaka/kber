# 代码库分析系统优化方案

## 概述

本文档整合了代码库分析系统的优化建议，涵盖 Repomix 集成、Token 管理、多粒度存储、跨代码库分析等多个方面。

---

## 一、Repomix 集成优化

### 1.1 Repomix 功能概述

| 功能 | 说明 |
|------|------|
| **代码打包** | 将整个仓库打包成单个 AI 友好文件 |
| **Token 统计** | 统计每个文件和整个仓库的 token 数 |
| **代码压缩** | 使用 Tree-sitter 提取关键代码结构 |
| **安全检查** | 检测敏感信息（API Key 等） |
| **多格式输出** | XML、Markdown、JSON、Plain Text |

### 1.2 集成方案

```python
# app/services/repomix_service.py

import subprocess
import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

class RepomixService:
    """Repomix 集成服务"""
    
    def analyze_repository(self, repo_dir: str) -> dict[str, Any]:
        """
        使用 Repomix 分析仓库
        
        Returns:
            {
                "token_counts": {file_path: token_count},
                "security_findings": [...],
                "directory_structure": str,
                "files": {file_path: content},
                "summary": {total_files, total_tokens, total_chars}
            }
        """
        output_file = os.path.join(repo_dir, ".repomix-analysis.json")
        
        cmd = [
            "npx", "repomix@latest",
            "--style", "json",
            "--output", output_file,
            "--token-count-tree", "0",
        ]
        
        logger.info("Running Repomix analysis: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode != 0:
            logger.error("Repomix failed: %s", result.stderr)
            raise Exception(f"Repomix failed: {result.stderr}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        analysis = self._parse_repomix_output(raw_data)
        os.remove(output_file)
        
        return analysis
    
    def _parse_repomix_output(self, raw_data: dict) -> dict[str, Any]:
        """解析 Repomix 输出"""
        files = raw_data.get("files", {})
        token_counts = {}
        
        for file_path, content in files.items():
            token_counts[file_path] = self._count_tokens(content)
        
        return {
            "token_counts": token_counts,
            "security_findings": [],
            "directory_structure": raw_data.get("directoryStructure", ""),
            "files": files,
            "summary": {
                "total_files": len(files),
                "total_tokens": sum(token_counts.values()),
                "total_chars": sum(len(c) for c in files.values()),
            }
        }
    
    def _count_tokens(self, text: str) -> int:
        """计算 Token 数量"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(text))
        except ImportError:
            return len(text) // 4

repomix_service = RepomixService()
```

### 1.3 集成到现有流程

```python
# app/services/repo_analyzer.py

from app.services.repomix_service import repomix_service

def analyze_repo(self, repo_id: int, notify_chat_ids: list[str] | None = None) -> dict[str, int]:
    # ... Step 1: Git 同步 ...
    
    # Step 1.5: Repomix 预分析（新增）
    if config.USE_REPOMIX:
        _notify("📊 正在使用 Repomix 分析代码库", progress=True)
        repomix_analysis = repomix_service.analyze_repository(repo_dir)
        
        # 获取 Token 统计
        token_counts = repomix_analysis["token_counts"]
        
        # 获取安全检查结果
        security_findings = repomix_analysis["security_findings"]
        
        _notify("✅ Repomix 分析完成", progress=True, done=True)
    
    # ... 后续步骤使用 repomix_analysis 优化 ...
```

---

## 二、Token 管理优化

### 2.1 Token 计数服务

```python
# app/services/token_service.py

import tiktoken
import logging
from typing import Any

logger = logging.getLogger(__name__)

class TokenService:
    """Token 计数服务"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._encoding = None
    
    @property
    def encoding(self):
        if self._encoding is None:
            self._encoding = tiktoken.encoding_for_model(self.model)
        return self._encoding
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数"""
        return len(self.encoding.encode(text))
    
    def count_block_tokens(self, block: dict[str, Any]) -> int:
        """计算 AST 代码块的 Token 数"""
        code = block.get("code", "")
        return self.count_tokens(code)
    
    def get_processing_strategy(self, token_counts: dict[str, int], 
                                  thresholds: dict[str, int] | None = None) -> dict[str, str]:
        """
        根据 Token 统计决定处理策略
        
        Returns:
            {file_path: strategy}  # "normal" | "compress" | "split"
        """
        thresholds = thresholds or {
            "compress": 8000,  # 大文件阈值
            "split": 4000,     # 中等文件阈值
        }
        
        strategies = {}
        for file_path, tokens in token_counts.items():
            if tokens >= thresholds["compress"]:
                strategies[file_path] = "compress"
            elif tokens >= thresholds["split"]:
                strategies[file_path] = "split"
            else:
                strategies[file_path] = "normal"
        
        return strategies

token_service = TokenService()
```

### 2.2 Token 阈值说明

| Token 数 | 处理策略 | 说明 |
|---------|---------|------|
| < 4000 | normal | 直接处理 |
| 4000-8000 | split | 智能分割 |
| > 8000 | compress | 压缩后处理 |

### 2.3 代码压缩机制

Repomix 使用 Tree-sitter 提取代码关键结构：

```
压缩前（原始代码）          压缩后（代码骨架）
─────────────────────      ─────────────────────
完整类定义                   类名 + 方法签名
完整方法实现                 方法签名 + 参数
注释和文档                   移除
空行和格式                   移除

Token 减少：90%+
```

---

## 三、多粒度知识存储

### 3.1 存储层次

```
┌─────────────────────────────────────────────────────────────┐
│                    多粒度存储架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: 文件级知识（完整代码）                             │
│  ├── 保留完整实现细节                                        │
│  ├── 保留注释和文档                                          │
│  └── 适合：理解代码逻辑、调试、重构                          │
│                                                             │
│  Level 2: 代码块级知识（AST 分割）                           │
│  ├── 函数/类级别的精确检索                                   │
│  └── 适合：查找特定功能实现                                  │
│                                                             │
│  Level 3: 模块级知识（摘要）                                 │
│  ├── 模块功能概述                                            │
│  └── 适合：了解项目结构                                      │
│                                                             │
│  Level 4: 仓库级知识（概览）                                 │
│  ├── 项目整体架构                                            │
│  └── 适合：快速了解项目                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 文件级知识生成

```python
# app/services/repo_analyzer.py

def _generate_file_knowledge(self, repo_dir: str, file_path: str, source: str, lang: str, repo_id: int, file_commit: dict | None = None) -> dict[str, Any]:
    """生成文件级知识（作为上下文补充）"""
    
    imports = self._extract_imports(source, lang)
    
    commit_info = {}
    if file_commit:
        last_commit = file_commit.get("last_commit", {})
        commit_info = {
            "commit_author": last_commit.get("author"),
            "commit_date": last_commit.get("date"),
            "contributors": json.dumps(file_commit.get("all_contributors", []), default=str),
        }
    
    return {
        "file_path": file_path,
        "language": lang,
        "block_type": "file",
        "block_name": os.path.basename(file_path),
        "code": source,
        "imports": imports,
        "repo_id": repo_id,
        **commit_info,
    }
```

### 3.3 导入信息提取

```python
def _extract_imports(self, source: str, lang: str) -> list[str]:
    """提取文件的导入语句"""
    imports = []
    import_patterns = {
        "python": r"^(?:import|from)\s+.+$",
        "java": r"^import\s+.+;$",
        "javascript": r"^import\s+.+from\s+.+$",
        "typescript": r"^import\s+.+from\s+.+$",
        "go": r"^import\s+.+$",
    }
    
    pattern = import_patterns.get(lang, "")
    if pattern:
        for line in source.split("\n"):
            if re.match(pattern, line.strip()):
                imports.append(line.strip())
    
    return imports[:20]
```

### 3.4 LLM Prompt 优化

```python
# app/services/llm_service.py

def generate_code_knowledge(self, repo_map: str, file_path: str, block_type: str, 
    block_name: str, language: str, code: str,
    commit_author: str | None = None,
    commit_date: str | None = None,
    contributors: str | None = None,
    imports: list[str] | None = None,  # 新增参数
) -> str:
    """生成代码知识描述"""
    
    # 构建导入信息段落
    imports_section = ""
    if imports:
        imports_section = "\n## 导入依赖\n" + "\n".join(f"- {imp}" for imp in imports[:10])
    
    prompt = f"""你是一个代码文档专家。请为以下代码生成一段结构化的知识描述。

## 项目概览
{repo_map}

## 代码块
文件：{file_path}
类型：{block_type}
名称：{block_name}
{imports_section}

## 开发者信息
{dev_info_section if dev_info_section else "暂无开发者信息记录。"}

```{language}
{code}
```

## 要求
生成的描述需要包含：
1. 【功能】：这段代码做了什么（一句话概括）
2. 【详细说明】：具体的业务逻辑、算法步骤（50-200字）
3. 【输入输出】：参数和返回值说明
4. 【依赖关系】：调用了哪些其他模块/方法
5. 【关键词】：便于检索的关键词（3-5个，逗号分隔）
6. 【维护信息】：基于上述开发者信息，简要说明该代码的维护者情况
"""
    
    return self.chat([
        {"role": "system", "content": "你是一个代码文档专家。"},
        {"role": "user", "content": prompt},
    ])
```

---

## 四、双模式存储与混合检索

### 4.1 双模式存储

```python
def _store_dual_mode(self, kb_id: int, file_path: str, content: str, compressed: str):
    """双模式存储：支持精确检索和概览检索"""
    
    # 1. 完整版向量化（用于精确检索）
    full_vector = embedding_service.embed(content)
    milvus_service.insert({
        "vector": full_vector,
        "content": content,
        "file_path": file_path,
        "mode": "full",
    })
    
    # 2. 压缩版向量化（用于概览检索）
    compressed_vector = embedding_service.embed(compressed)
    milvus_service.insert({
        "vector": compressed_vector,
        "content": compressed,
        "file_path": file_path,
        "mode": "compressed",
    })
```

### 4.2 混合检索策略

```python
def hybrid_search(self, kb_id: int, query: str, top_k: int = 10) -> list[dict]:
    """混合检索：压缩版快速召回 + 完整版精确匹配"""
    
    query_vector = embedding_service.embed(query)
    
    # 1. 压缩版快速召回（Top 50）
    compressed_results = milvus_service.search(
        kb_id, query_vector, 
        top_k=50,
        filter_expr='mode == "compressed"'
    )
    
    # 2. 根据压缩版结果，获取完整版
    file_paths = [r["file_path"] for r in compressed_results[:10]]
    
    full_results = milvus_service.search(
        kb_id, query_vector,
        top_k=top_k,
        filter_expr=f'mode == "full" and file_path in {file_paths}'
    )
    
    return full_results
```

---

## 五、跨代码库业务逻辑理解

### 5.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    跨代码库分析架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: 单代码库理解（现有）                               │
│  ├── AST 解析 + 向量化                                       │
│  └── 代码块级知识                                            │
│                                                             │
│  Level 2: 跨代码库关系（新增）                               │
│  ├── API 调用关系                                            │
│  ├── 数据流向                                                │
│  └── 依赖关系图                                              │
│                                                             │
│  Level 3: 业务流程理解（新增）                               │
│  ├── 端到端流程追踪                                          │
│  └── 架构视图                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 API 调用关系分析

```python
# app/services/cross_repo_analyzer.py

class CrossRepoAnalyzer:
    """跨代码库关系分析"""
    
    def analyze_api_calls(self, kb_ids: list[int]) -> dict[str, list[dict]]:
        """分析跨代码库的 API 调用关系"""
        
        # 1. 提取所有 API 定义
        api_definitions = {}
        for kb_id in kb_ids:
            hits = milvus_service.search_by_filter(
                kb_id, 
                filter_expr='block_type == "method" and signature contains "Mapping"'
            )
            for hit in hits:
                api_path = self._extract_api_path(hit["signature"])
                if api_path:
                    api_definitions[api_path] = {
                        "kb_id": kb_id,
                        "file_path": hit["file_path"],
                        "method_name": hit["block_name"],
                    }
        
        # 2. 提取所有 API 调用
        api_calls = []
        for kb_id in kb_ids:
            hits = milvus_service.search_by_filter(
                kb_id,
                filter_expr='code contains "http://" or code contains "https://"'
            )
            for hit in hits:
                called_apis = self._extract_api_calls(hit["code"])
                for api in called_apis:
                    api_calls.append({
                        "caller_kb_id": kb_id,
                        "caller_file": hit["file_path"],
                        "api_path": api,
                    })
        
        # 3. 构建调用关系图
        return self._build_call_graph(api_definitions, api_calls)
```

### 5.3 业务流程追踪

```python
def trace_business_flow(self, kb_ids: list[int], entry_point: str) -> dict[str, Any]:
    """追踪业务流程"""
    
    # 1. 找到入口点
    entry = None
    for kb_id in kb_ids:
        hits = milvus_service.search(kb_id, embedding_service.embed(entry_point), top_k=5)
        if hits:
            entry = hits[0]
            break
    
    if not entry:
        return {"error": "Entry point not found"}
    
    # 2. 追踪调用链
    flow = {"entry": entry, "steps": []}
    visited = set()
    queue = [entry]
    
    while queue:
        current = queue.pop(0)
        if current["id"] in visited:
            continue
        visited.add(current["id"])
        
        # 分析当前代码块的调用
        calls = self._extract_method_calls(current["code"])
        for call in calls:
            for kb_id in kb_ids:
                hits = milvus_service.search(kb_id, embedding_service.embed(call), top_k=3)
                for hit in hits:
                    if hit["block_name"] == call:
                        flow["steps"].append({
                            "from": current["block_name"],
                            "to": hit["block_name"],
                            "from_file": current["file_path"],
                            "to_file": hit["file_path"],
                            "kb_id": kb_id,
                        })
                        queue.append(hit)
    
    return flow
```

### 5.4 智能问答增强

```python
def answer_with_cross_repo_context(self, chat_id: str, question: str) -> str:
    """带跨代码库上下文的问答"""
    
    kb_ids = self._get_bound_kb_ids(chat_id)
    
    # 分析问题类型
    question_type = self._analyze_question_type(question)
    
    if question_type == "flow":
        return self._answer_flow_question(kb_ids, question)
    elif question_type == "relation":
        return self._answer_relation_question(kb_ids, question)
    elif question_type == "impact":
        return self._answer_impact_question(kb_ids, question)
    else:
        return self.answer(chat_id, question)

def _analyze_question_type(self, question: str) -> str:
    """分析问题类型"""
    flow_keywords = ["流程", "调用链", "怎么执行", "如何处理"]
    relation_keywords = ["关系", "依赖", "调用", "连接"]
    impact_keywords = ["影响", "修改", "变更", "改动"]
    
    for kw in flow_keywords:
        if kw in question:
            return "flow"
    for kw in relation_keywords:
        if kw in question:
            return "relation"
    for kw in impact_keywords:
        if kw in question:
            return "impact"
    
    return "general"
```

---

## 六、配置项

```python
# app/config.py

class Config:
    # Repomix 集成
    USE_REPOMIX: bool = True
    REPOMIX_COMPRESS_THRESHOLD: int = 8000
    REPOMIX_SPLIT_THRESHOLD: int = 4000
    REPOMIX_TIMEOUT: int = 600
    
    # Token 管理
    MAX_CODE_TOKENS: int = 8000
    WARN_CODE_TOKENS: int = 4000
    
    # 双模式存储
    DUAL_MODE_STORAGE: bool = True
    
    # 混合检索
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_SEARCH_RECALL: int = 50
    
    # 跨代码库分析
    CROSS_REPO_ANALYSIS_ENABLED: bool = True
```

---

## 七、实现路线图

### Phase 1: Repomix 集成（1-2 天）

- [ ] 添加 `repomix_service.py`
- [ ] 实现 Token 统计
- [ ] 实现安全检查
- [ ] 集成到 `analyze_repo` 流程

### Phase 2: Token 管理（1-2 天）

- [ ] 添加 `token_service.py`
- [ ] 实现代码块 Token 计数
- [ ] 实现智能处理策略
- [ ] 添加大文件警告

### Phase 3: 多粒度存储（2-3 天）

- [ ] 添加文件级知识生成
- [ ] 实现导入信息提取
- [ ] 修改 LLM Prompt
- [ ] 更新数据库 Schema

### Phase 4: 双模式存储（1-2 天）

- [ ] 修改 Milvus Schema
- [ ] 实现双模式存储
- [ ] 实现混合检索
- [ ] 优化 RAG 服务

### Phase 5: 跨代码库分析（1-2 周）

- [ ] 添加 `cross_repo_analyzer.py`
- [ ] 实现 API 调用关系分析
- [ ] 实现业务流程追踪
- [ ] 实现智能问答增强

---

## 八、预期效果

| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| **大文件处理** | 超限报错 | 压缩后正常处理 |
| **Token 控制** | 无 | 智能分割 |
| **安全检查** | 简单正则 | Repomix 内置 |
| **检索精度** | 单一模式 | 双模式混合 |
| **上下文完整性** | 缺少导入信息 | 完整上下文 |
| **跨代码库理解** | 无 | API 调用关系 + 流程追踪 |

---

## 九、相关文件

| 文件 | 说明 |
|------|------|
| `app/services/repomix_service.py` | Repomix 集成服务（新增） |
| `app/services/token_service.py` | Token 计数服务（新增） |
| `app/services/cross_repo_analyzer.py` | 跨代码库分析（新增） |
| `app/services/repo_analyzer.py` | 代码库分析主逻辑（修改） |
| `app/services/rag_service.py` | RAG 检索服务（修改） |
| `app/services/llm_service.py` | LLM 调用服务（修改） |
| `app/services/milvus_service.py` | 向量存储服务（修改） |