# 代码库分析文档

## 1. 概述

本文档描述如何将 Git 代码库解析、归纳为结构化知识，并存储到 Milvus 向量库中，使其可被 RAG 系统检索。整个流程设计为异步数据处理流水线，支持增量更新和知识去膨胀。

## 2. 架构总览

整个流水线分为五个阶段：

```
拉取与降噪 → AST 结构化解析 → LLM 归纳总结 → 向量化 → 存储与更新
```

## 3. 阶段一：代码获取与降噪 (Ingestion & Filtering)

### 3.1 仓库拉取

- 首次分析：完整克隆（`git clone`）到本地目录（不使用 `--depth 1`，因为后续增量更新需要完整历史来执行 diff）
- 增量更新：`git fetch origin` 获取最新代码，通过 `git diff {last_commit_hash}..origin/{branch} --name-status` 确定变更文件列表，再 `git pull` 合并
- 克隆目录：`/data/repos/{repo_id}/`
- 记录当前 `commit_hash` 到 `code_repo.last_commit_hash`
- **私有仓库认证**：从应用配置中读取 Personal Access Token (PAT)，自动注入到 HTTPS URL 中（`https://{PAT}@gitlab.myco.com/...`）。配置项：`GIT_BASE_URL`（GitLab 地址，如 `https://gitlab.mycompany.com`）、`GIT_PAT`（访问令牌）。用户绑定时只需提供 `org/repo` 短路径，系统自动拼接完整 URL

### 3.2 文件过滤

**白名单策略**（仅处理以下类型文件）：

```
*.py, *.js, *.ts, *.tsx, *.jsx, *.java, *.go, *.rs,
*.cs, *.cpp, *.c, *.h, *.rb, *.php, *.swift, *.kt,
*.scala, *.vue, *.sql, *.proto, *.graphql
```

**黑名单排除**：

- 遵循 `.gitignore` 规则（使用 pathspec 库解析）
- 额外排除：`node_modules/`, `vendor/`, `dist/`, `build/`, `target/`, `.git/`, `__pycache__/`, 测试 mock 数据文件, 生成代码目录
- 排除大文件：单文件 > 100KB 跳过
- 排除 minified 文件：单行超过 500 字符的文件跳过

### 3.3 敏感信息检测

在发送代码给 LLM 之前：

- 使用正则匹配检测硬编码的 API Key、密码、Token（常见模式：`password=`, `api_key=`, `secret`, `Bearer` 等）
- 检测到敏感信息的行，替换为 `[REDACTED]`
- 可选：集成 `gitleaks` 做更全面的扫描

## 4. 阶段二：结构化提取与 AST 分块 (AST Parsing & Chunking)

### 4.1 AST 解析工具

使用 **Tree-sitter** 将源码解析为 AST，按语义单元提取代码块。

**不能按字数/行数切分代码**，必须按语法结构切分。

### 4.2 提取粒度

按以下优先级提取代码块：

| 层级 | 提取单元 | 示例 |
|------|----------|------|
| 1 | 类/结构体定义 | `class UserService`, `struct Config` |
| 2 | 方法/函数定义 | `def process_order()`, `func HandleRequest()` |
| 3 | 模块级常量/配置 | `const API_ENDPOINTS = {...}` |
| 4 | 接口/类型定义 | `interface IUserRepo`, `type Props = {...}` |

### 4.3 元数据保留

每个代码块需保留以下元数据：

```json
{
  "file_path": "src/services/user_service.py",
  "language": "python",
  "block_type": "method",          // class / method / function / interface / constant
  "block_name": "create_user",
  "parent_class": "UserService",   // 方法所属类名，顶层函数为 null
  "start_line": 42,
  "end_line": 78,
  "signature": "def create_user(self, name: str, email: str) -> User",
  "imports": ["from models import User", "from utils import validate_email"],
  "repo_id": 1
}
```

### 4.4 上下文增强

提取方法时，自动附带：

- 所属类的定义签名（不含方法体）
- 相关 import 语句
- 紧邻的注释/docstring

## 5. 阶段三：LLM 归纳总结 (Contextualization & Summarization)

### 5.1 仓库地图生成 (Repo Map)

在开始归纳前，生成一份轻量级的仓库结构概览：

```
项目结构：
src/
  ├── services/         # 业务服务层
  │   ├── user_service.py    (UserService: create_user, update_user, delete_user)
  │   └── order_service.py   (OrderService: create_order, cancel_order)
  ├── models/           # 数据模型
  │   ├── user.py
  │   └── order.py
  └── api/              # API 路由
      └── routes.py

主要依赖：Flask, SQLAlchemy, Redis
```

### 5.2 LLM 归纳 Prompt

对每个代码块调用阿里云百炼 API（minimax-m2.5）：

```
你是一个代码文档专家。请为以下代码生成一段结构化的知识描述。

## 项目概览
{Repo Map}

## 代码块
文件：{file_path}
类型：{block_type}
名称：{block_name}

```{language}
{代码内容}
```

## 要求
生成的描述需要包含：
1. 【功能】：这段代码做了什么（一句话概括）
2. 【详细说明】：具体的业务逻辑、算法步骤（50-200字）
3. 【输入输出】：参数和返回值说明
4. 【依赖关系】：调用了哪些其他模块/方法
5. 【关键词】：便于检索的关键词（3-5个，逗号分隔）

注意：描述要面向"检索场景"优化，让开发者通过自然语言提问时能找到这段代码。
```

### 5.3 批量处理策略

- 并发度：每个代码库最多 5 个并发 LLM 调用
- 速率限制：遵守阿里云百炼 API 的 QPS 限制（默认 10 QPS）
- 批次大小：每批处理 50 个代码块
- 失败重试：单个代码块失败重试 2 次，仍失败则跳过并记录日志

## 6. 阶段四：向量化 (Embedding)

### 6.1 Embedding 内容

将 LLM 生成的知识描述（而非原始代码）进行 Embedding，因为自然语言描述更适合与用户的自然语言提问进行语义匹配。

### 6.2 Embedding 模型

与消息归纳使用相同的 Embedding 模型（阿里云百炼 Embedding API 或自部署 BGE-M3），保证全系统向量空间一致。

## 7. 阶段五：存储与增量更新

### 7.1 Milvus 存储结构

代码知识存储在对应知识库的 Collection 中（与聊天知识共用 Collection，Schema 见 [schema.md](schema.md) 第 3 节），通过 `source=code` 区分。

代码知识的额外 Dynamic Field 包括 `file_path`、`language`、`block_type`、`block_name`、`commit_hash`，用于按文件路径/语言过滤检索。

### 7.2 增量更新策略

代码库的知识更新采用 **diff-based 增量更新**：

**触发时机**：
- 绑定代码库时：全量首次分析
- 定时任务：每 30 分钟检查一次各代码库的远端更新（`git fetch` + `git diff`）
- 手动触发：用户可通过命令触发

**更新流程**：

```
1. git fetch origin
2. git diff {last_commit_hash}..origin/{branch} --name-status
3. 对变更文件分类：
   - A (新增文件)：完整解析 → 归纳 → 插入 Milvus
   - M (修改文件)：重新解析 → 归纳 → 替换 Milvus 中该文件的旧知识
   - D (删除文件)：从 Milvus 中删除该文件对应的所有知识向量
4. git pull
5. 更新 code_repo.last_commit_hash
6. 更新 code_repo.last_analyzed_at
```

### 7.3 知识去膨胀（代码库特有）

代码知识的去膨胀比聊天知识更可控：

- **文件级替换**：修改文件时，整个文件的旧知识全部删除，重新生成（因为代码块可能被重构，旧知识已失效）
- **清理孤儿知识**：每次全量分析后，检查 Milvus 中是否存在已不在仓库中的文件路径对应的知识，清除之
- **Repo Map 更新**：每次分析后重新生成 Repo Map，确保全局视图一致

## 8. 定时任务设计

### 8.1 代码库更新检查任务

- **频率**：每 30 分钟
- **并发**：每个代码库独立执行，互不阻塞
- **流程**：
  1. 遍历所有未删除的 `code_repo` 记录
  2. 对每个仓库执行 `git fetch`，比较 commit hash
  3. 若有更新，执行增量分析流程
  4. 记录任务日志到 `summarize_task_log`（`task_type=code`）

### 8.2 全量重建任务（可选）

- **频率**：每周一次（周日凌晨）
- **目的**：修复可能的增量遗漏，清理孤儿知识
- **流程**：全量重新解析仓库，与 Milvus 中的现有知识做 diff，更新差异部分

## 9. 相关数据库表

本模块涉及的表（DDL 统一见 [schema.md](schema.md)）：

| 表名 | 用途 | schema.md 章节 |
|------|------|----------------|
| `code_repo` | 代码库元信息 | 2.3 |
| `code_block` | 代码块解析记录 | 2.9 |
| `summarize_task_log` | 归纳任务执行记录（task_type=code） | 2.10 |

## 10. 错误处理

- **Git 操作失败**：记录错误，不影响其他仓库的处理。重试 3 次后发送告警。
- **AST 解析失败**：跳过解析失败的文件，记录日志。部分语言的语法扩展可能不被 Tree-sitter 支持。
- **LLM 调用失败**：同 analyze-messages.md 的容错策略。
- **Milvus 写入失败**：重试 3 次，失败后暂存到本地队列，下次任务时重新写入。
