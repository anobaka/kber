# KBER - 企业级知识库管理系统

通过飞书机器人作为统一交互入口，自动采集群聊消息和代码仓库内容，经 LLM 归纳总结后存入向量数据库，提供智能知识检索与问答能力。

## 系统架构

```
┌──────────────────────────────────────────┐
│           飞书机器人 (入口)                │
│     长连接接收消息 / 命令解析 / 消息回复    │
└─────┬──────────────┬──────────────┬──────┘
      │              │              │
┌─────▼─────┐  ┌─────▼─────┐  ┌───▼───┐
│  命令处理  │  │  消息采集  │  │  RAG  │
│ 绑定/解绑  │  │ 实时+补偿  │  │ 检索  │
└─────┬─────┘  └─────┬─────┘  └───┬───┘
      │              │             │
┌─────▼──────────────▼─────┐  ┌───▼────┐
│         MySQL            │  │ Milvus │
│  业务数据 / 消息存储      │  │ 向量库  │
└─────┬──────────┬─────────┘  └───▲────┘
      │          │                │
┌─────▼─────┐ ┌──▼──────────┐    │
│ 代码分析   │ │ 消息归纳    │────┘
│ AST + LLM │ │ LLM 总结    │
└───────────┘ └─────────────┘
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 交互入口 | 飞书开放平台（WebSocket 长连接） |
| 后端服务 | Python 3.11+ |
| 关系型数据库 | MySQL 8.0 |
| 向量数据库 | Milvus 2.4 |
| LLM 推理 | 阿里云百炼 API（OpenAI 兼容接口） |
| Embedding | 阿里云百炼 Embedding API |
| AST 解析 | Tree-sitter |
| 定时任务 | APScheduler |
| DB 迁移 | Alembic |
| 容器化 | Docker + Docker Compose |

## 项目结构

```
kber/
├── main.py                          # 主入口
├── alembic/                         # 数据库迁移
│   ├── env.py
│   └── versions/                    # 迁移脚本（自动生成）
├── app/
│   ├── config.py                    # 配置管理（环境变量）
│   ├── db/
│   │   ├── models.py                # SQLAlchemy ORM 模型
│   │   └── session.py               # 数据库连接管理
│   ├── services/
│   │   ├── milvus_service.py        # Milvus 向量库操作
│   │   ├── llm_service.py           # LLM 调用封装
│   │   ├── embedding_service.py     # 文本向量化
│   │   ├── security_service.py      # Prompt 安全防护
│   │   ├── message_analyzer.py      # 消息归纳（话题分组 → 知识提取）
│   │   ├── repo_analyzer.py         # 代码库分析（AST → 知识生成）
│   │   └── rag_service.py           # RAG 检索与回答
│   ├── bot/
│   │   ├── handler.py               # 飞书 Bot 消息收发
│   │   └── commands.py              # 命令路由与处理
│   └── scheduler/
│       └── tasks.py                 # 定时任务调度
├── docs/                            # 设计文档
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## 快速开始

### Docker 部署（推荐）

1. **配置环境变量**

```bash
cp .env.example .env
# 编辑 .env，填写飞书 App ID/Secret 和 LLM API Key
```

2. **启动所有服务**

```bash
docker compose up -d
```

这会启动 MySQL、Milvus（含 etcd + MinIO）、Redis 以及 KBER 应用。

3. **查看日志**

```bash
docker compose logs -f kber
```

### 本地开发

1. **前置依赖**

确保本地已运行 MySQL 8.0、Milvus 2.4、Redis（可选）。

2. **安装 Python 依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

```bash
cp .env.example .env
# 编辑 .env
```

4. **初始化数据库**

```bash
# 生成首次迁移脚本
alembic revision --autogenerate -m "initial schema"
# 执行迁移
alembic upgrade head
```

5. **启动服务**

```bash
python main.py
```

## 数据库迁移

项目使用 [Alembic](https://alembic.sqlalchemy.org/) 管理数据库 Schema 变更。

```bash
# 修改 app/db/models.py 后，生成迁移脚本
alembic revision --autogenerate -m "描述本次变更"

# 查看当前迁移状态
alembic current

# 应用所有待执行的迁移
alembic upgrade head

# 回退一个版本
alembic downgrade -1

# 查看迁移历史
alembic history
```

> 应用启动时会自动执行 `alembic upgrade head`，确保数据库 Schema 始终是最新的。

## 命令列表

在飞书群中 @机器人 发送以下命令：

| 命令 | 说明 |
|------|------|
| `绑定知识库 {名称}` | 将本群聊天记录纳入指定知识库 |
| `解绑知识库 {名称}` | 解除本群与知识库的绑定 |
| `绑定代码库 {org/repo 或 完整URL}` | 关联代码库并自动分析 |
| `解绑代码库 {org/repo 或 完整URL}` | 解除代码库关联 |
| `添加知识 [知识库名称] {内容}` | 手动添加知识（单知识库时可省略名称） |
| `我的ID` | 获取你的用户 ID（用于申请管理员权限） |
| `帮助` | 显示帮助信息 |
| `立即总结` | 立即触发知识归纳（管理员） |
| `查询知识库` | 查看所有知识库状态（管理员） |
| `重建知识库 {名称}` | 清除并重建指定知识库（管理员） |
| `停止构建 {名称}` | 停止正在构建的知识库任务（管理员） |
| `enable-debug` | 开启本群 Debug 模式，显示详细工作进度（管理员） |
| `disable-debug` | 关闭本群 Debug 模式（管理员） |
| 直接提问 | 基于知识库的 RAG 问答 |

## 定时任务

| 任务 | 频率 | 说明 |
|------|------|------|
| 消息归纳 | 每 5 分钟 | 对未处理消息做话题分组、状态判定、知识提取 |
| 历史消息补偿 | 每 30 分钟 | 拉取最近消息补偿漏收 |
| 代码库更新检查 | 每 30 分钟 | git fetch + diff 增量更新代码知识 |

## 设计文档

详细设计参见 `docs/` 目录：

- [all-in-one.md](docs/all-in-one.md) — 总体设计
- [channel.md](docs/channel.md) — 飞书机器人交互
- [analyze-messages.md](docs/analyze-messages.md) — 聊天记录分析
- [analyze-repo.md](docs/analyze-repo.md) — 代码库分析
- [schema.md](docs/schema.md) — 数据库 Schema

## License

MIT
