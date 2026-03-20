# 跨代码库业务逻辑分析方案

## 一、现有代码分析流程

### 1.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    现有代码分析架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Git 同步                                           │
│  └── clone / pull 代码仓库                                  │
│                                                             │
│  Step 2: 文件扫描                                           │
│  └── 过滤扩展名、黑名单目录                                  │
│                                                             │
│  Step 3: AST 解析                                           │
│  └── Tree-sitter 解析 → 代码块（函数/类）                   │
│                                                             │
│  Step 4: 代码块同步                                         │
│  └── 检测变更、重试失败块                                    │
│                                                             │
│  Step 5: LLM 知识生成                                       │
│  └── 块级别描述 → 向量化 → 存入 Milvus                      │
│                                                             │
│  Step 6: 模块摘要                                           │
│  └── 目录级别摘要（级联更新）                                │
│                                                             │
│  Step 7: 仓库概览                                           │
│  └── 仓库级别架构描述                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 数据模型

```
┌──────────────────┐     ┌──────────────────┐
│  KnowledgeBase   │     │     CodeRepo     │
├──────────────────┤     ├──────────────────┤
│ id               │◄────│ kb_id            │
│ name             │     │ id               │
│ kb_type          │     │ git_url          │
└──────────────────┘     └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │    CodeBlock     │
                         ├──────────────────┤
                         │ id               │
                         │ repo_id          │
                         │ file_path        │
                         │ block_type       │
                         │ block_name       │
                         │ description      │
                         │ content_hash     │
                         │ commit_author    │
                         │ contributors     │
                         └──────────────────┘
```

### 1.3 现有系统的局限性

| 局限性 | 说明 |
|--------|------|
| **单仓库分析** | 每个代码仓库独立分析，无跨仓库关联 |
| **代码块粒度** | 知识粒度是函数/类级别，无业务逻辑抽象 |
| **无依赖分析** | 没有分析代码块之间的调用关系 |
| **无数据流分析** | 没有分析数据在系统中的流转 |
| **无 API 关系** | 没有分析服务间的 API 调用关系 |

---

## 二、跨代码库业务逻辑分析方案

### 方案一：跨仓库依赖关系图谱

#### 2.1.1 概念图

```
┌─────────────────────────────────────────────────────────────┐
│                  跨仓库依赖关系图谱                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    import     ┌─────────┐                     │
│  │ Repo A  │──────────────►│ Repo B  │                     │
│  │ (服务A) │               │ (公共库) │                     │
│  └────┬────┘               └─────────┘                     │
│       │                                                    │
│       │ API调用                                            │
│       ▼                                                    │
│  ┌─────────┐    import     ┌─────────┐                     │
│  │ Repo C  │──────────────►│ Repo D  │                     │
│  │ (服务B) │               │ (SDK)   │                     │
│  └─────────┘               └─────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.2 数据模型

```python
class CrossRepoDependency(Base):
    __tablename__ = "cross_repo_dependency"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_repo_id: Mapped[int] = mapped_column(BigInteger)  # 源仓库
    target_repo_id: Mapped[int] = mapped_column(BigInteger)  # 目标仓库
    dependency_type: Mapped[str]  # import, api_call, config_ref
    source_file: Mapped[str]      # 源文件
    target_module: Mapped[str]    # 目标模块
    confidence: Mapped[float]     # 置信度
```

#### 2.1.3 实现代码

```python
def analyze_cross_repo_imports(self, repo_id: int, all_repos: list[int]) -> list[dict]:
    """分析跨仓库导入关系"""
    dependencies = []
    
    # 获取当前仓库的所有导入语句
    imports = self._extract_all_imports(repo_id)
    
    # 匹配其他仓库的模块
    for other_repo_id in all_repos:
        if other_repo_id == repo_id:
            continue
        other_modules = self._get_repo_modules(other_repo_id)
        
        for imp in imports:
            for module in other_modules:
                if self._match_import(imp, module):
                    dependencies.append({
                        "source_repo_id": repo_id,
                        "target_repo_id": other_repo_id,
                        "dependency_type": "import",
                        "source_file": imp["file"],
                        "target_module": module["name"],
                    })
    
    return dependencies
```

---

### 方案二：API 接口关系映射

#### 2.2.1 概念图

```
┌─────────────────────────────────────────────────────────────┐
│                  API 接口关系映射                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    API 注册表                        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  POST /api/users        → UserService.createUser    │   │
│  │  GET  /api/users/{id}   → UserService.getUser       │   │
│  │  POST /api/orders       → OrderService.createOrder  │   │
│  │  GET  /api/orders/{id}  → OrderService.getOrder     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  API 调用关系                        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  OrderService.createOrder                            │   │
│  │    └── calls → UserService.getUser (验证用户)        │   │
│  │    └── calls → PaymentService.charge (支付)         │   │
│  │    └── calls → NotificationService.send (通知)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 数据模型

```python
class ApiEndpoint(Base):
    __tablename__ = "api_endpoint"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    repo_id: Mapped[int] = mapped_column(BigInteger)
    http_method: Mapped[str]      # GET, POST, PUT, DELETE
    path: Mapped[str]             # /api/users/{id}
    handler_block_id: Mapped[int] # 处理函数的 CodeBlock ID
    description: Mapped[str]      # API 描述
    request_schema: Mapped[str]   # 请求参数 JSON Schema
    response_schema: Mapped[str]  # 响应格式 JSON Schema
```

#### 2.2.3 框架适配器

```python
class ApiExtractor:
    """API 端点提取器 - 支持多种框架"""
    
    # FastAPI 路由检测
    FASTAPI_PATTERNS = [
        r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
    ]
    
    # Spring Boot 路由检测
    SPRING_PATTERNS = [
        r'@GetMapping\s*\(\s*["\']([^"\']+)["\']\)',
        r'@PostMapping\s*\(\s*["\']([^"\']+)["\']\)',
        r'@RequestMapping\s*\(\s*["\']([^"\']+)["\']\)',
    ]
    
    # Flask 路由检测
    FLASK_PATTERNS = [
        r'@app\.route\s*\(\s*["\']([^"\']+)["\']',
    ]
    
    # Gin (Go) 路由检测
    GIN_PATTERNS = [
        r'\.(GET|POST|PUT|DELETE)\s*\(\s*["\']([^"\']+)["\']',
    ]
    
    def extract_endpoints(self, source: str, lang: str) -> list[dict]:
        """从源码中提取 API 端点"""
        endpoints = []
        patterns = self._get_patterns(lang)
        
        for pattern in patterns:
            for match in re.finditer(pattern, source, re.IGNORECASE):
                endpoints.append({
                    "method": match.group(1).upper() if len(match.groups()) > 1 else "GET",
                    "path": match.group(2) if len(match.groups()) > 1 else match.group(1),
                })
        
        return endpoints
```

---

### 方案三：业务流程图谱

#### 2.3.1 概念图

```
┌─────────────────────────────────────────────────────────────┐
│                    业务流程图谱                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【用户注册流程】                                           │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ 用户提交  │───►│ 验证邮箱  │───►│ 创建账户  │             │
│  │ 注册表单  │    │ (发送验证)│    │ (数据库)  │             │
│  └──────────┘    └──────────┘    └────┬─────┘             │
│                                        │                   │
│                       ┌────────────────┼────────────────┐  │
│                       ▼                ▼                ▼  │
│                 ┌──────────┐    ┌──────────┐    ┌──────────┐
│                 │ 发送欢迎  │    │ 初始化    │    │ 记录日志  │
│                 │ 邮件      │    │ 用户配置  │    │ (审计)   │
│                 └──────────┘    └──────────┘    └──────────┘
│                                                             │
│  涉及服务: UserService, EmailService, ConfigService, LogService
│  涉及仓库: user-service, notification-service, common-lib
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.3.2 数据模型

```python
class BusinessFlow(Base):
    __tablename__ = "business_flow"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str]                    # 流程名称
    description: Mapped[str]             # 流程描述
    entry_point: Mapped[str]             # 入口 API
    involved_repos: Mapped[str]          # 涉及的仓库 JSON 列表
    involved_services: Mapped[str]       # 涉及的服务 JSON 列表
    flow_steps: Mapped[str]              # 流程步骤 JSON
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

#### 2.3.3 流程自动发现

```python
def discover_business_flows(self, repo_id: int) -> list[dict]:
    """从代码中自动发现业务流程"""
    flows = []
    
    # 1. 识别入口点（API 端点）
    entry_points = self._get_api_endpoints(repo_id)
    
    # 2. 追踪调用链
    for entry in entry_points:
        call_chain = self._trace_call_chain(entry["handler_block_id"])
        
        # 3. 构建流程图
        flow = {
            "name": self._infer_flow_name(entry, call_chain),
            "entry_point": f"{entry['method']} {entry['path']}",
            "flow_steps": self._build_flow_steps(call_chain),
            "involved_services": self._extract_services(call_chain),
        }
        flows.append(flow)
    
    return flows

def _trace_call_chain(self, block_id: int, visited: set = None) -> list[dict]:
    """追踪函数调用链"""
    if visited is None:
        visited = set()
    
    if block_id in visited:
        return []
    visited.add(block_id)
    
    block = self._get_block(block_id)
    calls = self._extract_function_calls(block["code"])
    
    chain = [{"block_id": block_id, "name": block["block_name"]}]
    
    for call in calls:
        called_block = self._find_block_by_name(call["name"])
        if called_block:
            sub_chain = self._trace_call_chain(called_block["id"], visited)
            chain.extend(sub_chain)
    
    return chain
```

---

### 方案四：数据模型关联分析

#### 2.4.1 概念图

```
┌─────────────────────────────────────────────────────────────┐
│                  数据模型关联分析                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    数据模型图谱                      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │   ┌──────────┐         ┌──────────┐                │   │
│  │   │  User    │────────►│  Order   │                │   │
│  │   │ (用户)   │  1:N    │ (订单)   │                │   │
│  │   └──────────┘         └────┬─────┘                │   │
│  │        │                    │                      │   │
│  │        │ 1:N                │ N:1                  │   │
│  │        ▼                    ▼                      │   │
│  │   ┌──────────┐         ┌──────────┐                │   │
│  │   │ Address  │         │ Product  │                │   │
│  │   │ (地址)   │         │ (商品)   │                │   │
│  │   └──────────┘         └──────────┘                │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  跨仓库数据流:                                              │
│  user-service.User → order-service.Order → inventory-service.Product
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 数据模型

```python
class DataModel(Base):
    __tablename__ = "data_model"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    repo_id: Mapped[int] = mapped_column(BigInteger)
    model_name: Mapped[str]           # User, Order, Product
    model_type: Mapped[str]           # entity, dto, vo
    fields: Mapped[str]               # JSON: 字段定义
    relations: Mapped[str]            # JSON: 关联关系
    source_block_id: Mapped[int]      # 来源代码块
```

#### 2.4.3 模型关系发现

```python
def analyze_data_models(self, repo_id: int) -> list[dict]:
    """分析数据模型及其关系"""
    models = []
    
    # 提取类定义
    classes = self._get_class_blocks(repo_id)
    
    for cls in classes:
        model = {
            "name": cls["block_name"],
            "fields": self._extract_fields(cls["code"]),
            "relations": self._extract_relations(cls["code"]),
        }
        
        # 检测 ORM 注解
        if self._has_orm_annotations(cls["code"]):
            model["model_type"] = "entity"
        elif self._is_dto_pattern(cls["block_name"]):
            model["model_type"] = "dto"
        
        models.append(model)
    
    return models

def _extract_relations(self, code: str) -> list[dict]:
    """提取模型关联关系"""
    relations = []
    
    # JPA/Hibernate 注解
    jpa_patterns = [
        (r'@OneToMany.*\w+\s+(\w+);', 'one_to_many'),
        (r'@ManyToOne.*\w+\s+(\w+);', 'many_to_one'),
        (r'@ManyToMany.*\w+\s+(\w+);', 'many_to_many'),
    ]
    
    # SQLAlchemy 关系
    sqlalchemy_patterns = [
        (r'relationship\(["\'](\w+)["\']', 'relationship'),
    ]
    
    # GORM (Go) 关系
    gorm_patterns = [
        (r'gorm:"foreignKey:(\w+)"', 'has_many'),
        (r'gorm:"references:(\w+)"', 'belongs_to'),
    ]
    
    for pattern, rel_type in jpa_patterns + sqlalchemy_patterns + gorm_patterns:
        for match in re.finditer(pattern, code):
            relations.append({
                "target_model": match.group(1),
                "type": rel_type,
            })
    
    return relations
```

---

### 方案五：服务调用链路追踪

#### 2.5.1 概念图

```
┌─────────────────────────────────────────────────────────────┐
│                  服务调用链路追踪                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【创建订单】调用链路                                       │
│                                                             │
│  ┌─────────────┐    HTTP     ┌─────────────┐              │
│  │ API Gateway │────────────►│ Order Svc   │              │
│  │   :8080     │             │   :8001     │              │
│  └─────────────┘             └──────┬──────┘              │
│                                     │                      │
│              ┌──────────────────────┼──────────────────┐   │
│              │                      │                  │   │
│              ▼                      ▼                  ▼   │
│       ┌─────────────┐       ┌─────────────┐    ┌─────────────┐
│       │ User Svc    │       │ Payment Svc │    │ Inventory   │
│       │   :8002     │       │   :8003     │    │ Svc :8004   │
│       └─────────────┘       └─────────────┘    └─────────────┘
│              │                      │                  │   │
│              ▼                      ▼                  ▼   │
│       ┌─────────────┐       ┌─────────────┐    ┌─────────────┐
│       │ PostgreSQL  │       │ Stripe API  │    │ Redis       │
│       │ (用户数据)  │       │ (支付网关)  │    │ (库存缓存)  │
│       └─────────────┘       └─────────────┘    └─────────────┘
│                                                             │
│  总耗时: 245ms | 关键路径: Order → Payment → Stripe        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.5.2 数据模型

```python
class ServiceRegistry(Base):
    __tablename__ = "service_registry"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    repo_id: Mapped[int] = mapped_column(BigInteger)
    service_name: Mapped[str]          # order-service
    service_type: Mapped[str]          # http, grpc, mq
    base_url: Mapped[str]              # http://order-service:8001
    health_check_path: Mapped[str]     # /health
    dependencies: Mapped[str]          # JSON: 依赖的其他服务

class ServiceCall(Base):
    __tablename__ = "service_call"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_repo_id: Mapped[int]        # 调用方仓库
    target_repo_id: Mapped[int]        # 被调用方仓库
    source_block_id: Mapped[int]       # 调用方代码块
    call_type: Mapped[str]             # http, grpc, mq
    target_url: Mapped[str]            # 目标 URL
    method: Mapped[str]                # HTTP 方法
```

#### 2.5.3 调用链分析

```python
def analyze_service_calls(self, repo_id: int) -> list[dict]:
    """分析服务间调用关系"""
    calls = []
    
    # HTTP 客户端调用检测
    http_patterns = [
        # Python requests
        r'requests\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        # JavaScript axios
        r'axios\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        # JavaScript fetch
        r'fetch\s*\(\s*["\']([^"\']+)["\']',
        # Java RestTemplate
        r'restTemplate\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        # Java Feign
        r'@FeignClient\s*\(\s*[^)]*name\s*=\s*["\']([^"\']+)["\']',
        # Go http.Client
        r'http\.Get\s*\(\s*["\']([^"\']+)["\']',
        r'http\.Post\s*\(\s*["\']([^"\']+)["\']',
    ]
    
    # RPC 调用检测
    rpc_patterns = [
        # gRPC
        r'grpc\.Dial\s*\(\s*["\']([^"\']+)["\']',
        # Dubbo
        r'@DubboReference\s*\(\s*[^)]*interface\s*=\s*["\']([^"\']+)["\']',
    ]
    
    # 消息队列调用检测
    mq_patterns = [
        # Kafka
        r'kafka\.Producer|kafka\.Consumer',
        # RabbitMQ
        r'amqp\.publish|amqp\.consume',
        # Redis Pub/Sub
        r'redis\.publish|redis\.subscribe',
    ]
    
    blocks = self._get_all_blocks(repo_id)
    
    for block in blocks:
        for pattern in http_patterns:
            for match in re.finditer(pattern, block["code"]):
                calls.append({
                    "source_block_id": block["id"],
                    "source_service": self._get_service_name(repo_id),
                    "target_url": match.group(2) if len(match.groups()) > 1 else match.group(1),
                    "method": match.group(1).upper() if len(match.groups()) > 1 else "GET",
                    "call_type": "http",
                })
    
    return calls
```

---

## 三、实施建议

### 3.1 优先级排序

| 优先级 | 方案 | 价值 | 实现难度 | 预计工期 |
|--------|------|------|----------|----------|
| **P0** | API 接口关系映射 | 高 | 中 | 1-2 周 |
| **P1** | 服务调用链路追踪 | 高 | 中 | 2-3 周 |
| **P2** | 跨仓库依赖关系图谱 | 中 | 低 | 2-3 周 |
| **P3** | 数据模型关联分析 | 中 | 中 | 3-4 周 |
| **P4** | 业务流程图谱 | 高 | 高 | 4-6 周 |

### 3.2 实施路线图

```
Phase 1 (1-2周): API 接口关系映射
├── 新增 api_endpoint 表
├── 实现多框架 API 提取器
│   ├── FastAPI (Python)
│   ├── Flask (Python)
│   ├── Spring Boot (Java)
│   ├── Gin (Go)
│   └── Express (JavaScript)
└── 集成到现有分析流程

Phase 2 (2-3周): 服务调用链路追踪
├── 新增 service_registry 表
├── 新增 service_call 表
├── 实现 HTTP/RPC 调用检测
│   ├── HTTP 客户端 (requests, axios, fetch, RestTemplate)
│   ├── RPC 框架 (gRPC, Dubbo)
│   └── 消息队列 (Kafka, RabbitMQ)
└── 构建调用链可视化

Phase 3 (2-3周): 跨仓库依赖分析
├── 新增 cross_repo_dependency 表
├── 实现导入关系分析
│   ├── Python import
│   ├── Java import
│   ├── Go import
│   └── JavaScript import
└── 构建依赖图谱

Phase 4 (3-4周): 数据模型分析
├── 新增 data_model 表
├── 实现 ORM 关系提取
│   ├── SQLAlchemy (Python)
│   ├── JPA/Hibernate (Java)
│   ├── GORM (Go)
│   └── TypeORM (JavaScript)
└── 构建数据模型图谱

Phase 5 (4-6周): 业务流程发现
├── 新增 business_flow 表
├── 实现调用链追踪
├── 自动发现业务流程
└── 构建流程可视化
```

### 3.3 技术选型建议

| 功能 | 推荐技术 | 说明 |
|------|----------|------|
| 图数据库 | Neo4j / ArangoDB | 存储关系图谱 |
| 图可视化 | D3.js / Cytoscape.js | 前端展示 |
| 流程图 | Mermaid / PlantUML | 文档生成 |
| AST 解析 | Tree-sitter | 已有基础 |

### 3.4 预期收益

1. **提升代码理解效率**
   - 快速定位跨服务调用关系
   - 自动生成系统架构文档

2. **降低维护成本**
   - 变更影响分析
   - 依赖冲突检测

3. **增强知识库能力**
   - 支持跨仓库问答
   - 业务流程自动发现

4. **辅助架构治理**
   - 服务依赖可视化
   - 循环依赖检测
   - API 版本管理

---

## 四、附录

### 4.1 支持的框架列表

#### API 框架

| 语言 | 框架 | 路由注解 |
|------|------|----------|
| Python | FastAPI | `@app.get()`, `@router.post()` |
| Python | Flask | `@app.route()` |
| Python | Django | `urlpatterns` |
| Java | Spring Boot | `@GetMapping`, `@PostMapping`, `@RequestMapping` |
| Java | JAX-RS | `@GET`, `@POST`, `@Path` |
| Go | Gin | `r.GET()`, `r.POST()` |
| Go | Echo | `e.GET()`, `e.POST()` |
| JavaScript | Express | `app.get()`, `app.post()` |
| JavaScript | NestJS | `@Get()`, `@Post()` |

#### ORM 框架

| 语言 | 框架 | 关系注解 |
|------|------|----------|
| Python | SQLAlchemy | `relationship()`, `ForeignKey` |
| Python | Django ORM | `ForeignKey`, `ManyToManyField` |
| Java | JPA/Hibernate | `@OneToMany`, `@ManyToOne`, `@ManyToMany` |
| Java | MyBatis | XML 配置 |
| Go | GORM | `gorm:"foreignKey"` |
| JavaScript | TypeORM | `@OneToMany`, `@ManyToOne` |
| JavaScript | Prisma | Schema 文件 |

#### RPC 框架

| 语言 | 框架 | 调用方式 |
|------|------|----------|
| 多语言 | gRPC | `grpc.Dial()` |
| Java | Dubbo | `@DubboReference` |
| Java | Feign | `@FeignClient` |
| Go | RPC | `rpc.Dial()` |

### 4.2 参考资料

- [Tree-sitter 官方文档](https://tree-sitter.github.io/tree-sitter/)
- [Neo4j 图数据库](https://neo4j.com/docs/)
- [Secretlint 安全检查](https://github.com/secretlint/secretlint)
- [Repomix 代码打包](https://github.com/yamadashy/repomix)