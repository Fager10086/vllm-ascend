# MindIE-PyMotor 安全威胁分析报告

**版本**：1.0  
**日期**：2026-06-12  
**分支**：master  
**分析方法**：白盒（源码审计 + 知识图谱调用链分析）

---

## 一、系统架构概述

MindIE-PyMotor 是华为 MindIE 平台的分布式大模型推理编排层，实现了 Prefill/Decode 分离（PD Disaggregation）、KV Cache 亲和调度、多节点容错等特性，对外暴露兼容 OpenAI/Anthropic 的 HTTP 推理接口。

### 主要组件与信任边界

```
[外部客户端]
      │ HTTP/HTTPS  (TB1 — 外部信任边界)
      ▼
[Coordinator / InferenceServer]   motor/coordinator/api_server/inference_server.py
      │
      ├── 内部 HTTP ──► [Router / Dispatch]      motor/coordinator/router/dispatch.py
      │
      ├── ZMQ (无认证) ► [Scheduler]             motor/coordinator/scheduler/
      │
      └── gRPC (可选mTLS) ► [Etcd]              motor/common/etcd/etcd_client.py

[NodeManager API]   motor/node_manager/api_server/node_manager_api.py
      │ HTTP（无认证）
      └── 控制 [EngineServer (vLLM/SGLang)]

[Controller]   motor/controller/api_server/controller_api.py
      │ 内部 HTTP（路径 ACL）
      └── 可观测性 / 容错编排
```

---

## 二、威胁汇总

| ID | 威胁 | 受影响组件 | STRIDE 类型 | 风险等级 |
|---|---|---|---|---|
| T01 | API Key 认证默认关闭 | InferenceServer | Spoofing | HIGH |
| T02 | API Key 明文短路检查存在时序侧信道 | InferenceServer | Information Disclosure | MEDIUM |
| T03 | NodeManager API 无任何认证 | NodeManager | Spoofing / DoS / EoP | **CRITICAL** |
| T04 | 限速器进程级隔离，多副本部署失效 | InferenceServer | DoS | HIGH |
| T05 | 限速中间件异常时 Fail-Open 放行 | InferenceServer | DoS | MEDIUM |
| T06 | Etcd 通信可降级为明文 gRPC | EtcdClient | Tampering / Info Disclosure | HIGH |
| T07 | Etcd 命名空间来自环境变量，可被污染 | EtcdClient | Tampering / EoP | MEDIUM |
| T08 | OLC 配置路径写入环境变量前无校验 | InferenceServer | Tampering | MEDIUM |
| T09 | 审计日志无法区分独立用户身份 | security_utils | Repudiation | MEDIUM |
| T10 | Scheduler ZMQ 传输无对端认证 | Scheduler | Spoofing / Tampering | MEDIUM |
| T11 | SSL Context 对象携带明文私钥密码 | CertUtil | Information Disclosure | MEDIUM |
| T12 | 路径遍历防护遗漏二次 URL 编码 | security_utils | Tampering | LOW |
| T13 | 敏感字段过滤依赖字段名黑名单 | security_utils | Information Disclosure | LOW |

---

## 三、威胁详细分析

### T01 — API Key 认证默认关闭

**风险等级**：HIGH  
**代码位置**：`motor/coordinator/api_server/inference_server.py:172-174`

```python
def verify_api_key(self, request: Request) -> None:
    if not self._api_key_config.enable_api_key:  # 默认 False
        return                                    # 直接放行，无需任何凭据
```

`APIKeyConfig.enable_api_key` 默认值为 `False`（`motor/config/coordinator.py:232`）。部署时若未显式将其设为 `True`，`/v1/completions`、`/v1/chat/completions`、`/anthropic/messages` 等所有推理接口对任意调用方完全开放。

**影响**：任何能访问 Coordinator 端口的客户端均可无限调用推理接口，消耗 GPU 算力，并获取模型输出。

---

### T02 — API Key 验证的时序侧信道

**风险等级**：MEDIUM  
**代码位置**：`motor/coordinator/api_server/inference_server.py:188-190`

```python
if api_key in self._api_key_config.valid_keys:   # ← 明文集合 O(n) 查找，非恒定时间
    return
if verify_api_key_against_valid_keys(api_key, self._api_key_config.valid_keys):  # PBKDF2
    return
```

第 188 行使用 Python `in` 操作符对 `valid_keys` 集合进行短路查找。若配置错误导致 `valid_keys` 中混入明文 Key，则命中路径完全绕过 PBKDF2 哈希验证，且 `in` 操作符不保证恒定时间，理论上存在时序侧信道。

**影响**：配置错误时降级为明文比较；对于大型 `valid_keys` 列表，响应时间差异可辅助枚举。

---

### T03 — NodeManager API 无任何认证（最高优先级）

**风险等级**：CRITICAL  
**代码位置**：`motor/node_manager/api_server/node_manager_api.py:46-176`

```python
app = FastAPI(lifespan=lifespan)  # 无任何认证 Middleware 或 Depends

@app.post("/node-manager/stop")   # 无保护
async def stop_instance(request: Request):
    await asyncio.to_thread(Daemon().stop)  # 停止所有推理引擎进程
```

route_map 工具输出确认所有 5 个路由（`start`、`stop`、`pause`、`resume`、`status`）的 `middleware: []`，即无任何中间件保护。

NodeManager 默认绑定 `0.0.0.0`（`node_manager_api.py:186`），在无网络隔离的情况下，集群内任意节点均可访问。

**影响分析**：

| 路由 | 攻击后果 |
|---|---|
| `POST /node-manager/stop` | 停止所有推理引擎进程，全节点服务中断 |
| `POST /node-manager/pause` | 将所有 endpoint 置为 PAUSED，触发 Controller 暂停流程 |
| `POST /node-manager/start` | 传入伪造的 `endpoints` 参数，影响引擎注册信息 |
| `GET /node-manager/status` | 泄露 endpoint 状态信息 |

**叠加风险**：当 `mgmt_tls_config.enable_tls=False` 时（`node_manager_api.py:231`），NodeManager 以 HTTP 启动，无认证 + 无加密双重缺失。

---

### T04 — 限速器进程级隔离，多副本部署失效

**风险等级**：HIGH  
**代码位置**：`motor/coordinator/middleware/fastapi_middleware.py:128-265`，`motor/coordinator/api_server/inference_server.py:217-237`

`SimpleRateLimiter` 使用进程内内存字典维护请求计数，多个 Coordinator Worker 进程之间不共享限速状态。以 `max_requests=100/min`、3 个副本为例，实际可达 300 req/min。

```python
class SimpleRateLimiter:
    # 内存字典，进程私有，不跨进程/副本共享
```

此外，OLC 提供者仅在 `olc` 库可用时生效，失败时静默降级为简单限速（`inference_server.py:212-213`），降级行为无外部告警。

**影响**：大规模并发攻击下，分布式限速形同虚设，GPU 资源可被耗尽。

---

### T05 — 限速中间件异常时 Fail-Open

**风险等级**：MEDIUM  
**代码位置**：`motor/coordinator/middleware/fastapi_middleware.py:241-245`

```python
except Exception as e:
    logger.error(f"Error in rate limiting middleware processing request: {e}")
    # Allow request by default when error occurs  ← 明确 Fail-Open 设计
    self.stats["allowed_requests"] += 1
    return await call_next(request)
```

任何导致 `rate_limiter.is_allowed()` 抛出异常的条件均会使限速完全失效，请求被放行。代码注释本身说明这是有意为之的设计选择，但在安全场景下属于不安全的默认行为。

---

### T06 — Etcd 通信可降级为明文 gRPC

**风险等级**：HIGH  
**代码位置**：`motor/common/etcd/etcd_client.py:54-69`

```python
if self.tls_config and self.tls_config.enable_tls:
    ...
    self.channel = grpc.secure_channel(...)
else:
    self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')  # 明文降级
```

当 `tls_config` 未配置或 `enable_tls=False` 时，所有 Etcd 通信以明文 gRPC 进行，不产生任何警告日志。

Etcd 中存储的数据包括：引擎实例注册信息、调度元数据、分布式锁状态。明文传输使内网攻击者可以：
1. 嗅探调度元数据，了解集群拓扑
2. 在无 mTLS 的 Etcd 服务端配置下，直接读写引擎注册信息，实施中间人攻击

---

### T07 — Etcd 命名空间路径来自环境变量

**风险等级**：MEDIUM  
**代码位置**：`motor/common/etcd/etcd_client.py:31-32`

```python
namespace = os.getenv("POD_NAMESPACE", "")
job_name = os.getenv("JOB_NAME", "")
# 所有 Key = namespace + "/" + job_name + key
```

`namespace` 和 `job_name` 在模块加载时从环境变量读取，固定为模块级变量。在共享进程环境或容器权限提升场景下，篡改这两个环境变量后重启进程，可将 Etcd 操作重定向至其他命名空间，实现横向越权访问。

---

### T08 — OLC 配置路径写入环境变量前无白名单校验

**风险等级**：MEDIUM  
**代码位置**：`motor/coordinator/api_server/inference_server.py:202-210`

```python
path = rate_limit_config.olc_config_path   # 来自配置，未经路径白名单校验
os.environ['OLC_CONFIG_PATH'] = path        # 直接写入进程环境变量
```

若 `olc_config_path` 可通过热更新接口或配置注入被攻击者控制，则可将 OLC 配置路径指向任意文件，影响 OLC 库的限速行为。`load_rate_limit_config` 中的 `validate_file_security`（`fastapi_middleware.py:72`）仅检查符号链接和文件权限，不校验路径是否在合法目录内。

---

### T09 — 审计日志无法区分独立用户身份

**风险等级**：MEDIUM  
**代码位置**：`motor/common/http/security_utils.py:102-107`

```python
if auth_header:
    user_id = "authenticated_user"   # 所有已认证请求使用相同标识
else:
    user_id = "anonymous"
```

所有携带 Authorization 头的请求在审计日志中均记录为 `user_id=authenticated_user`，无论实际使用哪个 API Key。这导致：
- 安全事件发生后无法溯源到具体调用方
- 无法对单个用户的异常行为进行审计关联
- 满足合规要求（如等保三级操作日志审计）存在困难

---

### T10 — Scheduler ZMQ 传输无对端认证

**风险等级**：MEDIUM  
**代码位置**：`motor/coordinator/scheduler/runtime/scheduler_client.py`

Coordinator 与 Scheduler 之间通过 ZMQ 异步套接字通信，传输 `SchedulerRequest`/`SchedulerResponse` 消息。当前实现无任何传输层认证机制（无 HMAC 签名、无 CURVE 加密、无 mTLS）。

在集群内部网络可达的前提下，攻击者可：
1. 向 Scheduler ZMQ 端口发送伪造的调度响应，影响 Prefill/Decode 引擎对分配
2. 发送大量恶意消息耗尽 Scheduler 处理队列，导致调度拒绝服务

---

### T11 — SSL Context 对象携带明文私钥密码

**风险等级**：MEDIUM  
**代码位置**：`motor/common/http/cert_util.py:747`

```python
context.password = password.decode(UTF8_ENCODING)  # 明文密码作为对象属性持久存储
```

`CertUtil.construct_cert_context` 在构建 SSL Context 后，将解密后的私钥密码以字符串属性形式附加到 `context` 对象上。若该对象在运行时被序列化、pickle、或通过调试接口暴露，私钥密码将明文泄露。

---

### T12 — 路径遍历防护遗漏部分编码变体

**风险等级**：LOW  
**代码位置**：`motor/common/http/security_utils.py:135-148`

当前 `validate_and_sanitize_path` 检测以下模式：`..`、`%2e%2e`、`%2f`、`%5c`，但未覆盖：
- `%252e%252e`（二次 URL 编码的 `..`，在某些框架的解码流水线中可绕过）
- `..%c0%af`（UTF-8 过长编码的 `/`）
- `%EF%BC%8F`（全角斜线 Unicode 变体）

实际风险取决于 FastAPI/uvicorn 的路径规范化行为，但防护完整性存在缺口。

---

### T13 — 敏感字段过滤依赖字段名黑名单匹配

**风险等级**：LOW  
**代码位置**：`motor/common/http/security_utils.py:51-65`

`filter_sensitive_body` 通过字段名子串匹配过滤敏感字段，当前黑名单包含 `password`、`token`、`api_key` 等 12 个词条。未覆盖的字段名变体（如 `bearer`、`auth`、`credential`、`private_key`、`x_api_key`）若出现在请求 Body 中，其明文值将被记录到日志。

---

## 四、风险矩阵

```
影响
  高 │ T03(CRIT) │ T01  T04  │           │
     │           │ T06       │           │
  中 │           │ T08  T09  │ T02  T05  │
     │           │ T10  T11  │ T07       │
  低 │           │           │ T12  T13  │
     └───────────┴───────────┴───────────┘
       确定触发      条件触发     难以利用
                    可能性
```

---

## 五、修复建议

| 优先级 | 问题 | 建议措施 | 关键代码位置 |
|---|---|---|---|
| P0 | T03 NodeManager 无认证 | 为所有 `/node-manager/*` 路由添加共享密钥或 mTLS 中间件；生产环境通过 NetworkPolicy 限制端口访问 | `node_manager_api.py:46` |
| P0 | T01 API Key 默认关闭 | 将 `APIKeyConfig.enable_api_key` 默认值改为 `True`，改为显式关闭 | `motor/config/coordinator.py:232` |
| P1 | T06 Etcd 明文降级 | 移除静默降级路径；至少在 `enable_tls=False` 时打印 `CRITICAL` 级别告警，要求运维确认 | `etcd_client.py:69` |
| P1 | T04 限速器进程级 | 文档明确说明 `SimpleRateLimitMiddleware` 为单进程限速；生产环境在网关层（Nginx/API Gateway）部署分布式限速 | `fastapi_middleware.py:128` |
| P2 | T09 审计无法溯源 | `log_audit_event` 中记录 token 哈希前缀（如 `sha256(token)[:12]`）替代固定字符串 | `security_utils.py:105` |
| P2 | T11 SSL Context 携带密码 | `construct_cert_context` 构建完成后删除 `context.password` 属性 | `cert_util.py:747` |
| P2 | T08 OLC 路径无校验 | 写入 `OLC_CONFIG_PATH` 前调用 `os.path.realpath()` 并校验路径在白名单目录内 | `inference_server.py:205` |
| P3 | T10 ZMQ 无认证 | 为 Scheduler ZMQ 通道添加 HMAC 消息签名或 ZMQ CURVE 加密 | `scheduler_client.py` |
| P3 | T02 时序侧信道 | L188 的 `in` 操作符改为 `hmac.compare_digest`，或仅保留 PBKDF2 验证路径 | `inference_server.py:188` |
| P3 | T12 路径遍历 | `validate_and_sanitize_path` 增加对 `%25` 二次编码的处理 | `security_utils.py:135` |

---

## 六、附录：关键代码索引

| 文件 | 安全相关逻辑 |
|---|---|
| `motor/coordinator/api_server/inference_server.py:172` | API Key 验证入口 |
| `motor/node_manager/api_server/node_manager_api.py:46` | NodeManager FastAPI 实例（无中间件）|
| `motor/common/etcd/etcd_client.py:68` | Etcd 明文 gRPC 降级 |
| `motor/coordinator/middleware/fastapi_middleware.py:241` | 限速 Fail-Open 异常处理 |
| `motor/coordinator/api_server/inference_server.py:205` | OLC 路径写入环境变量 |
| `motor/common/http/security_utils.py:96` | 审计日志 `log_audit_event` |
| `motor/common/http/security_utils.py:128` | 路径遍历防护 |
| `motor/common/http/security_utils.py:45` | 敏感字段过滤 |
| `motor/common/http/key_encryption.py:275` | API Key PBKDF2 验证 |
| `motor/common/http/cert_util.py:747` | SSL Context 明文密码属性 |
| `motor/coordinator/scheduler/runtime/scheduler_client.py` | ZMQ 调度通信（无认证）|
