# Shape Profiler 设计开发文档

## 1. 概述

`shape_profiler.py` 是一个针对 vLLM Ascend 推理引擎的算子 Shape 采集工具。其核心目标是：在真实推理场景中，记录每次算子调用时的**完整入参+出参 shape/dtype 组合**，用于后续的算子优化、图编译 profiling 与性能分析。

工具支持以下场景：
- 单卡离线推理
- 单进程 Serve 模式
- 多进程 Data Parallel（DP）Serve 模式
- 手动 DP 模式（配合 `launch_online_dp.py`）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        shape_profiler.py                        │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐ │
│  │  Hook Layer  │───▶│ ShapeProfiler │───▶│  Export Pipeline │ │
│  └──────────────┘    │   (singleton) │    └──────────────────┘ │
│                      └───────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Hook Layer

负责拦截算子调用，共三类 Hook：

| Hook 类型 | 实现方式 | 覆盖范围 |
|---|---|---|
| torch ops | `OpOverloadPacketWrapper` 代理对象替换 `torch.ops.*` | 约 50 个 NPU/标准算子 |
| Triton ops | `TritonKernelWrapper` 代理对象替换模块属性 | 约 14 个 Triton kernel |
| Custom ops | Hook `direct_register_custom_op` 在注册时包装 `op_func` | 所有通过 vllm 注册的自定义算子 |
| Graph mode | `TorchDispatchMode`（`_ShapeDispatchMode`） | 图模式下的所有 dispatch op |

**代理类结构：**

```
OpOverloadPacketWrapper
  __call__ → record_call() → return result
  __getattr__ → 透传原始属性

TritonKernelWrapper
  __getitem__(grid) → _TritonLaunchedKernelWrapper
    __call__ → record_call() → return result
```

### 2.2 ShapeProfiler（单例）

所有进程共享一个 `ShapeProfiler` 实例（通过 `__new__` 实现单例）。

核心职责：
- `install_hooks(output_dir)` — 安装所有 Hook，初始化输出文件
- `record_call(op_name, args, kwargs, result)` — 提取 shape/dtype，写入 JSONL
- `aggregate_records(records)` — 将原始记录聚合为按调用签名统计的结构
- `generate_reports(output_dir)` — 调度导出 JSON 和 Markdown 报告
- `_cleanup()` — `atexit` 注册，进程退出时自动 flush

---

## 3. 数据模型

### 3.1 原始记录格式（JSONL）

每次算子调用写一条 JSON 行，入参与出参严格绑定，反映一次完整调用快照：

```json
{
  "op_name": "npu_rms_norm",
  "arg_shapes": [
    {
      "arg_idx": 0,
      "tensors": [{"shape": [2048, 4096], "dtype": "torch.float16"}]
    },
    {
      "arg_idx": 1,
      "tensors": [{"shape": [4096], "dtype": "torch.float16"}]
    }
  ],
  "kwarg_shapes": [],
  "output_shapes": [{"shape": [2048, 4096], "dtype": "torch.float16"}],
  "pid": 12345
}
```

- `arg_shapes[i].tensors`：第 `i` 个位置参数中的所有 tensor（支持嵌套 list/tuple/dict）
- `kwarg_shapes[i].tensors`：第 `i` 个关键字参数中的所有 tensor
- `output_shapes`：输出值中的所有 tensor
- 一条 record = 一次完整调用，不产生笛卡尔积组合

### 3.2 聚合模型（内存 / JSON 输出）

```python
@dataclass
class OpShapeRecord:
    op_name: str
    call_count: int
    call_signatures: Dict[str, int]  # 签名字符串 -> 出现次数
```

**调用签名（call signature）** 格式：

```
arg0=[32, 128]:torch.float16|[128, 64]:torch.float16, arg1=..., key=..., out=[32, 64]:torch.float16
```

- 单个参数位置上的多个 tensor 用 `|` 分隔
- 所有参数（位置参数、关键字参数、输出）拼接为一个字符串
- 相同签名的多次调用只累加计数，不重复存储

---

## 4. 多进程 / DP 模式

### 4.1 文件命名规范

| 场景 | 文件名 |
|---|---|
| 单进程 | `shape_records_pid<PID>.jsonl` |
| DP 模式 worker | `shape_records_dp<DP_RANK>_pid<PID>.jsonl` |
| 合并后 | `shape_records_merged.jsonl` |

### 4.2 DP 模式工作流

```
主进程
  └─ 启动 N 个子进程（multiprocessing.Process 或 subprocess）
       每个子进程设置环境变量：
         SHAPE_PROFILER_OUTPUT_DIR=<output_dir>
         SHAPE_PROFILER_DP_RANK=<rank>
         SHAPE_PROFILER_DP_SIZE=<total>
         ASCEND_RT_VISIBLE_DEVICES=<devices>
       子进程运行 vllm serve → 自动安装 hook → 写 JSONL 文件

Ctrl+C → 主进程等待子进程退出（15s timeout）
       → 调用 _merge_pid_files() 合并所有 JSONL
       → 生成报告
```

### 4.3 关键环境变量

| 变量 | 作用 |
|---|---|
| `SHAPE_PROFILER_OUTPUT_DIR` | worker 进程写入 JSONL 的目录 |
| `SHAPE_PROFILER_DP_RANK` | 当前 worker 的 DP rank |
| `SHAPE_PROFILER_DP_SIZE` | DP 总规模 |
| `VLLM_WORKER_MULTIPROC_METHOD` | TP > 1 时强制 `spawn` |

---

## 5. 导出流水线

```
JSONL 文件
    │
    ▼
load_records()          读取所有行，解析为 List[Dict]
    │
    ▼
aggregate_records()     按 op_name 分组，构建 call_signatures 计数器
    │
    ├──▶ export_json()      → shape_stats.json
    │
    └──▶ export_markdown()  → shape_stats_report.md
```

**`shape_stats.json` 结构：**

```json
{
  "metadata": {
    "start_time": "...",
    "end_time": "...",
    "total_records": 1855,
    "unique_ops": 42,
    "dp_ranks": [0, 1, 2, 3],
    "dp_rank_count": 4
  },
  "records": [
    {
      "op_name": "npu_rms_norm",
      "call_count": 128,
      "call_signatures": {
        "arg0=[2048, 4096]:torch.float16, ..., out=...": 64,
        "arg0=[1024, 4096]:torch.float16, ..., out=...": 64
      }
    }
  ]
}
```

---

## 6. 使用方式

### 6.1 单卡离线推理

```bash
python shape_profiler.py --model /path/to/model --output-dir ./output
```

### 6.2 单进程 Serve

```bash
python shape_profiler.py --model /path/to/model --serve --port 8000 \
    --output-dir ./output
```

### 6.3 DP Serve（模板脚本）

```bash
python shape_profiler.py --model /path/to/model --serve \
    --dp-size 8 --tp-size 1 --dp-size-local 8 \
    --dp-template ./run_dp_template.sh \
    --output-dir ./output
```

### 6.4 手动 DP 模式

```bash
export SHAPE_PROFILER_OUTPUT_DIR=./output
python launch_online_dp.py --dp-size 8 ...
# 运行完毕后：
python shape_profiler.py --merge-only --output-dir ./output
```

### 6.5 代码嵌入

```python
from shape_profiler import shape_profile
with shape_profile("./output"):
    llm.generate(prompts)
```

---

## 7. 关键设计决策

### 7.1 调用签名而非分维统计

早期实现对每个参数位置的 shape 独立统计，导致无法反映同一次调用中不同参数之间的组合关系（笛卡尔积问题）。

当前实现将一次调用的所有入参+出参 shape 序列化为一个签名字符串，以签名为 key 统计频次，完整保留参数组合信息。

### 7.2 去重（Dedup）

`patch_shape_profiler.py` 中的 worker 端记录器通过 `_seen_keys` set 对 `(op_name, arg_shapes, kwarg_shapes, output_shapes)` 四元组去重，避免相同调用模式的记录爆炸式增长。

### 7.3 图模式兼容

检测到 `--enforce-eager` 不在 `sys.argv` 时（即图模式），跳过 torch ops 的 monkey-patch（会引起 graph break），改用 `TorchDispatchMode` 在 dispatch 层捕获 shape，对图编译透明。

### 7.4 跨模块引用修补

`_patch_cross_module_refs()` 扫描 `sys.modules`，将其他模块中已持有原始函数引用的属性替换为 wrapped 版本，确保 Hook 在导入顺序不确定时依然生效。

---

## 8. 输出文件一览

| 文件 | 说明 |
|---|---|
| `shape_records_pid<PID>.jsonl` | 单进程原始记录（实时写入） |
| `shape_records_dp<R>_pid<P>.jsonl` | DP worker 原始记录 |
| `shape_records_merged.jsonl` | 合并后的记录（多进程场景） |
| `shape_stats.json` | 聚合统计（JSON，供程序消费） |
| `shape_stats_report.md` | 聚合统计（Markdown，供人阅读） |
