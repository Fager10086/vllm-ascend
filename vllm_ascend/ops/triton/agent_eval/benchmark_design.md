# Benchmark 算子性能批量对比工具 — 设计开发文档

## 1. 概述

`benchmark.py` 是一个面向昇腾 NPU 的算子性能批量对比脚本。其核心流程为：

1. 通过 `msprof` 采集算子在真实硬件上的**实测性能**
2. 通过 `tilesim` 理论建模获取算子的**理论极限性能**
3. 按 shape + dtype 自动匹配实测与理论数据，计算 **ratio（实测 / 理论）**
4. 根据算子类型（Vector / Cube / CV 融合）和 bound 类型（计算 / 搬运）选择对应阈值，判定是否达标
5. 输出三份 CSV 报告：全量、达标、未达标

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                          benchmark.py                            │
│                                                                  │
│  输入 CSV                                                        │
│  (kernel_name, kernel_path, golden_path)                         │
│          │                                                       │
│          ▼                                                       │
│  ┌──────────────┐     ┌───────────────────┐                      │
│  │   msprof     │     │   tilesim         │                      │
│  │  (实测采集)   │     │  (理论建模)        │                      │
│  └──────┬───────┘     └────────┬──────────┘                      │
│         │                      │                                 │
│         ▼                      ▼                                 │
│  op_summary_*.csv      stdout 解析                                │
│  (实测 duration)       (理论 Latency, AIC/AIV 指标)               │
│         │                      │                                 │
│         └──────┐    ┌──────────┘                                 │
│                ▼    ▼                                            │
│          match_key 匹配                                           │
│          (shape + dtype 对齐)                                     │
│                │                                                 │
│                ▼                                                 │
│     ratio = 实测 / 理论                                           │
│     阈值判定 (按算子类型)                                          │
│                │                                                 │
│                ▼                                                 │
│  ┌──────────────────────────────┐                                │
│  │ 输出 CSV × 3                 │                                │
│  │  benchmark_result.csv (全量)  │                                │
│  │  _above.csv (未达标)          │                                │
│  │  _below.csv (已达标)          │                                │
│  └──────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块

### 3.1 msprof 性能采集

| 函数 | 职责 |
|---|---|
| `run_msprof(kernel_path, result_dir)` | 调用 `msprof --application="python -m pytest <kernel_path>"` 采集性能数据，从输出解析 `PROF_xxx` 目录路径 |
| `find_op_summary_csv(prof_dir)` | 在 `prof_dir/mindstudio_profiler_output/` 下查找 `op_summary_*.csv`（取最新） |
| `extract_kernel_perf(csv, kernel_name)` | 从 op_summary CSV 中按 kernel_name 模糊匹配，提取每行的 Op Name / Input Shapes / Duration 等字段 |

**按 kernel_path 分组执行**：多个算子共享同一测试脚本时只运行一次 `msprof`，避免重复采集。

### 3.2 理论极限性能

| 函数 | 职责 |
|---|---|
| `get_theoretical_perf(golden_path)` | 运行 tilesim 理论建模脚本，获取 stdout 输出 |
| `parse_theoretical_cases(text)` | 从 stdout 中解析多组测试用例，每组包含 input shape/dtype + 5 个指标 |
| `classify_operator(metrics)` | 根据 AIC/AIV 指标判断算子类型和 bound 类型 |

**理论建模输出格式**（每个用例）：
```
input[0]: (1, 7168) bf16
input[1]: (256,) bf16
Latency:      12.5 us
AIC time:     0.0 us
AIV time:     10.2 us
AIC CUBE:     0.0 us
AIV VEC:      8.1 us
```

**解析状态机**：遇到 `input[0]` 时开始新用例，遇到 `AIV VEC` 时 flush 当前用例。

### 3.3 算子分类体系

根据理论建模的 5 个指标，将算子分为 **4 类 × 2 种 bound = 8 种**：

```
                     ┌─ compute_bound (AIV time ≤ AIV VEC)
 pure_vector ────────┤
 (AIC=0, AIV≠0)      └─ memory_bound  (AIV time > AIV VEC)

                     ┌─ compute_bound (AIC time ≤ AIC CUBE)
 pure_cube ──────────┤
 (AIC≠0, AIV=0)      └─ memory_bound  (AIC time > AIC CUBE)

                     ┌─ compute_bound (AIC time ≤ AIC CUBE)
 cv_cube_bound ──────┤
 (AIC > AIV)          └─ memory_bound  (AIC time > AIC CUBE)

                     ┌─ compute_bound (AIV time ≤ AIV VEC)
 cv_vector_bound ────┤
 (AIV ≥ AIC)          └─ memory_bound  (AIV time > AIV VEC)
```

每种组合有独立的达标阈值，默认值见 `DEFAULT_THRESHOLDS`。

### 3.4 实测-理论匹配

匹配使用 `match_key`：将 shape 和 dtype 规范化后拼接为唯一 key。

```
match_key = "1x7168_BFLOAT16|256_BFLOAT16"
```

**匹配策略（三级降级）**：

1. **精确匹配**：实测 match_key == 理论 match_key
2. **单用例直接匹配**：理论只有 1 个用例时直接使用（适用 varlen 场景，实测与理论 T 维度不同）
3. **dtype 模糊匹配**：多用例时按第一个 dtype 匹配

---

## 4. 数据模型

### 4.1 输入 CSV

```csv
kernel_name, kernel_path, golden_path
AddRmsNorm, tests/test_add_rms_norm.py, golden/add_rms_norm.py
```

- `kernel_name`：算子名称，用于在 op_summary CSV 中模糊匹配
- `kernel_path`：pytest 测试脚本路径（msprof 执行目标）
- `golden_path`：tilesim 理论建模脚本路径

### 4.2 输出 CSV 字段

| 字段 | 说明 |
|---|---|
| `kernel_name` | 算子名称 |
| `kernel_path` | 测试脚本路径 |
| `golden_path` | 理论建模脚本路径 |
| `input_shapes` | 输入 shape（如 `1x7168;256`） |
| `input_data_types` | 输入 dtype（如 `BFLOAT16;BFLOAT16`） |
| `output_shapes` | 输出 shape |
| `output_data_types` | 输出 dtype |
| `actual_duration_us` | 实测耗时（微秒） |
| `theoretical_duration_us` | 理论极限耗时（微秒） |
| `aic_time_us` | AIC 理论时间 |
| `aiv_time_us` | AIV 理论时间 |
| `aic_cube_us` | AIC CUBE 时间 |
| `aiv_vec_us` | AIV VEC 时间 |
| `op_category` | 算子类型 |
| `bound_detail` | Bound 类型 |
| `threshold_key` | 阈值配置 key |
| `threshold` | 使用的阈值 |
| `ratio` | 实测 / 理论（越接近 1 越好） |

### 4.3 输出文件

| 文件 | 说明 |
|---|---|
| `benchmark_result.csv` | 全量报告 |
| `benchmark_result_above.csv` | ratio ≥ 阈值（未达标） |
| `benchmark_result_below.csv` | ratio < 阈值（已达标） |

---

## 5. 阈值配置

### 5.1 默认阈值

```python
DEFAULT_THRESHOLDS = {
    "pure_vector_compute_bound":      2.0,
    "pure_vector_memory_bound":       1.5,
    "pure_cube_compute_bound":        2.0,
    "pure_cube_memory_bound":         1.5,
    "cv_cube_bound_compute_bound":    2.5,
    "cv_cube_bound_memory_bound":     2.0,
    "cv_vector_bound_compute_bound":  2.5,
    "cv_vector_bound_memory_bound":   2.0,
    "default":                        2.0,
}
```

### 5.2 自定义配置

通过 `--threshold-config` 指定 JSON 文件，格式同 `DEFAULT_THRESHOLDS`，用户配置会覆盖默认值。

`--threshold` 提供全局覆盖：所有类型统一使用同一阈值。

---

## 6. 使用方式

### 6.1 基本用法

```bash
python3 benchmark.py tasks.csv
```

### 6.2 自定义输出和目录

```bash
python3 benchmark.py tasks.csv -o report.csv --result-dir ./prof_result
```

### 6.3 自定义阈值

```bash
# 全局统一阈值
python3 benchmark.py tasks.csv --threshold 1.5

# 按算子类型配置
python3 benchmark.py tasks.csv --threshold-config my_thresholds.json
```

---

## 7. 主流程（main）

```
1. 解析 CLI 参数，加载阈值配置
2. 解析输入 CSV → tasks 列表
3. 按 kernel_path 分组
4. 对每组执行 msprof 采集 → 获取 op_summary CSV
5. 对每个 task:
   5.1 从 op_summary 提取实测性能（可能多行）
   5.2 运行 tilesim 获取理论极限（可能多用例）
   5.3 match_key 匹配实测 ↔ 理论
   5.4 计算 ratio，判定是否达标
6. 按阈值分拣为 above / below
7. 输出三份 CSV
```

---

## 8. 关键设计决策

### 8.1 按 kernel_path 分组执行 msprof

同一测试脚本可能覆盖多个算子（或同一算子的多种 shape），分组避免重复采集。一次 `msprof` 运行的 `op_summary` CSV 包含该脚本中所有算子的数据。

### 8.2 match_key 匹配机制

理论建模脚本可能产生多个用例（不同 shape），需要将实测记录精确匹配到对应的理论用例。使用 `shape_dtype` 拼接的 key 实现自动对齐。

三级降级策略确保在 varlen 等动态 shape 场景下也能完成匹配。

### 8.3 分类驱动的动态阈值

不同类型的算子（Vector / Cube / CV 融合）在不同 bound 下的优化空间不同。搬运 bound 的算子受限于带宽，阈值更宽松（如 1.5）；计算 bound 和 CV 融合的算子阈值更严格（如 2.0-2.5）。

### 8.4 dtype 规范化

op_summary CSV 和 tilesim 使用不同的 dtype 表示（如 `DT_BF16` vs `bf16`）。`_normalize_dtype()` 统一转换为大写标准形式（`BFLOAT16`），确保匹配 key 一致。

---

## 9. 外部依赖

| 工具 | 用途 | 调用方式 |
|---|---|---|
| `msprof` | 昇腾 NPU 性能采集 | `msprof --application="python -m pytest ..." --output=...` |
| `tilesim` | 理论极限性能建模 | `python -m examples.api.operator_api.pytorch_examples.main --script ...`（在 `tilesim-master/` 目录下执行） |
| `pytest` | 执行算子测试脚本 | 被 msprof 包裹调用 |
