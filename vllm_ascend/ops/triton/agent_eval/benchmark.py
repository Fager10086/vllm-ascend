#!/usr/bin/env python3
###############################################################################
# benchmark.py — 算子性能批量对比脚本
#
# 用法:
#   python3 benchmark.py <input_csv> [-o output_csv] [--result-dir DIR]
#
# 输入 CSV 格式 (无表头 或 有表头均可):
#   kernel_name, kernel_path, golden_path
#
# 示例:
#   python3 benchmark.py tasks.csv
#   python3 benchmark.py tasks.csv -o report.csv --result-dir ./prof_result
#
# 阈值配置:
#   脚本会根据理论建模输出自动判断算子类型, 不同类型使用不同阈值.
#   算子类型分类:
#     - pure_vector       : 纯 Vector 算子 (仅 AIV time != 0)
#     - pure_cube         : 纯 Cube 算子 (仅 AIC time != 0)
#     - cv_cube_bound     : CV 融合, Cube Bound (AIC time > AIV time)
#     - cv_vector_bound   : CV 融合, Vector Bound (AIV time >= AIC time)
#   每种类型进一步区分搬运 Bound / 计算 Bound:
#     - Vector: AIV time > AIV VEC → 搬运 Bound, 否则计算 Bound
#     - Cube:   AIC time > AIC CUBE → 搬运 Bound, 否则计算 Bound
#
#   可通过 --threshold-config JSON 文件自定义各类型阈值, 格式:
#   {
#     "pure_vector_compute_bound": 2.0,
#     "pure_vector_memory_bound": 1.5,
#     "pure_cube_compute_bound": 2.0,
#     "pure_cube_memory_bound": 1.5,
#     "cv_cube_bound_compute_bound": 2.5,
#     "cv_cube_bound_memory_bound": 2.0,
#     "cv_vector_bound_compute_bound": 2.5,
#     "cv_vector_bound_memory_bound": 2.0,
#     "default": 2.0
#   }
###############################################################################

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


# ============================================================================
# 默认阈值配置
# ============================================================================
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


# ============================================================================
# 1. 运行 msprof 采集性能数据
# ============================================================================
def run_msprof(kernel_path: str, result_dir: str) -> str:
    """
    对指定测试脚本运行 msprof, 返回性能数据保存目录 (PROF_xxx 路径).
    """
    # cmd = (
    #     f'msprof --application="python -m pytest {kernel_path}" '
    #     f'--output="{result_dir}"'
    # )
    cmd = (
        f'msprof --output="{result_dir}" '
        f'python -m pytest {kernel_path}'
    )
    print(f"\n[INFO] 运行 msprof: {cmd}")

    proc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    combined = proc.stdout + proc.stderr

    # 实时打印 msprof 输出 (简略)
    for line in combined.splitlines():
        print(f"  | {line}")

    if proc.returncode != 0:
        print(f"[ERROR] msprof 执行失败 (退出码 {proc.returncode}), 脚本: {kernel_path}")
        return ""

    # 解析 "Data is saved in /xxx/PROF_xxxx"
    matches = re.findall(r'Data is saved in (\S+)', combined)
    if not matches:
        print("[ERROR] 无法从 msprof 输出中解析性能数据路径")
        return ""

    prof_dir = matches[-1]
    print(f"[INFO] 性能数据路径: {prof_dir}")
    return prof_dir


# ============================================================================
# 2. 从 op_summary CSV 中提取指定 kernel 的性能数据
# ============================================================================
def find_op_summary_csv(prof_dir: str) -> str:
    """在 prof_dir/mindstudio_profiler_output/ 下查找 op_summary_*.csv"""
    pattern = os.path.join(prof_dir, "mindstudio_profiler_output", "op_summary_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] 未找到 op_summary CSV: {pattern}")
        return ""
    return files[-1]  # 取最新的


def _clean_shape(raw: str) -> str:
    """去除 CSV 额外引号, 将逗号分隔的维度用 x 连接, 保留分号分隔的多个 shape."""
    raw = raw.strip().strip('"')
    # 每个 shape 以分号分隔, 维度以逗号分隔 → 转为 d0xd1x... 形式
    parts = [p.strip() for p in raw.split(';') if p.strip()]
    return ';'.join(p.replace(',', 'x') for p in parts)


def extract_kernel_perf(op_summary_csv: str, kernel_name: str) -> list[dict]:
    """
    从 op_summary CSV 中检索 kernel_name, 返回匹配行列表.
    每个元素: {"op_name", "op_type", "task_type", "input_shapes",
               "input_data_types", "output_shapes", "output_data_types", "duration_us"}
    """
    results = []
    with open(op_summary_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, quotechar='"')
        if not reader.fieldnames:
            return results
        if "Op Name" not in reader.fieldnames or "Task Duration(us)" not in reader.fieldnames:
            print(f"[WARN] CSV 缺少必要列: {op_summary_csv}")
            return results

        for row in reader:
            op_name = row.get("Op Name", "").strip()
            if kernel_name.lower() in op_name.lower():
                results.append({
                    "op_name":           op_name,
                    "op_type":           row.get("OP Type", "").strip(),
                    "task_type":         row.get("Task Type", "").strip(),
                    "input_shapes":      _clean_shape(row.get("Input Shapes", "")),
                    "input_data_types":  row.get("Input Data Types", "").strip().strip('"'),
                    "output_shapes":     _clean_shape(row.get("Output Shapes", "")),
                    "output_data_types": row.get("Output Data Types", "").strip().strip('"'),
                    "duration_us":       row.get("Task Duration(us)", "").strip(),
                })
    return results


# ============================================================================
# 3. 获取理论极限性能 + 算子类型分类
# ============================================================================
def _normalize_dtype(dtype: str) -> str:
    """统一 dtype 表示: bf16/BF16→BF16, float/FLOAT/float32/FLOAT32→FLOAT, etc."""
    d = dtype.strip().upper()
    # strip "DT_" prefix emitted by op_summary (e.g. "DT_BF16" → "BF16")
    if d.startswith("DT_"):
        d = d[3:]
    alias = {
        "FP16": "FLOAT16", "FP32": "FLOAT32", "FP64": "FLOAT64",
        "BF16": "BFLOAT16",
        "FLOAT": "FLOAT32", "DOUBLE": "FLOAT64",
        "HALF": "FLOAT16",
        "INT": "INT32", "LONG": "INT64",
        "BOOL": "BOOL",
    }
    return alias.get(d, d)


def _make_match_key(shapes: list[str], dtypes: list[str]) -> str:
    """
    根据 shape + dtype 列表生成匹配 key.
    shape 格式: "1x7168" 等 (维度用 x 连接)
    dtype 格式: 统一大写
    返回: "1x7168_BFLOAT16|256_BFLOAT16|..."
    """
    pairs = []
    for i in range(max(len(shapes), len(dtypes))):
        s = shapes[i].strip() if i < len(shapes) else ""
        d = _normalize_dtype(dtypes[i]) if i < len(dtypes) else ""
        pairs.append(f"{s}_{d}")
    return "|".join(pairs)


def make_match_key_from_actual(rec: dict) -> str:
    """从实测记录 (op_summary) 生成匹配 key."""
    shapes = [s.strip() for s in rec["input_shapes"].split(';') if s.strip()]
    dtypes = [d.strip() for d in rec["input_data_types"].split(';') if d.strip()]
    return _make_match_key(shapes, dtypes)


def parse_theoretical_cases(text: str) -> list[dict]:
    """
    从理论极限性能脚本的打屏输出中解析多个测试用例.

    每个用例的输出格式:
        input[0]: (1, 7168) bf16
        input[1]: (256,) bf16
        ...
        Latency:      xxx us
        AIC time:     xxx us
        AIV time:     xxx us
        AIC CUBE:     xxx us
        AIV VEC:      xxx us

    返回: [{
        "match_key": str,        # 用于和实测数据匹配的 key
        "input_shapes": str,     # "1x7168;256" 格式
        "input_data_types": str, # "BF16;BF16" 格式
        "latency": float,
        "aic_time": float, "aiv_time": float,
        "aic_cube": float, "aiv_vec": float,
        "op_category": str, "bound_detail": str,
    }, ...]
    """
    cases = []
    lines = text.splitlines()

    # 收集当前 case 的 input 信息
    cur_inputs_shapes = []   # ["1x7168", "256", ...]
    cur_inputs_dtypes = []   # ["bf16", "bf16", ...]
    cur_metrics_lines = []   # 收集指标行

    input_pat = re.compile(r'input\[\d+\]:\s*\(([^)]*)\)\s*(\S+)')
    metric_patterns = {
        "latency":  re.compile(r'Latency:\s+([\d.]+)\s*us'),
        "aic_time": re.compile(r'AIC time:\s+([\d.]+)\s*us'),
        "aiv_time": re.compile(r'AIV time:\s+([\d.]+)\s*us'),
        "aic_cube": re.compile(r'AIC CUBE:\s+([\d.]+)\s*us'),
        "aiv_vec":  re.compile(r'AIV VEC:\s+([\d.]+)\s*us'),
    }

    def flush_case():
        """将当前收集的 input + metrics 组装成一个 case."""
        if not cur_inputs_shapes:
            return
        # 解析指标
        block = '\n'.join(cur_metrics_lines)
        metrics = {}
        for key, pat in metric_patterns.items():
            m = pat.search(block)
            metrics[key] = float(m.group(1)) if m else 0.0

        if metrics.get("latency", 0) <= 0:
            return

        op_category, bound_detail = classify_operator(metrics)

        # 构建 shape 和 dtype 字符串 (用于输出 CSV)
        shapes_normalized = []
        for s in cur_inputs_shapes:
            dims = [d.strip() for d in s.split(',') if d.strip()]
            shapes_normalized.append('x'.join(dims))

        dtypes_normalized = [_normalize_dtype(d) for d in cur_inputs_dtypes]

        match_key = _make_match_key(shapes_normalized, dtypes_normalized)

        cases.append({
            "match_key":        match_key,
            "input_shapes":     ';'.join(shapes_normalized),
            "input_data_types": ';'.join(dtypes_normalized),
            **metrics,
            "op_category":      op_category,
            "bound_detail":     bound_detail,
        })

    for line in lines:
        line_stripped = line.strip()

        # 匹配 input 行
        m_input = input_pat.search(line_stripped)
        if m_input:
            # 如果遇到 input[0] 且已有积累, 说明进入新 case
            idx_match = re.search(r'input\[(\d+)\]', line_stripped)
            if idx_match and int(idx_match.group(1)) == 0 and cur_inputs_shapes:
                flush_case()
                cur_inputs_shapes = []
                cur_inputs_dtypes = []
                cur_metrics_lines = []
            cur_inputs_shapes.append(m_input.group(1))
            cur_inputs_dtypes.append(m_input.group(2))
            continue

        # 匹配指标行
        is_metric = False
        for pat in metric_patterns.values():
            if pat.search(line_stripped):
                is_metric = True
                break
        if is_metric:
            cur_metrics_lines.append(line_stripped)
            # AIV VEC 是最后一个指标, 遇到后 flush
            if metric_patterns["aiv_vec"].search(line_stripped):
                flush_case()
                cur_inputs_shapes = []
                cur_inputs_dtypes = []
                cur_metrics_lines = []

    # 处理尾部残留
    if cur_inputs_shapes and cur_metrics_lines:
        flush_case()

    return cases


def classify_operator(metrics: dict) -> tuple[str, str]:
    """
    根据理论建模指标判断算子类型和 bound 类型.

    返回: (op_category, bound_detail)
      op_category:
        - "pure_vector"     : 纯 Vector (AIC time == 0, AIV time != 0)
        - "pure_cube"       : 纯 Cube   (AIC time != 0, AIV time == 0)
        - "cv_cube_bound"   : CV 融合, Cube Bound  (AIC time > AIV time)
        - "cv_vector_bound" : CV 融合, Vector Bound (AIV time >= AIC time)
        - "unknown"         : 无法判断
      bound_detail:
        - "compute_bound"   : 计算 Bound
        - "memory_bound"    : 搬运 Bound
        - "unknown"         : 无法判断
    """
    aic_time = metrics.get("aic_time", 0.0)
    aiv_time = metrics.get("aiv_time", 0.0)
    aic_cube = metrics.get("aic_cube", 0.0)
    aiv_vec  = metrics.get("aiv_vec", 0.0)

    if aic_time == 0 and aiv_time == 0:
        return "unknown", "unknown"

    if aic_time == 0 and aiv_time != 0:
        op_category = "pure_vector"
        bound_detail = "memory_bound" if aiv_time > aiv_vec else "compute_bound"
    elif aic_time != 0 and aiv_time == 0:
        op_category = "pure_cube"
        bound_detail = "memory_bound" if aic_time > aic_cube else "compute_bound"
    else:
        # CV 融合
        if aic_time > aiv_time:
            op_category = "cv_cube_bound"
        else:
            op_category = "cv_vector_bound"
        # bound 取决于主导核
        if op_category == "cv_cube_bound":
            bound_detail = "memory_bound" if aic_time > aic_cube else "compute_bound"
        else:
            bound_detail = "memory_bound" if aiv_time > aiv_vec else "compute_bound"

    return op_category, bound_detail


def get_threshold_key(op_category: str, bound_detail: str) -> str:
    """将算子分类映射到阈值配置的 key."""
    if op_category == "unknown" or bound_detail == "unknown":
        return "default"
    return f"{op_category}_{bound_detail}"


def get_threshold(op_category: str, bound_detail: str,
                  thresholds: dict) -> float:
    """根据算子分类获取对应阈值."""
    key = get_threshold_key(op_category, bound_detail)
    return thresholds.get(key, thresholds.get("default", 2.0))


def get_theoretical_perf(golden_path: str) -> list[dict]:
    """
    运行理论极限性能脚本, 返回多个测试用例的理论性能列表.
    每个元素包含: match_key, input_shapes, input_data_types,
                  latency, aic_time, aiv_time, aic_cube, aiv_vec,
                  op_category, bound_detail.
    失败时返回空列表.
    """
    # tilesim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tilesim-master")
    tilesim_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = (
        f'python -m examples.api.operator_api.pytorch_examples.main '
        f'--script {golden_path}'
    )
    print(f"  [CMD] {cmd}  (cwd={tilesim_dir})")

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=tilesim_dir)
    combined = proc.stdout + proc.stderr

    for line in combined.splitlines():
        print(f"    | {line}")

    if proc.returncode != 0:
        print(f"  [ERROR] 理论极限性能脚本执行失败 (退出码 {proc.returncode})")
        return []

    cases = parse_theoretical_cases(combined)

    if not cases:
        print("  [ERROR] 未能从输出中解析任何测试用例")
        return []

    for c in cases:
        print(f"  [INFO] 理论用例: shapes={c['input_shapes']}  dtypes={c['input_data_types']}  "
              f"Latency={c['latency']}us  "
              f"AIC={c['aic_time']}  AIV={c['aiv_time']}  "
              f"类型={c['op_category']}/{c['bound_detail']}")

    return cases


# ============================================================================
# 4. 主流程
# ============================================================================
def parse_input_csv(csv_path: str) -> list[dict]:
    """
    解析输入 CSV, 支持有表头 / 无表头两种格式.
    返回: [{"kernel_name", "kernel_path", "golden_path"}, ...]
    """
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)

        # 尝试判断是否有表头
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        reader = csv.reader(f)
        if has_header:
            header = next(reader)  # 跳过表头

        for line_no, cols in enumerate(reader, start=2 if has_header else 1):
            cols = [c.strip() for c in cols]
            if len(cols) < 3:
                print(f"[WARN] 第 {line_no} 行列数不足, 跳过: {cols}")
                continue
            rows.append({
                "kernel_name":  cols[0],
                "kernel_path":  cols[1],
                "golden_path":  cols[2],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="算子性能批量对比")
    parser.add_argument("input_csv", help="输入 CSV 文件 (kernel_name, kernel_path, golden_path)")
    parser.add_argument("-o", "--output", default="benchmark_result.csv",
                        help="输出全量对比 CSV (默认: benchmark_result.csv)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="全局 ratio 阈值 (覆盖所有类型的默认阈值)")
    parser.add_argument("--threshold-config", default=None,
                        help="阈值配置 JSON 文件路径, 按算子类型配置不同阈值")
    parser.add_argument("--result-dir", default="./result",
                        help="msprof 输出目录 (默认: ./result)")
    args = parser.parse_args()

    # ---- 加载阈值配置 ----
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.threshold_config:
        if not os.path.isfile(args.threshold_config):
            print(f"[ERROR] 阈值配置文件不存在: {args.threshold_config}")
            sys.exit(1)
        with open(args.threshold_config, encoding='utf-8') as f:
            user_thresholds = json.load(f)
        thresholds.update(user_thresholds)
        print(f"[INFO] 已加载阈值配置: {args.threshold_config}")

    if args.threshold is not None:
        for key in thresholds:
            thresholds[key] = args.threshold
        print(f"[INFO] 全局阈值覆盖: 所有类型统一使用 {args.threshold}")

    if not os.path.isfile(args.input_csv):
        print(f"[ERROR] 输入文件不存在: {args.input_csv}")
        sys.exit(1)

    # ---- 4.1 解析输入 ----
    tasks = parse_input_csv(args.input_csv)
    if not tasks:
        print("[ERROR] 输入 CSV 无有效数据")
        sys.exit(1)

    print("=" * 64)
    print(" 算子性能批量对比")
    print("=" * 64)
    print(f" 输入文件  : {args.input_csv}")
    print(f" 算子数量  : {len(tasks)}")
    print(f" 输出文件  : {args.output}")
    print(f" msprof 目录: {args.result_dir}")
    print(f" 阈值配置  :")
    for k, v in thresholds.items():
        print(f"   {k:40s} = {v}")
    print("=" * 64)

    # ---- 4.2 按 kernel_path 分组, 避免重复执行 msprof ----
    path_groups: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        path_groups[t["kernel_path"]].append(t)

    print(f"\n[INFO] 共 {len(path_groups)} 个不同的测试脚本需要执行 msprof\n")

    # kernel_path -> op_summary_csv 路径
    path_to_summary: dict[str, str] = {}

    for idx, kernel_path in enumerate(path_groups, start=1):
        print(f"\n{'─' * 64}")
        print(f" [{idx}/{len(path_groups)}] 测试脚本: {kernel_path}")
        print(f"{'─' * 64}")

        prof_dir = run_msprof(kernel_path, args.result_dir)
        if not prof_dir:
            print(f"[WARN] 跳过 {kernel_path} — msprof 失败")
            continue

        summary_csv = find_op_summary_csv(prof_dir)
        if summary_csv:
            path_to_summary[kernel_path] = summary_csv
            print(f"[INFO] op_summary CSV: {summary_csv}")

    # ---- 4.3 汇总: 提取实测性能 + 理论性能 → 输出对比 CSV ----
    print(f"\n{'=' * 64}")
    print(" 汇总对比结果")
    print(f"{'=' * 64}\n")

    output_rows = []

    for t in tasks:
        kernel_name = t["kernel_name"]
        kernel_path = t["kernel_path"]
        golden_path = t["golden_path"]

        summary_csv = path_to_summary.get(kernel_path, "")
        if not summary_csv:
            print(f"[WARN] {kernel_name}: 无实测数据 (msprof 未成功)")
            output_rows.append({
                "kernel_name":           kernel_name,
                "kernel_path":           kernel_path,
                "golden_path":           golden_path,
                "input_shapes":          "",
                "input_data_types":      "N/A",
                "output_shapes":         "N/A",
                "output_data_types":     "N/A",
                "actual_duration_us":    "N/A",
                "theoretical_duration_us": "N/A",
                "aic_time_us":           "N/A",
                "aiv_time_us":           "N/A",
                "aic_cube_us":           "N/A",
                "aiv_vec_us":            "N/A",
                "op_category":           "N/A",
                "bound_detail":          "N/A",
                "threshold_key":         "N/A",
                "threshold":             "N/A",
                "ratio":                 "N/A",
            })
            continue

        # 提取实测性能 (可能多行)
        perf_records = extract_kernel_perf(summary_csv, kernel_name)

        if not perf_records:
            print(f"[WARN] {kernel_name}: 在 op_summary 中未匹配到")
            output_rows.append({
                "kernel_name":           kernel_name,
                "kernel_path":           kernel_path,
                "golden_path":           golden_path,
                "input_shapes":          "",
                "input_data_types":      "N/A",
                "output_shapes":         "N/A",
                "output_data_types":     "N/A",
                "actual_duration_us":    "NOT_FOUND",
                "theoretical_duration_us": "N/A",
                "aic_time_us":           "N/A",
                "aiv_time_us":           "N/A",
                "aic_cube_us":           "N/A",
                "aiv_vec_us":            "N/A",
                "op_category":           "N/A",
                "bound_detail":          "N/A",
                "threshold_key":         "N/A",
                "threshold":             "N/A",
                "ratio":                 "N/A",
            })
            continue

        # 获取理论极限性能 (多个用例)
        print(f"\n[INFO] {kernel_name}: 获取理论极限性能 (golden: {golden_path})")
        theo_cases = get_theoretical_perf(golden_path)

        # 建立 match_key → theo_case 的索引
        theo_map: dict[str, dict] = {}
        for tc in theo_cases:
            theo_map[tc["match_key"]] = tc

        if theo_cases:
            print(f"  [INFO] 理论用例数: {len(theo_cases)}")
        else:
            print(f"  [WARN] 未获取到理论极限数据")

        for rec in perf_records:
            # 先精确匹配 key，失败则按 dtype 匹配（varlen 场景 T 维度不同）
            actual_key = make_match_key_from_actual(rec)
            matched = theo_map.get(actual_key)

            if not matched and len(theo_cases) == 1:
                # 单用例时直接使用（varlen 下实测 T != 理论 T）
                matched = theo_cases[0]
            elif not matched and theo_cases:
                # 多用例时按 dtype 匹配
                actual_dtypes = [_normalize_dtype(d) for d in rec["input_data_types"].split(';') if d.strip()]
                for tc in theo_cases:
                    theo_dtypes = tc["input_data_types"].split(';')
                    if actual_dtypes and theo_dtypes and actual_dtypes[0] == theo_dtypes[0]:
                        matched = tc
                        break

            if matched:
                theoretical   = matched["latency"]
                op_category   = matched["op_category"]
                bound_detail  = matched["bound_detail"]
                aic_time_val  = matched["aic_time"]
                aiv_time_val  = matched["aiv_time"]
                aic_cube_val  = matched["aic_cube"]
                aiv_vec_val   = matched["aiv_vec"]
                thresh_key    = get_threshold_key(op_category, bound_detail)
                thresh_val    = get_threshold(op_category, bound_detail, thresholds)
            else:
                theoretical   = -1.0
                op_category   = "N/A"
                bound_detail  = "N/A"
                aic_time_val  = "N/A"
                aiv_time_val  = "N/A"
                aic_cube_val  = "N/A"
                aiv_vec_val   = "N/A"
                thresh_key    = "N/A"
                thresh_val    = "N/A"
                print(f"  [WARN] 实测记录未匹配到理论用例: key={actual_key}")

            try:
                actual = float(rec["duration_us"])
                if isinstance(theoretical, (int, float)) and theoretical > 0:
                    ratio = round(actual / theoretical, 4)
                else:
                    ratio = "N/A"
            except (ValueError, TypeError):
                actual = rec["duration_us"]
                ratio = "N/A"

            output_rows.append({
                "kernel_name":             kernel_name,
                "kernel_path":             kernel_path,
                "golden_path":             golden_path,
                "input_shapes":            rec["input_shapes"],
                "input_data_types":        rec["input_data_types"],
                "output_shapes":           rec["output_shapes"],
                "output_data_types":       rec["output_data_types"],
                "actual_duration_us":      rec["duration_us"],
                "theoretical_duration_us": theoretical if theoretical > 0 else "N/A",
                "aic_time_us":             aic_time_val,
                "aiv_time_us":             aiv_time_val,
                "aic_cube_us":             aic_cube_val,
                "aiv_vec_us":              aiv_vec_val,
                "op_category":             op_category,
                "bound_detail":            bound_detail,
                "threshold_key":           thresh_key,
                "threshold":               thresh_val,
                "ratio":                   ratio,
            })

            status = "✗ 未达标" if isinstance(ratio, float) and isinstance(thresh_val, (int, float)) and ratio >= thresh_val else "✓ 达标"
            if not isinstance(ratio, float):
                status = "? N/A"
            theo_str = f"{theoretical:>10.3f}" if isinstance(theoretical, (int, float)) and theoretical > 0 else "       N/A"
            matched_str = "匹配" if matched else "未匹配"
            print(f"  [{matched_str}] Shape={rec['input_shapes']:30s}  "
                  f"实测={rec['duration_us']:>10s} us  "
                  f"理论={theo_str} us  "
                  f"比值={ratio}  "
                  f"阈值={thresh_val}  {status}")

    # ---- 4.4 按各行自身阈值分拣 ----
    rows_above = []  # ratio >= 阈值 (未达标)
    rows_below = []  # ratio <  阈值 (达标)

    for row in output_rows:
        r = row["ratio"]
        row_thresh = row["threshold"]
        if isinstance(r, (int, float)) and isinstance(row_thresh, (int, float)):
            if r >= row_thresh:
                rows_above.append(row)
            else:
                rows_below.append(row)
        else:
            rows_above.append(row)

    # ---- 4.5 写出三个 CSV ----
    fieldnames = [
        "kernel_name",
        "kernel_path",
        "golden_path",
        "input_shapes",
        "input_data_types",
        "output_shapes",
        "output_data_types",
        "actual_duration_us",
        "theoretical_duration_us",
        "aic_time_us",
        "aiv_time_us",
        "aic_cube_us",
        "aiv_vec_us",
        "op_category",
        "bound_detail",
        "threshold_key",
        "threshold",
        "ratio",
    ]

    base, ext = os.path.splitext(args.output)
    out_all   = args.output
    out_above = f"{base}_above{ext}"   # ratio >= 阈值
    out_below = f"{base}_below{ext}"   # ratio <  阈值

    def write_csv(path, rows):
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_all,   output_rows)
    write_csv(out_above, rows_above)
    write_csv(out_below, rows_below)

    print(f"\n{'=' * 64}")
    print(f" 完成! (按算子类型自动选择阈值)")
    print(f"{'─' * 64}")
    print(f" 全量报告  : {os.path.abspath(out_all):50s} ({len(output_rows)} 条)")
    print(f" 未达标     : {os.path.abspath(out_above):50s} ({len(rows_above)} 条)")
    print(f" 已达标     : {os.path.abspath(out_below):50s} ({len(rows_below)} 条)")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
