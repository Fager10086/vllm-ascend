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
# 流程:
#   1. 按 kernel_path 分组，每个 kernel_path 只运行一次 msprof
#   2. 从 msprof 产出的 op_summary CSV 中检索各 kernel_name 的实测性能
#   3. 运行理论极限性能脚本 (当前用随机数模拟)
#   4. 按阈值分拣, 输出三个 CSV:
#      - 全量对比表
#      - ratio >= 阈值 (性能未达标)
#      - ratio <  阈值 (性能达标)
###############################################################################

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict


# ============================================================================
# 1. 运行 msprof 采集性能数据
# ============================================================================
def run_msprof(kernel_path: str, result_dir: str) -> str:
    """
    对指定测试脚本运行 msprof, 返回性能数据保存目录 (PROF_xxx 路径).
    """
    cmd = (
        f'msprof --application="python -m pytest {kernel_path}" '
        f'--output="{result_dir}"'
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


def extract_kernel_perf(op_summary_csv: str, kernel_name: str) -> list[dict]:
    """
    从 op_summary CSV 中检索 kernel_name, 返回匹配行列表.
    每个元素: {"op_name", "op_type", "task_type", "input_shapes", "duration_us"}
    """
    results = []
    with open(op_summary_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return results
        if "Op Name" not in reader.fieldnames or "Task Duration(us)" not in reader.fieldnames:
            print(f"[WARN] CSV 缺少必要列: {op_summary_csv}")
            return results

        for row in reader:
            op_name = row.get("Op Name", "").strip()
            if kernel_name in op_name:
                results.append({
                    "op_name":      op_name,
                    "op_type":      row.get("OP Type", "").strip(),
                    "task_type":    row.get("Task Type", "").strip(),
                    "input_shapes": row.get("Input Shapes", "").strip().strip('"'),
                    "duration_us":  row.get("Task Duration(us)", "").strip(),
                })
    return results


# ============================================================================
# 3. 获取理论极限性能
# ============================================================================
def get_theoretical_perf(golden_path: str) -> float:
    """
    运行理论极限性能脚本, 返回理论极限耗时 (us).
    从打屏输出中解析 "Latency:      xxx us" 获取理论极限值.
    """
    cmd = (
        f'python -m examples.api.operator_api.pytorch_examples.main '
        f'--script {golden_path}'
    )
    print(f"  [CMD] {cmd}")

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr

    for line in combined.splitlines():
        print(f"    | {line}")

    if proc.returncode != 0:
        print(f"  [ERROR] 理论极限性能脚本执行失败 (退出码 {proc.returncode})")
        return -1.0

    # 解析 "Latency:      xxx us"
    match = re.search(r'Latency:\s+([\d.]+)\s*us', combined)
    if not match:
        print("  [ERROR] 未能从输出中解析 Latency 值")
        return -1.0

    theoretical = float(match.group(1))
    print(f"  [INFO] 理论极限性能: {theoretical} us")
    return theoretical


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
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="ratio 阈值 (默认: 2.0), "
                             ">= 阈值输出到 *_above 表, < 阈值输出到 *_below 表")
    parser.add_argument("--result-dir", default="./result",
                        help="msprof 输出目录 (默认: ./result)")
    args = parser.parse_args()

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
    print(f" ratio 阈值 : {args.threshold}")
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
                "actual_duration_us":    "N/A",
                "theoretical_duration_us": "N/A",
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
                "actual_duration_us":    "NOT_FOUND",
                "theoretical_duration_us": "N/A",
                "ratio":                 "N/A",
            })
            continue

        # 获取理论极限性能
        print(f"\n[INFO] {kernel_name}: 获取理论极限性能 (golden: {golden_path})")
        theoretical = get_theoretical_perf(golden_path)

        for rec in perf_records:
            try:
                actual = float(rec["duration_us"])
                if theoretical > 0:
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
                "actual_duration_us":      rec["duration_us"],
                "theoretical_duration_us": theoretical,
                "ratio":                   ratio,
            })

            print(f"  Shape={rec['input_shapes']:30s}  "
                  f"实测={rec['duration_us']:>10s} us  "
                  f"理论={theoretical:>10.3f} us  "
                  f"比值={ratio}")

    # ---- 4.4 按阈值分拣 ----
    threshold = args.threshold
    rows_above = []  # ratio >= 阈值 (未达标)
    rows_below = []  # ratio <  阈值 (达标)

    for row in output_rows:
        r = row["ratio"]
        if isinstance(r, (int, float)):
            if r >= threshold:
                rows_above.append(row)
            else:
                rows_below.append(row)
        else:
            # ratio 为 N/A 等非数值, 归入未达标表
            rows_above.append(row)

    # ---- 4.5 写出三个 CSV ----
    fieldnames = [
        "kernel_name",
        "kernel_path",
        "golden_path",
        "input_shapes",
        "actual_duration_us",
        "theoretical_duration_us",
        "ratio",
    ]

    base, ext = os.path.splitext(args.output)
    out_all   = args.output
    out_above = f"{base}_above{ext}"   # ratio >= 阈值
    out_below = f"{base}_below{ext}"   # ratio <  阈值

    def write_csv(path, rows):
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_all,   output_rows)
    write_csv(out_above, rows_above)
    write_csv(out_below, rows_below)

    print(f"\n{'=' * 64}")
    print(f" 完成! 阈值 = {threshold}")
    print(f"{'─' * 64}")
    print(f" 全量报告     : {os.path.abspath(out_all):50s} ({len(output_rows)} 条)")
    print(f" 未达标 (>={threshold}) : {os.path.abspath(out_above):50s} ({len(rows_above)} 条)")
    print(f" 已达标 (<{threshold})  : {os.path.abspath(out_below):50s} ({len(rows_below)} 条)")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
