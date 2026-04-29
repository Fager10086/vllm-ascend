"""
gdn_attention_core 算子性能基准测试

基于 shape_stats_report.md 中采集的 gdn_attention_core 算子 shape 数据，
对每个 shape 配置进行性能测试，输出 JSON 格式的执行时间数据，
供 roofline_analyzer.py 进行 Roofline 建模分析。

使用方法:
    python bench_gdn_attention_core.py --output ./bench_results
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch_npu


SHAPE_CONFIGS = [
    {"tokens": 1,    "count": 9792, "desc": "decode (bs=1)"},
    {"tokens": 2,    "count": 192,  "desc": "decode (bs=2)"},
    {"tokens": 4,    "count": 192,  "desc": "decode (bs=4)"},
    {"tokens": 8,    "count": 192,  "desc": "decode (bs=8)"},
    {"tokens": 16,   "count": 192,  "desc": "decode (bs=16)"},
    {"tokens": 8192, "count": 96,   "desc": "prefill"},
]

QKV_DIM = 2048
NUM_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16
DEFAULT_WARMUP_ITERS = 10
DEFAULT_BENCH_ITERS = 100


def get_device_name() -> str:
    try:
        return torch.npu.get_device_name(0)
    except Exception:
        return "unknown"


def create_inputs(tokens: int, device: torch.device) -> Dict[str, torch.Tensor]:
    mixed_qkv = torch.randn(tokens, QKV_DIM, dtype=DTYPE, device=device)
    b = torch.randn(tokens, NUM_HEADS, dtype=DTYPE, device=device)
    a = torch.randn(tokens, NUM_HEADS, dtype=DTYPE, device=device)
    core_attn_out = torch.zeros(tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return {
        "mixed_qkv": mixed_qkv,
        "b": b,
        "a": a,
        "core_attn_out": core_attn_out,
    }


def run_gdn_attention_core(inputs: Dict[str, torch.Tensor]) -> None:
    torch.ops.vllm.gdn_attention_core(
        inputs["mixed_qkv"],
        inputs["b"],
        inputs["a"],
        inputs["core_attn_out"],
        "",
    )


def benchmark_single_shape(
    tokens: int,
    device: torch.device,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    bench_iters: int = DEFAULT_BENCH_ITERS,
) -> Dict:
    inputs = create_inputs(tokens, device)

    for _ in range(warmup_iters):
        run_gdn_attention_core(inputs)
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    times_ms = []
    for _ in range(bench_iters):
        inputs = create_inputs(tokens, device)

        start_event.record()
        run_gdn_attention_core(inputs)
        end_event.record()
        torch.npu.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    p90_ms = times_ms[int(len(times_ms) * 0.9)]
    p99_ms = times_ms[int(len(times_ms) * 0.99)]

    input_bytes = (
        tokens * QKV_DIM * 2
        + tokens * NUM_HEADS * 2
        + tokens * NUM_HEADS * 2
        + tokens * NUM_HEADS * HEAD_DIM * 2
    )
    output_bytes = tokens * NUM_HEADS * HEAD_DIM * 2
    total_bytes = input_bytes + output_bytes

    return {
        "tokens": tokens,
        "median_ms": round(median_ms, 4),
        "mean_ms": round(mean_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "p90_ms": round(p90_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "total_bytes": total_bytes,
        "bandwidth_gbps": round(total_bytes / (median_ms * 1e-3) / 1e9, 2) if median_ms > 0 else 0,
    }


def estimate_flops(tokens: int) -> int:
    H = NUM_HEADS
    D = HEAD_DIM
    S = tokens

    qk_flops = tokens * H * S * D * 2
    sv_flops = tokens * H * S * D * 2
    gate_flops = tokens * H * 4
    conv_flops = tokens * H * D * 3 * 2
    total = qk_flops + sv_flops + gate_flops + conv_flops
    return total


def generate_roofline_times(results: List[Dict]) -> Dict[str, float]:
    execution_times = {}
    for r in results:
        tokens = r["tokens"]
        median_ms = r["median_ms"]
        shape_key = f"custom_op.gdn_attention_core_T{tokens}"
        execution_times[shape_key] = median_ms
    return execution_times


def generate_extended_roofline_data(results: List[Dict]) -> List[Dict]:
    records = []
    for r in results:
        tokens = r["tokens"]
        flops = estimate_flops(tokens)
        median_ms = r["median_ms"]
        total_bytes = r["total_bytes"]

        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0
        achieved_perf = flops / (median_ms * 1e-3) if median_ms > 0 else 0

        records.append({
            "op_name": "custom_op.gdn_attention_core",
            "shape_key": f"T={tokens}",
            "tokens": tokens,
            "arg_shapes": [
                {"arg_idx": 0, "shape": [tokens, QKV_DIM], "dtype": "torch.bfloat16"},
                {"arg_idx": 1, "shape": [tokens, NUM_HEADS], "dtype": "torch.bfloat16"},
                {"arg_idx": 2, "shape": [tokens, NUM_HEADS], "dtype": "torch.bfloat16"},
                {"arg_idx": 3, "shape": [tokens, NUM_HEADS, HEAD_DIM], "dtype": "torch.bfloat16"},
            ],
            "output_shapes": [],
            "flops": flops,
            "total_bytes": total_bytes,
            "arithmetic_intensity": round(arithmetic_intensity, 4),
            "execution_time_ms": median_ms,
            "achieved_flops": round(achieved_perf, 0),
            "bandwidth_gbps": r["bandwidth_gbps"],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="gdn_attention_core 算子性能基准测试")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./bench_results",
        help="输出目录路径",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="预热迭代次数",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_BENCH_ITERS,
        help="基准测试迭代次数",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU 设备编号",
    )

    args = parser.parse_args()
    warmup_iters = args.warmup
    bench_iters = args.iters

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    device_name = get_device_name()

    print("=" * 70)
    print("gdn_attention_core 算子性能基准测试")
    print("=" * 70)
    print(f"设备: {device_name} (npu:{args.device})")
    print(f"数据类型: {DTYPE}")
    print(f"预热次数: {warmup_iters}, 测试次数: {bench_iters}")
    print(f"算子参数: QKV_DIM={QKV_DIM}, NUM_HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}")
    print("-" * 70)

    results = []
    for config in SHAPE_CONFIGS:
        tokens = config["tokens"]
        desc = config["desc"]
        count = config["count"]

        print(f"\n测试 Shape: T={tokens} ({desc}), 实际调用次数={count}")

        try:
            result = benchmark_single_shape(tokens, device, warmup_iters, bench_iters)
            result["desc"] = desc
            result["actual_call_count"] = count
            result["flops"] = estimate_flops(tokens)
            result["flops_per_call"] = estimate_flops(tokens)
            results.append(result)

            print(f"  中位数: {result['median_ms']:.4f} ms")
            print(f"  平均值: {result['mean_ms']:.4f} ms")
            print(f"  P90:    {result['p90_ms']:.4f} ms")
            print(f"  P99:    {result['p99_ms']:.4f} ms")
            print(f"  最小值: {result['min_ms']:.4f} ms")
            print(f"  最大值: {result['max_ms']:.4f} ms")
            print(f"  FLOPS:  {result['flops']:,}")
            print(f"  带宽:   {result['bandwidth_gbps']:.2f} GB/s")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            results.append({
                "tokens": tokens,
                "desc": desc,
                "actual_call_count": count,
                "error": str(e),
            })

    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"{'T':>6} | {'中位数(ms)':>10} | {'FLOPS':>14} | {'带宽(GB/s)':>10} | {'AI(FLOP/B)':>10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['tokens']:>6} | {'FAILED':>10} | {'-':>14} | {'-':>10} | {'-':>10}")
        else:
            ai = r["flops"] / r["total_bytes"] if r["total_bytes"] > 0 else 0
            print(f"{r['tokens']:>6} | {r['median_ms']:>10.4f} | {r['flops']:>14,} | {r['bandwidth_gbps']:>10.2f} | {ai:>10.2f}")

    execution_times = generate_roofline_times(results)
    times_path = os.path.join(args.output, "gdn_attention_core_times.json")
    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(execution_times, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Roofline 执行时间数据已保存: {times_path}")

    extended_data = generate_extended_roofline_data(results)
    extended_path = os.path.join(args.output, "gdn_attention_core_roofline.json")
    with open(extended_path, "w", encoding="utf-8") as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 扩展 Roofline 数据已保存: {extended_path}")

    summary = {
        "device": device_name,
        "dtype": str(DTYPE),
        "op_name": "custom_op.gdn_attention_core",
        "parameters": {
            "qkv_dim": QKV_DIM,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        },
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "results": results,
    }
    summary_path = os.path.join(args.output, "gdn_attention_core_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 完整测试报告已保存: {summary_path}")


if __name__ == "__main__":
    main()
