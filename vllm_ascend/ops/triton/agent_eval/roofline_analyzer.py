"""
Roofline 性能分析工具

基于 Roofline 模型分析算子性能，生成性能评价报告。

使用方法:
    python roofline_analyzer.py --input ./output/shape_records_merged.jsonl --output ./roofline_report
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OpShapeInfo:
    op_name: str
    arg_shapes: List[Dict]
    kwarg_shapes: List[Dict]
    output_shapes: List[Dict]
    call_count: int = 1


@dataclass
class RooflinePoint:
    op_name: str
    arithmetic_intensity: float
    achieved_performance: float
    peak_performance: float
    peak_bandwidth: float
    bottleneck_type: str
    optimization_potential: float
    shape_info: str


class AscendHardwareConfig:
    PEAK_FLOPS_FP16 = 310e12
    PEAK_FLOPS_BF16 = 620e12
    PEAK_FLOPS_FP32 = 155e12
    PEAK_BANDWIDTH = 1.2e12

    DTYPE_BYTES = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.int8": 1,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.bool": 1,
    }


class FLOPsCalculator:
    @staticmethod
    def get_dtype_bytes(dtype_str: str) -> int:
        dtype_str = dtype_str.replace("torch.", "")
        dtype_map = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int8": 1,
            "int32": 4,
            "int64": 8,
            "bool": 1,
        }
        return dtype_map.get(dtype_str, 4)

    @staticmethod
    def calculate_matmul_flops(
        M: int, N: int, K: int, batch_size: int = 1
    ) -> Tuple[int, int]:
        flops = 2 * batch_size * M * N * K
        input_bytes = batch_size * (M * K + K * N) * 2
        output_bytes = batch_size * M * N * 2
        total_bytes = input_bytes + output_bytes
        return flops, total_bytes

    @staticmethod
    def calculate_rmsnorm_flops(N: int, batch_size: int = 1) -> Tuple[int, int]:
        flops = batch_size * N * 5
        input_bytes = batch_size * N * 2
        output_bytes = batch_size * N * 2
        total_bytes = input_bytes + output_bytes
        return flops, total_bytes

    @staticmethod
    def calculate_layernorm_flops(N: int, batch_size: int = 1) -> Tuple[int, int]:
        flops = batch_size * N * 10
        input_bytes = batch_size * N * 2
        output_bytes = batch_size * N * 2
        total_bytes = input_bytes + output_bytes
        return flops, total_bytes

    @staticmethod
    def calculate_softmax_flops(N: int, batch_size: int = 1) -> Tuple[int, int]:
        flops = batch_size * N * 5
        input_bytes = batch_size * N * 2
        output_bytes = batch_size * N * 2
        total_bytes = input_bytes + output_bytes
        return flops, total_bytes

    @staticmethod
    def calculate_attention_flops(
        B: int, H: int, S: int, D: int, causal: bool = False
    ) -> Tuple[int, int]:
        if causal:
            flops = B * H * S * S * D
        else:
            flops = 2 * B * H * S * S * D
        input_bytes = B * H * (S * D * 2 + S * D * 2)
        output_bytes = B * H * S * D * 2
        total_bytes = input_bytes + output_bytes
        return flops, total_bytes

    @staticmethod
    def calculate_elementwise_flops(N: int, batch_size: int = 1) -> Tuple[int, int]:
        flops = batch_size * N
        total_bytes = batch_size * N * 4
        return flops, total_bytes

    @staticmethod
    def calculate_gemm_flops(M: int, N: int, K: int) -> Tuple[int, int]:
        return FLOPsCalculator.calculate_matmul_flops(M, N, K)

    @staticmethod
    def estimate_from_shapes(
        op_name: str, arg_shapes: List[Dict], output_shapes: List[Dict]
    ) -> Tuple[int, int]:
        flops = 0
        total_bytes = 0

        def get_shape_size(shape_dict: Dict) -> int:
            shape = shape_dict.get("shape", [])
            return int(np.prod(shape)) if shape else 0

        def get_dtype_size(shape_dict: Dict) -> int:
            dtype = shape_dict.get("dtype", "torch.float16")
            return FLOPsCalculator.get_dtype_bytes(dtype)

        input_bytes = sum(
            get_shape_size(s) * get_dtype_size(s) for s in arg_shapes
        )
        output_bytes = sum(
            get_shape_size(s) * get_dtype_size(s) for s in output_shapes
        )
        total_bytes = input_bytes + output_bytes

        op_lower = op_name.lower()

        if "matmul" in op_lower or "linear" in op_lower or "bmm" in op_lower:
            if len(arg_shapes) >= 2:
                shape1 = arg_shapes[0].get("shape", [])
                shape2 = arg_shapes[1].get("shape", [])
                if len(shape1) >= 2 and len(shape2) >= 2:
                    M = shape1[-2] if len(shape1) >= 2 else 1
                    K = shape1[-1] if len(shape1) >= 1 else 1
                    N = shape2[-1] if len(shape2) >= 1 else 1
                    batch = int(np.prod(shape1[:-2])) if len(shape1) > 2 else 1
                    flops, _ = FLOPsCalculator.calculate_matmul_flops(
                        M, N, K, batch
                    )

        elif "rms_norm" in op_lower or "rmsnorm" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 5

        elif "layer_norm" in op_lower or "layernorm" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 10

        elif "softmax" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 5

        elif "attention" in op_lower or "flash_attn" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 10

        elif "silu" in op_lower or "sigmoid" in op_lower or "gelu" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 4

        elif "swiglu" in op_lower or "swi_glu" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 6

        elif "add" in op_lower or "sub" in op_lower or "mul" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements

        elif "cat" in op_lower or "concat" in op_lower:
            flops = 0

        elif "transpose" in op_lower or "permute" in op_lower or "reshape" in op_lower:
            flops = 0

        elif "rope" in op_lower or "rotary" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 10

        elif "moe" in op_lower or "gating" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        elif "all_reduce" in op_lower or "all_gather" in op_lower:
            flops = 0

        elif "npu_add_rms_norm" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 6

        elif "npu_sparse_flash_attention" in op_lower or "sparse_attention" in op_lower:
            if len(arg_shapes) >= 1:
                shape = arg_shapes[0].get("shape", [])
                if len(shape) >= 3:
                    B, H, S = shape[0], shape[1], shape[2]
                    D = shape[3] if len(shape) > 3 else 128
                    flops = 2 * B * H * S * S * D * 0.1
                else:
                    flops = sum(get_shape_size(s) for s in arg_shapes) * 10
            else:
                flops = 0

        elif "mla_preprocess" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 20

        elif "bgmv" in op_lower or "sgmv" in op_lower:
            if len(arg_shapes) >= 2:
                x_shape = arg_shapes[0].get("shape", [])
                w_shape = arg_shapes[1].get("shape", [])
                if len(x_shape) >= 2 and len(w_shape) >= 2:
                    M = x_shape[0] if x_shape else 1
                    K = x_shape[-1] if x_shape else 1
                    N = w_shape[-1] if w_shape else 1
                    flops = 2 * M * N * K
                else:
                    flops = sum(get_shape_size(s) for s in arg_shapes) * 2
            else:
                flops = 0

        elif "chunk" in op_lower and "delta" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 50

        elif "causal_conv1d" in op_lower:
            if len(arg_shapes) >= 2:
                x_shape = arg_shapes[0].get("shape", [])
                w_shape = arg_shapes[1].get("shape", [])
                if len(x_shape) >= 2 and len(w_shape) >= 2:
                    B, L, C = x_shape[0], x_shape[1], x_shape[2]
                    K = w_shape[2] if len(w_shape) > 2 else 3
                    flops = B * L * C * K * 2
                else:
                    flops = sum(get_shape_size(s) for s in arg_shapes) * 2
            else:
                flops = 0

        elif "lightning_indexer" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        elif "quantize" in op_lower or "quant" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 3

        elif "dequantize" in op_lower or "dequant" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        elif "index_select" in op_lower or "gather" in op_lower:
            flops = 0

        elif "unique" in op_lower:
            flops = 0

        elif "split" in op_lower:
            flops = 0

        elif "remainder" in op_lower or "mod" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        elif "max" in op_lower or "min" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements

        elif "cos" in op_lower or "sin" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 10

        elif "l2norm" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 5

        elif "fused_gdn_gating" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 10

        elif "solve_tril" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 20

        elif "recompute_w_u" in op_lower or "wy_fast" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 30

        elif "chunk_fwd_o" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 20

        elif "dispatch_ffn" in op_lower or "gmm_combine" in op_lower:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        elif "matmul_allreduce" in op_lower:
            if len(arg_shapes) >= 2:
                shape1 = arg_shapes[0].get("shape", [])
                shape2 = arg_shapes[1].get("shape", [])
                if len(shape1) >= 2 and len(shape2) >= 2:
                    M = shape1[-2] if len(shape1) >= 2 else 1
                    K = shape1[-1] if len(shape1) >= 1 else 1
                    N = shape2[-1] if len(shape2) >= 1 else 1
                    flops = 2 * M * N * K
                else:
                    flops = sum(get_shape_size(s) for s in arg_shapes) * 2
            else:
                flops = 0

        elif "mm_reduce_scatter" in op_lower:
            if len(arg_shapes) >= 2:
                shape1 = arg_shapes[0].get("shape", [])
                shape2 = arg_shapes[1].get("shape", [])
                if len(shape1) >= 2 and len(shape2) >= 2:
                    M = shape1[-2] if len(shape1) >= 2 else 1
                    K = shape1[-1] if len(shape1) >= 1 else 1
                    N = shape2[-1] if len(shape2) >= 1 else 1
                    flops = 2 * M * N * K
                else:
                    flops = sum(get_shape_size(s) for s in arg_shapes) * 2
            else:
                flops = 0

        else:
            total_elements = sum(get_shape_size(s) for s in arg_shapes)
            flops = total_elements * 2

        return flops, total_bytes


class RooflineAnalyzer:
    def __init__(
        self,
        peak_flops: float = AscendHardwareConfig.PEAK_FLOPS_FP16,
        peak_bandwidth: float = AscendHardwareConfig.PEAK_BANDWIDTH,
    ):
        self.peak_flops = peak_flops
        self.peak_bandwidth = peak_bandwidth
        self.op_records: Dict[str, List[OpShapeInfo]] = defaultdict(list)
        self.roofline_points: List[RooflinePoint] = []

    def load_shape_records(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Shape records file not found: {filepath}")

        records_by_op = defaultdict(lambda: defaultdict(int))

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    op_name = record.get("op_name", "unknown")
                    shape_key = json.dumps(
                        {
                            "args": record.get("arg_shapes", []),
                            "kwargs": record.get("kwarg_shapes", []),
                            "outputs": record.get("output_shapes", []),
                        },
                        sort_keys=True,
                    )
                    records_by_op[op_name][shape_key] += 1
                except json.JSONDecodeError:
                    continue

        for op_name, shape_counts in records_by_op.items():
            for shape_key, count in shape_counts.items():
                shape_data = json.loads(shape_key)
                op_info = OpShapeInfo(
                    op_name=op_name,
                    arg_shapes=shape_data["args"],
                    kwarg_shapes=shape_data["kwargs"],
                    output_shapes=shape_data["outputs"],
                    call_count=count,
                )
                self.op_records[op_name].append(op_info)

        print(f"[RooflineAnalyzer] 加载了 {len(self.op_records)} 个算子的 shape 信息")

    def analyze_operator(
        self, op_info: OpShapeInfo, execution_time_ms: Optional[float] = None
    ) -> RooflinePoint:
        flops, total_bytes = FLOPsCalculator.estimate_from_shapes(
            op_info.op_name, op_info.arg_shapes, op_info.output_shapes
        )

        if total_bytes == 0:
            arithmetic_intensity = 0
        else:
            arithmetic_intensity = flops / total_bytes

        if execution_time_ms and execution_time_ms > 0:
            achieved_performance = flops / (execution_time_ms * 1e-3)
        else:
            achieved_performance = 0

        roofline_limit = min(
            self.peak_flops, arithmetic_intensity * self.peak_bandwidth
        )

        if arithmetic_intensity < self.peak_flops / self.peak_bandwidth:
            bottleneck_type = "Memory Bound"
        else:
            bottleneck_type = "Compute Bound"

        if roofline_limit > 0 and achieved_performance > 0:
            optimization_potential = (
                roofline_limit - achieved_performance
            ) / roofline_limit
        else:
            optimization_potential = 0

        shape_str = self._format_shapes(op_info)

        return RooflinePoint(
            op_name=op_info.op_name,
            arithmetic_intensity=arithmetic_intensity,
            achieved_performance=achieved_performance,
            peak_performance=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            bottleneck_type=bottleneck_type,
            optimization_potential=optimization_potential,
            shape_info=shape_str,
        )

    def _format_shapes(self, op_info: OpShapeInfo) -> str:
        parts = []
        for i, arg in enumerate(op_info.arg_shapes[:3]):
            shape = arg.get("shape", [])
            dtype = arg.get("dtype", "unknown")
            parts.append(f"arg{i}={shape}")
        return ", ".join(parts)

    def analyze_all(self, execution_times: Optional[Dict[str, float]] = None):
        self.roofline_points = []

        for op_name, op_infos in self.op_records.items():
            for op_info in op_infos:
                exec_time = None
                if execution_times:
                    exec_time = execution_times.get(op_name)

                point = self.analyze_operator(op_info, exec_time)
                self.roofline_points.append(point)

        print(f"[RooflineAnalyzer] 分析了 {len(self.roofline_points)} 个算子实例")

    def plot_roofline(self, output_path: str):
        if not self.roofline_points:
            print("[RooflineAnalyzer] 没有数据可绘制")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        intensities = np.logspace(-1, 4, 100)
        memory_bound = intensities * self.peak_bandwidth
        memory_bound = np.minimum(memory_bound, self.peak_flops)
        ax.loglog(intensities, memory_bound, "b-", linewidth=2, label="Memory Bound")
        ax.axhline(y=self.peak_flops, color="r", linestyle="--", linewidth=2, label=f"Peak Performance ({self.peak_flops/1e12:.0f} TFLOPS)")

        ridge_point = self.peak_flops / self.peak_bandwidth
        ax.axvline(x=ridge_point, color="g", linestyle=":", linewidth=1, label=f"Ridge Point ({ridge_point:.1f} FLOP/Byte)")

        op_colors = {}
        unique_ops = list(set(p.op_name for p in self.roofline_points))
        colormap = plt.cm.get_cmap("tab20", len(unique_ops))
        for i, op in enumerate(unique_ops):
            op_colors[op] = colormap(i)

        for point in self.roofline_points:
            if point.arithmetic_intensity > 0:
                color = op_colors[point.op_name]
                marker = "o" if point.bottleneck_type == "Memory Bound" else "s"
                ax.scatter(
                    point.arithmetic_intensity,
                    point.achieved_performance if point.achieved_performance > 0 else self.peak_flops * 0.1,
                    c=[color],
                    marker=marker,
                    s=100,
                    alpha=0.7,
                    label=point.op_name if point.op_name not in [p.op_name for p in self.roofline_points[:self.roofline_points.index(point)]] else "",
                )

        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
        ax.set_ylabel("Performance (FLOP/s)", fontsize=12)
        ax.set_title("Roofline Model - Ascend NPU", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.98,
            f"Peak Bandwidth: {self.peak_bandwidth/1e12:.1f} TB/s\n"
            f"Peak Performance: {self.peak_flops/1e12:.0f} TFLOPS\n"
            f"Ridge Point: {ridge_point:.1f} FLOP/Byte",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[RooflineAnalyzer] Roofline 图已保存到: {output_path}")

    def generate_report(self, output_path: str):
        if not self.roofline_points:
            print("[RooflineAnalyzer] 没有数据可生成报告")
            return

        report_lines = []
        report_lines.append("# Roofline 性能分析报告\n")
        report_lines.append(f"## 硬件配置\n")
        report_lines.append(f"- 峰值算力: {self.peak_flops/1e12:.0f} TFLOPS")
        report_lines.append(f"- 峰值带宽: {self.peak_bandwidth/1e12:.1f} TB/s")
        report_lines.append(f"- Ridge Point: {self.peak_flops/self.peak_bandwidth:.1f} FLOP/Byte\n")

        ridge_point = self.peak_flops / self.peak_bandwidth

        op_stats = defaultdict(lambda: {
            "count": 0,
            "intensities": [],
            "shapes": [],
            "bottleneck": "",
            "total_flops": 0,
            "total_bytes": 0,
        })
        for p in self.roofline_points:
            op_stats[p.op_name]["count"] += 1
            op_stats[p.op_name]["intensities"].append(p.arithmetic_intensity)
            op_stats[p.op_name]["shapes"].append(p.shape_info)
            op_stats[p.op_name]["bottleneck"] = p.bottleneck_type

        report_lines.append("## 🔥 需要优化的算子列表\n")
        report_lines.append("> 按优化优先级排序，综合考虑调用频率、算术强度和瓶颈类型\n")

        optimization_candidates = []
        for op_name, stats in op_stats.items():
            intensities = stats["intensities"]
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            min_intensity = min(intensities) if intensities else 0
            
            if min_intensity < ridge_point and max(intensities) >= ridge_point:
                bottleneck_type = "混合"
            elif min_intensity < ridge_point:
                bottleneck_type = "内存瓶颈"
            else:
                bottleneck_type = "计算瓶颈"
            
            priority_score = 0
            priority_reasons = []
            
            if bottleneck_type == "内存瓶颈":
                if avg_intensity < 10:
                    priority_score += 100
                    priority_reasons.append("极低算术强度")
                elif avg_intensity < 50:
                    priority_score += 50
                    priority_reasons.append("低算术强度")
                
                if stats["count"] > 100:
                    priority_score += 30
                    priority_reasons.append("高频调用")
                elif stats["count"] > 10:
                    priority_score += 10
                    priority_reasons.append("中频调用")
            elif bottleneck_type == "计算瓶颈":
                if stats["count"] > 100:
                    priority_score += 20
                    priority_reasons.append("高频计算密集")
            else:
                if stats["count"] > 50:
                    priority_score += 15
                    priority_reasons.append("混合瓶颈高频")
            
            optimization_candidates.append({
                "op_name": op_name,
                "count": stats["count"],
                "avg_intensity": avg_intensity,
                "min_intensity": min_intensity,
                "max_intensity": max(intensities) if intensities else 0,
                "bottleneck_type": bottleneck_type,
                "priority_score": priority_score,
                "priority_reasons": priority_reasons,
            })

        optimization_candidates.sort(key=lambda x: (-x["priority_score"], -x["count"]))

        high_priority = [c for c in optimization_candidates if c["priority_score"] >= 50]
        medium_priority = [c for c in optimization_candidates if 20 <= c["priority_score"] < 50]
        low_priority = [c for c in optimization_candidates if c["priority_score"] > 0 and c["priority_score"] < 20]

        if high_priority:
            report_lines.append(f"\n### 🚨 高优先级 ({len(high_priority)} 个)\n")
            report_lines.append("| 算子 | 调用次数 | 平均强度 | 瓶颈类型 | 优化原因 |")
            report_lines.append("|------|----------|----------|----------|----------|")
            for c in high_priority:
                reasons_str = ", ".join(c["priority_reasons"])
                report_lines.append(
                    f"| {c['op_name']} | {c['count']} | {c['avg_intensity']:.2f} | {c['bottleneck_type']} | {reasons_str} |"
                )

        if medium_priority:
            report_lines.append(f"\n### ⚠️ 中优先级 ({len(medium_priority)} 个)\n")
            report_lines.append("| 算子 | 调用次数 | 平均强度 | 瓶颈类型 | 优化原因 |")
            report_lines.append("|------|----------|----------|----------|----------|")
            for c in medium_priority:
                reasons_str = ", ".join(c["priority_reasons"])
                report_lines.append(
                    f"| {c['op_name']} | {c['count']} | {c['avg_intensity']:.2f} | {c['bottleneck_type']} | {reasons_str} |"
                )

        if low_priority:
            report_lines.append(f"\n### 📋 低优先级 ({len(low_priority)} 个)\n")
            report_lines.append("| 算子 | 调用次数 | 平均强度 | 瓶颈类型 | 优化原因 |")
            report_lines.append("|------|----------|----------|----------|----------|")
            for c in low_priority[:10]:
                reasons_str = ", ".join(c["priority_reasons"])
                report_lines.append(
                    f"| {c['op_name']} | {c['count']} | {c['avg_intensity']:.2f} | {c['bottleneck_type']} | {reasons_str} |"
                )

        report_lines.append("\n## 算子性能分析\n")

        sorted_points = sorted(
            self.roofline_points, key=lambda p: p.arithmetic_intensity, reverse=True
        )

        memory_bound_ops = [p for p in sorted_points if p.bottleneck_type == "Memory Bound"]
        compute_bound_ops = [p for p in sorted_points if p.bottleneck_type == "Compute Bound"]

        report_lines.append(f"### 内存瓶颈算子 ({len(memory_bound_ops)} 个)\n")
        report_lines.append("| 算子 | 算术强度 | 瓶颈类型 | Shape |")
        report_lines.append("|------|----------|----------|-------|")
        for p in memory_bound_ops[:20]:
            report_lines.append(
                f"| {p.op_name} | {p.arithmetic_intensity:.2f} | {p.bottleneck_type} | {p.shape_info} |"
            )

        report_lines.append(f"\n### 计算瓶颈算子 ({len(compute_bound_ops)} 个)\n")
        report_lines.append("| 算子 | 算术强度 | 瓶颈类型 | Shape |")
        report_lines.append("|------|----------|----------|-------|")
        for p in compute_bound_ops[:20]:
            report_lines.append(
                f"| {p.op_name} | {p.arithmetic_intensity:.2f} | {p.bottleneck_type} | {p.shape_info} |"
            )

        report_lines.append("\n## 优化建议\n")

        if memory_bound_ops:
            report_lines.append("### 内存瓶颈算子优化建议")
            report_lines.append("- 减少内存访问次数")
            report_lines.append("- 优化数据布局（如使用 NZ 格式）")
            report_lines.append("- 算子融合减少中间结果存储")
            report_lines.append("- 使用更大的 batch size 提高数据复用\n")

        if compute_bound_ops:
            report_lines.append("### 计算瓶颈算子优化建议")
            report_lines.append("- 使用专用加速器（如 Cube 单元）")
            report_lines.append("- 提高并行度")
            report_lines.append("- 优化计算算法减少冗余计算\n")

        report_lines.append("## 算子统计\n")
        report_lines.append("| 算子 | 调用次数 | 最小强度 | 最大强度 | 平均强度 | 中位数 | 瓶颈类型 |")
        report_lines.append("|------|----------|----------|----------|----------|--------|----------|")
        for op_name, stats in sorted(op_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            intensities = stats["intensities"]
            min_intensity = min(intensities) if intensities else 0
            max_intensity = max(intensities) if intensities else 0
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            median_intensity = sorted(intensities)[len(intensities)//2] if intensities else 0
            
            if min_intensity < ridge_point and max_intensity >= ridge_point:
                bottleneck_type = "混合"
            elif min_intensity < ridge_point:
                bottleneck_type = "内存瓶颈"
            else:
                bottleneck_type = "计算瓶颈"
            
            report_lines.append(
                f"| {op_name} | {stats['count']} | {min_intensity:.2f} | {max_intensity:.2f} | {avg_intensity:.2f} | {median_intensity:.2f} | {bottleneck_type} |"
            )

        report_lines.append("\n## 算子 Shape 分布详情\n")
        report_lines.append("> 展示每个算子不同 Shape 的算术强度分布\n")
        
        for op_name, stats in sorted(op_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
            report_lines.append(f"\n### {op_name} (调用 {stats['count']} 次)\n")
            
            shape_intensity_map = defaultdict(list)
            for intensity, shape in zip(stats["intensities"], stats["shapes"]):
                shape_intensity_map[shape].append(intensity)
            
            report_lines.append("| Shape | 调用次数 | 平均强度 | 最小 | 最大 | 瓶颈类型 |")
            report_lines.append("|-------|----------|----------|------|------|----------|")
            
            sorted_shapes = sorted(shape_intensity_map.items(), key=lambda x: len(x[1]), reverse=True)
            for shape, intensities in sorted_shapes[:5]:
                count = len(intensities)
                avg_i = sum(intensities) / count
                min_i = min(intensities)
                max_i = max(intensities)
                
                if avg_i < ridge_point:
                    bt = "内存"
                else:
                    bt = "计算"
                
                report_lines.append(f"| {shape} | {count} | {avg_i:.2f} | {min_i:.2f} | {max_i:.2f} | {bt} |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"[RooflineAnalyzer] 报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Roofline 性能分析工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入文件路径 (shape_records_merged.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./roofline_report",
        help="输出目录路径",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=AscendHardwareConfig.PEAK_FLOPS_FP16,
        help="峰值算力 (FLOPS)",
    )
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=AscendHardwareConfig.PEAK_BANDWIDTH,
        help="峰值带宽 (Bytes/s)",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    analyzer = RooflineAnalyzer(
        peak_flops=args.peak_flops,
        peak_bandwidth=args.peak_bandwidth,
    )

    analyzer.load_shape_records(args.input)
    analyzer.analyze_all()

    analyzer.plot_roofline(os.path.join(args.output, "roofline.png"))
    analyzer.generate_report(os.path.join(args.output, "roofline_report.md"))


if __name__ == "__main__":
    main()

