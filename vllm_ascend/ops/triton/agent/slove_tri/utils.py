# Copyright(c) 2025 Huawei Technologies Co.; Ltd.
import os
import glob
import torch_npu
import pandas as pd

from typing import Callable, Any
from enum import Enum


class OpType(Enum):
    PURE_CUBE = "c"  # pure cube op
    PURE_VECTOR = "v"  # pure vecotr op
    CV = "cv"  # cv fused op
    OTHER = "other"  # composition of small op


# reference: https://gitcode.com/Ascend/triton-ascend-kernels/blob/master/CONTRIBUTING.md
REFERENCE_OP_THRESHOLDS = {
    OpType.PURE_CUBE: 0.9,  # 0.9x AscendC
    OpType.PURE_VECTOR: 0.9,  # 0.9x AscendC
    OpType.CV: 0.7,  # 0.7x AscendC
    OpType.OTHER: 1.0,  # 1.0x AscendC
}


class Benchmark:
    def __init__(
        self,
        triton_op: Callable[[], Any],
        torch_op: Callable[[], Any],
        op_type: OpType,
        wait: int = 1,
        warmup: int = 6,
        active: int = 10,
        repeat: int = 1,
        skip_first: int = 1,
    ):
        self.triton_op = triton_op
        self.torch_op = torch_op
        self.torch_perf_file_path = os.path.join(
            os.path.join(os.getcwd(), "perf_traces"), "Torch"
        )
        self.triton_perf_file_path = os.path.join(
            os.path.join(os.getcwd(), "perf_traces"), "Triton"
        )
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.skip_first = skip_first
        self.op_type: OpType = op_type if op_type is not None else OpType.OTHER

    def run_npu_profiler(
        self,
        fn: Callable[[], Any],
        trace_file,
    ):
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            schedule=torch_npu.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
                skip_first=self.skip_first,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_file),
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(
                self.repeat * (self.wait + self.warmup + self.active) + self.skip_first
            ):
                torch_npu.npu.synchronize()
                fn()
                torch_npu.npu.synchronize()
                prof.step()

    def get_latest_op_summary_csvs(self, base_dir):
        """
        base_dir: perf_traces/Torch or perf_traces/Triton
        """
        candidates = []

        for name in os.listdir(base_dir):
            if name.endswith("_ascend_pt"):
                full = os.path.join(base_dir, name)
                if os.path.isdir(full):
                    candidates.append(full)

        if not candidates:
            raise RuntimeError(
                f"[Benchmark] No *_ascend_pt directory found under {base_dir}"
            )

        # sort by timestamp
        candidates.sort(key=lambda x: x.split("_")[-3])
        print(f"[Benchmark] Current Dir:{candidates[-1]}")

        pattern = os.path.join(
            candidates[
                -1
            ],  # take the file with largest timestamp, which is the most recently generated one
            "PROF_*",
            "mindstudio_profiler_output",
            "op_summary*.csv",
        )

        matches = glob.glob(pattern)
        if not matches:
            raise RuntimeError(
                f"[Benchmark] No op_summary csv found under {candidates[-1]}"
            )

        return matches

    def run(self):
        # run profiler
        print(f"\n[Benchmark] {'='*40} Run Torch Operation {'='*40}")
        self.run_npu_profiler(fn=self.torch_op, trace_file=self.torch_perf_file_path)
        print(f"[Benchmark] {'='*40} Run Triton Operation {'='*40}")
        self.run_npu_profiler(fn=self.triton_op, trace_file=self.triton_perf_file_path)

        torch_csv_list = self.get_latest_op_summary_csvs(self.torch_perf_file_path)
        triton_csv_list = self.get_latest_op_summary_csvs(self.triton_perf_file_path)
        torch_sum = 0.0
        triton_sum = 0.0
        for csv in torch_csv_list:
            torch_data = pd.read_csv(csv)
            if "Task Duration(us)" not in torch_data.columns:
                raise ValueError(f"'task duration' column not found !")
            torch_col = torch_data["Task Duration(us)"]
            torch_sum += torch_col.sum()

        for csv in triton_csv_list:
            triton_data = pd.read_csv(csv)
            if "Task Duration(us)" not in triton_data.columns:
                raise ValueError(f"'task duration' column not found !")
            triton_col = triton_data["Task Duration(us)"]
            triton_sum += triton_col.sum()

        # calculate speedup
        torch_average = torch_sum / self.active
        triton_average = triton_sum / self.active
        speedup = torch_average / triton_average

        # assert
        if self.op_type in (
            OpType.PURE_CUBE,
            OpType.PURE_VECTOR,
            OpType.CV,
            OpType.OTHER,
        ):
            threshold = REFERENCE_OP_THRESHOLDS[self.op_type]
        else:
            raise ValueError(f"[Benckmark] unknown op type: {self.op_type}")

        print(
            f"[Benckmark] Result: SpeedUp: {speedup:.6f}|Threshold: {threshold}|Torch: {torch_average}(us)|Triton: {triton_average}(us)"
        )
        assert speedup >= threshold, f"Performacne not meeting threshold!"
