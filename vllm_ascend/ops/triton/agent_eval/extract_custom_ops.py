#!/usr/bin/env python3
"""
从 vllm-ascend 仓库中提取所有自定义算子名称（Triton 和 Torch）。

使用方式:
    python extract_custom_ops.py [--repo-root /path/to/vllm-ascend] [--json]

输出四类算子:
  1. C++ Torch 算子 (TORCH_LIBRARY, namespace: _C_ascend)
  2. Dummy Fusion 占位算子
  3. Python Torch 算子 (direct_register_custom_op, namespace: vllm)
  4. Triton 算子 (@triton.jit kernel)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class OpInfo:
    name: str
    op_type: str        # "cpp_torch" | "python_torch" | "triton" | "dummy_fusion"
    namespace: str      # "_C_ascend" | "vllm" | ""
    file_path: str
    line_number: int
    description: str = ""


def find_cpp_torch_ops(repo_root: Path) -> list[OpInfo]:
    binding_file = repo_root / "csrc" / "torch_binding.cpp"
    if not binding_file.exists():
        return []

    content = binding_file.read_text(errors="replace")
    lines = content.splitlines()
    ops = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if "ops.def(" in line:
            line_no = i + 1
            buf = line
            paren_count = buf.count("(") - buf.count(")")
            j = i + 1
            while paren_count > 0 and j < len(lines):
                buf += " " + lines[j]
                paren_count += lines[j].count("(") - lines[j].count(")")
                j += 1

            m = re.search(r'ops\.def\(\s*"(\w+)', buf)
            if m:
                op_name = m.group(1)
                ops.append(OpInfo(
                    name=op_name,
                    op_type="cpp_torch",
                    namespace="_C_ascend",
                    file_path=str(binding_file.relative_to(repo_root)),
                    line_number=line_no,
                    description=f"torch.ops._C_ascend.{op_name}",
                ))
            i = j
        else:
            i += 1
    return ops


def find_dummy_fusion_ops(repo_root: Path) -> list[OpInfo]:
    init_file = repo_root / "vllm_ascend" / "ops" / "__init__.py"
    if not init_file.exists():
        return []

    content = init_file.read_text(errors="replace")
    ops = []
    for i, line in enumerate(content.splitlines(), 1):
        m = re.search(r'torch\.ops\._C_ascend\.(\w+)\s*=\s*dummyFusionOp', line)
        if m:
            ops.append(OpInfo(
                name=m.group(1),
                op_type="dummy_fusion",
                namespace="_C_ascend",
                file_path=str(init_file.relative_to(repo_root)),
                line_number=i,
                description=f"Dummy fusion op: torch.ops._C_ascend.{m.group(1)}",
            ))
    return ops


def find_python_torch_ops(repo_root: Path) -> list[OpInfo]:
    ops = []
    seen = set()

    for py_file in repo_root.rglob("*.py"):
        if "__pycache__" in str(py_file) or " copy" in py_file.name:
            continue
        try:
            content = py_file.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        rel_path = str(py_file.relative_to(repo_root))

        # Single-line match
        for i, line in enumerate(content.splitlines(), 1):
            m = re.search(r'direct_register_custom_op\(.*op_name\s*=\s*"(\w+)"', line)
            if m:
                key = (m.group(1), rel_path)
                if key not in seen:
                    seen.add(key)
                    ops.append(OpInfo(
                        name=m.group(1),
                        op_type="python_torch",
                        namespace="vllm",
                        file_path=rel_path,
                        line_number=i,
                        description=f"torch.ops.vllm.{m.group(1)}",
                    ))

        # Multi-line match
        for m in re.finditer(
            r'direct_register_custom_op\(\s*\n\s*op_name\s*=\s*"(\w+)"',
            content,
        ):
            line_no = content[:m.start()].count("\n") + 1
            key = (m.group(1), rel_path)
            if key not in seen:
                seen.add(key)
                ops.append(OpInfo(
                    name=m.group(1),
                    op_type="python_torch",
                    namespace="vllm",
                    file_path=rel_path,
                    line_number=line_no,
                    description=f"torch.ops.vllm.{m.group(1)}",
                ))
    return ops


def find_triton_ops(repo_root: Path) -> list[OpInfo]:
    ops = []

    for py_file in repo_root.rglob("*.py"):
        if "__pycache__" in str(py_file) or " copy" in py_file.name:
            continue
        try:
            content = py_file.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        lines = content.splitlines()
        rel_path = str(py_file.relative_to(repo_root))
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r"^@triton\.jit\b", line):
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith("@"):
                        j += 1
                        continue
                    m = re.match(r"def\s+(\w+)\s*\(", next_line)
                    if m:
                        ops.append(OpInfo(
                            name=m.group(1),
                            op_type="triton",
                            namespace="",
                            file_path=rel_path,
                            line_number=j + 1,
                            description=f"Triton kernel: {m.group(1)}",
                        ))
                        break
                    elif next_line == "" or next_line.startswith("#"):
                        j += 1
                        continue
                    else:
                        break
                i = j + 1
            else:
                i += 1
    return ops


def print_table(title, ops):
    if not ops:
        print(f"\n{'=' * 60}")
        print(f"  {title} (none)")
        print(f"{'=' * 60}")
        return

    print(f"\n{'=' * 80}")
    print(f"  {title} ({len(ops)} ops)")
    print(f"{'=' * 80}")

    max_name = max(len(o.name) for o in ops)
    max_ns = max((len(o.namespace) for o in ops), default=0)
    max_file = max(len(o.file_path) for o in ops)

    cn = max(max_name, 7)
    cns = max(max_ns, 9) if max_ns else 0
    cf = max(max_file, 4)

    if cns:
        print(f"  {'#':<4} {'Op Name':<{cn}}  {'Namespace':<{cns}}  {'File':<{cf}}  Line")
        print(f"  {'---':<4} {'-'*cn}  {'-'*cns}  {'-'*cf}  ----")
    else:
        print(f"  {'#':<4} {'Op Name':<{cn}}  {'File':<{cf}}  Line")
        print(f"  {'---':<4} {'-'*cn}  {'-'*cf}  ----")

    for idx, op in enumerate(ops, 1):
        if cns:
            print(f"  {idx:<4} {op.name:<{cn}}  {op.namespace:<{cns}}  {op.file_path:<{cf}}  {op.line_number}")
        else:
            print(f"  {idx:<4} {op.name:<{cn}}  {op.file_path:<{cf}}  {op.line_number}")


TITLES = {
    "cpp_torch":    "C++ Torch Ops (TORCH_LIBRARY, ns: _C_ascend)",
    "dummy_fusion": "Dummy Fusion Placeholder Ops (ns: _C_ascend)",
    "python_torch": "Python Torch Ops (direct_register_custom_op, ns: vllm)",
    "triton":       "Triton Kernels (@triton.jit)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract all custom op names (Triton & Torch) from vllm-ascend"
    )
    parser.add_argument(
        "--repo-root", type=str, default="/vllm-workspace/vllm-ascend",
        help="vllm-ascend repo root (default: /vllm-workspace/vllm-ascend)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--type",
        choices=["all", "cpp_torch", "python_torch", "triton", "dummy_fusion"],
        default="all",
        help="Filter by op type (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"[ERROR] Repo path does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)

    all_ops = {}
    if args.type in ("all", "cpp_torch"):
        all_ops["cpp_torch"] = find_cpp_torch_ops(repo_root)
    if args.type in ("all", "dummy_fusion"):
        all_ops["dummy_fusion"] = find_dummy_fusion_ops(repo_root)
    if args.type in ("all", "python_torch"):
        all_ops["python_torch"] = find_python_torch_ops(repo_root)
    if args.type in ("all", "triton"):
        all_ops["triton"] = find_triton_ops(repo_root)

    if args.json:
        result = {}
        for cat, ops in all_ops.items():
            result[cat] = [asdict(op) for op in ops]
        result["summary"] = {cat: len(ops) for cat, ops in all_ops.items()}
        result["summary"]["total"] = sum(len(ops) for ops in all_ops.values())
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()  # trailing newline
    else:
        print(f"Scanning: {repo_root}")
        print("-" * 80)

        for cat, ops in all_ops.items():
            print_table(TITLES.get(cat, cat), ops)

        total = sum(len(ops) for ops in all_ops.values())
        print(f"\n{'=' * 80}")
        print("  Summary")
        print(f"{'=' * 80}")
        for cat, ops in all_ops.items():
            label = TITLES.get(cat, cat).split("(")[0].strip()
            print(f"  {label}: {len(ops)}")
        print(f"  {'-' * 40}")
        print(f"  Total: {total} custom ops")


if __name__ == "__main__":
    main()
