#!/usr/bin/env python3
"""
Parse MindStudio profiler op_summary CSV and calculate kernel duration statistics.

Usage:
    python parse_op_summary.py <prof_path> <warm_up> <kernel1> [kernel2] [...]

Example:
    python parse_op_summary.py /vllm-workspace/result/PROF_000001_20260322084801113_JNCPFMRRPJLDFMDC 2 kernel_a kernel_b
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path


def find_op_summary_csv(prof_path: str) -> str:
    """Find the op_summary*.csv file in the mindstudio_profiler_output folder."""
    search_pattern = os.path.join(prof_path, "mindstudio_profiler_output", "op_summary_*.csv")
    matches = glob.glob(search_pattern)
    
    if not matches:
        raise FileNotFoundError(f"No op_summary*.csv found in {prof_path}/mindstudio_profiler_output/")
    
    if len(matches) > 1:
        print(f"Warning: Multiple op_summary files found, using: {matches[0]}")
    
    return matches[0]


def parse_kernel_durations(csv_path: str, kernel_names: list[str], warm_up: int) -> dict[str, float]:
    """
    Parse op_summary CSV and calculate average Task Duration(us) for specified kernels.
    
    Args:
        csv_path: Path to the op_summary CSV file
        kernel_names: List of kernel names to search for
        warm_up: Number of initial calls to exclude for each kernel
    
    Returns:
        Dict mapping kernel name to its average Task Duration(us)
    """
    kernel_durations: dict[str, list[float]] = {name: [] for name in kernel_names}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # First row is title/header
        header = next(reader)
        
        # Find column indices
        try:
            op_name_idx = header.index('Op Name')
            task_duration_idx = header.index('Task Duration(us)')
        except ValueError as e:
            raise ValueError(f"Required column not found in CSV header. Header: {header}") from e
        
        # Parse data rows
        for row in reader:
            if len(row) <= max(op_name_idx, task_duration_idx):
                continue
            
            op_name = row[op_name_idx].strip()
            
            # Check if this row matches any of our target kernels
            for kernel_name in kernel_names:
                if op_name == kernel_name:
                    try:
                        duration = float(row[task_duration_idx].strip())
                        kernel_durations[kernel_name].append(duration)
                    except (ValueError, IndexError):
                        # Skip rows with invalid duration
                        continue
    
    # Calculate averages after excluding warm_up calls
    results = {}
    for kernel_name in kernel_names:
        durations = kernel_durations[kernel_name]
        if len(durations) <= warm_up:
            print(f"Warning: Kernel '{kernel_name}' has {len(durations)} calls, "
                  f"but warm_up={warm_up}. Using all {len(durations)} calls.", 
                  file=sys.stderr)
            avg_duration = sum(durations) / len(durations) if durations else 0.0
        else:
            remaining_durations = durations[warm_up:]
            avg_duration = sum(remaining_durations) / len(remaining_durations)
        
        results[kernel_name] = avg_duration
        print(f"  {kernel_name}: {len(durations)} calls, "
              f"excluded {warm_up} warm_up, "
              f"averaging {len(durations) - warm_up} calls, "
              f"avg duration = {avg_duration:.3f} us")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse MindStudio profiler op_summary CSV and calculate kernel duration statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single kernel with 2 warm_up calls excluded
    python parse_op_summary.py /path/to/PROF_xxx 2 kernel_name
    
    # Multiple kernels
    python parse_op_summary.py /path/to/PROF_xxx 2 kernel_a kernel_b kernel_c
        """
    )
    
    parser.add_argument('prof_path', 
                        help='Path to the PROF_xxxxx folder containing mindstudio_profiler_output/')
    parser.add_argument('warm_up', type=int,
                        help='Number of initial calls to exclude as warm-up for each kernel')
    parser.add_argument('kernels', nargs='+',
                        help='Kernel name(s) to search for in Op Name column')
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.isdir(args.prof_path):
        print(f"Error: Path does not exist or is not a directory: {args.prof_path}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Find the op_summary CSV
    csv_path = find_op_summary_csv(args.prof_path)
    print(f"Parsing: {csv_path}")
    print(f"Kernels: {args.kernels}")
    print(f"Warm-up calls excluded per kernel: {args.warm_up}")
    print()
    
    # Calculate durations
    results = parse_kernel_durations(csv_path, args.kernels, args.warm_up)
    
    # Sum all averages
    total_duration = sum(results.values())
    
    # Print result
    print()
    print(f"Task Duration(us): {total_duration}")


if __name__ == "__main__":
    main()
