import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", "from pptx import Presentation; print('OK')"],
    capture_output=True,
    text=True,
    cwd=r"D:\vllm-ascend",
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("RETURN:", result.returncode)
