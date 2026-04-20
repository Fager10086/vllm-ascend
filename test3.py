# Test script
import os

os.chdir(r"D:\vllm-ascend")
from pptx import Presentation

print("Import OK")
prs = Presentation()
print("Create OK")
output_path = r"D:\vllm-ascend\test.pptx"
prs.save(output_path)
print(f"Saved to: {output_path}")

if os.path.exists(output_path):
    print("File exists!")
    print(f"Size: {os.path.getsize(output_path)} bytes")
else:
    print("File not found after save!")
