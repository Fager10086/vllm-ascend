# Test script
import os

os.chdir(r"D:\vllm-ascend")
from pptx import Presentation

print("Import OK")
prs = Presentation()
print("Create OK")
prs.save(r"C:\Users\admin\Desktop\test.pptx")
print("Save OK")
