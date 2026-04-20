import sys

print("STEP1: Starting", file=sys.stderr)
from pptx import Presentation

print("STEP2: Import OK", file=sys.stderr)
from pptx.util import Inches, Pt

print("STEP3: Utils OK", file=sys.stderr)
prs = Presentation()
print("STEP4: Created", file=sys.stderr)
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
print("STEP5: Dimensions set", file=sys.stderr)
output = "D:/vllm-ascend/test.pptx"
print(f"STEP6: Saving to {output}", file=sys.stderr)
prs.save(output)
print("STEP7: Saved", file=sys.stderr)
