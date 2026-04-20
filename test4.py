import os
from pptx import Presentation

print("Starting...")

try:
    prs = Presentation()
    print("Presentation created")

    # Use absolute path
    output_path = "D:/vllm-ascend/test.pptx"
    prs.save(output_path)
    print(f"Saved to: {output_path}")

    if os.path.exists(output_path):
        print(f"File exists! Size: {os.path.getsize(output_path)}")
    else:
        print("File NOT found")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

print("Done")
