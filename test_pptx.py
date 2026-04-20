from pptx import Presentation
from pptx.util import Pt
import os

prs = Presentation()
prs.slide_width = 10000000
prs.slide_height = 5625000

slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(slide_layout)

title_box = slide.shapes.add_textbox(50000, 2000000, 12000000, 1000000)
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Test Title"
p.font.size = Pt(44000)

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop, "FA_Chunked_Prefill_LSE.pptx")
prs.save(output_path)
print(f"Saved to: {output_path}")
