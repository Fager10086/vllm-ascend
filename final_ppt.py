from pptx import Presentation
from pptx.util import Inches, Pt
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

s1 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s1.shapes.add_textbox(Inches(1), Inches(3), Inches(11), Inches(1))
p = tb.text_frame.paragraphs[0]
p.text = "Flash Attention与Chunked Prefill:\n为什么需要LSE支持"
p.font.size = Pt(36)
p.alignment = 1

s2 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s2.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(6))
p = tb.text_frame.paragraphs[0]
p.text = "背景：长序列Prefill的内存瓶颈\n\n• 传统Prefill需要一次性加载全部KV\n• 内存复杂度：O(n²)\n• Chunked Prefill：分块处理\n• 内存降为O(chunk²)"
p.font.size = Pt(20)

s3 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s3.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(6))
p = tb.text_frame.paragraphs[0]
p.text = "核心问题：如何合并分块输出？\n\n❌ 错误：O = w1*O1 + w2*O2\n→ 数值完全错误\n\n✅ 正确：利用LSE融合\n→ 每个Chunk FA返回Output + LSE\n→ 通过数学公式正确合并"
p.font.size = Pt(20)

s4 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s4.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(6))
p = tb.text_frame.paragraphs[0]
p.text = "LSE融合公式\n\nLSE = log(Σ exp(q·kᵢ))\n\nLSE_max = max(LSE₁, LSE₂, ...)\nLSE_final = log(Σ exp(LSE_i - LSE_max)) + LSE_max\n\nOutput_final = Σ exp(LSE_i - LSE_final) × Output_i"
p.font.size = Pt(20)

s5 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s5.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(6))
p = tb.text_frame.paragraphs[0]
p.text = "LSE支持的关键价值\n\n内存效率 + 数值正确\n• 可处理超长序列 (>32K)\n• 正确计算 logprobs\n• 正确计算 attention after\n\n没有LSE：\n• 接受错误结果 OR 内存爆炸"
p.font.size = Pt(20)

s6 = prs.slides.add_slide(prs.slide_layouts[6])
tb = s6.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(6))
p = tb.text_frame.paragraphs[0]
p.text = "总结\n\n1. 分块处理 = 内存效率\n   但无法直接合并输出\n\n2. LSE = 在线Softmax的数学载体\n   包含完整的exp-sum信息\n\n3. LSE融合 = 数值正确性\n   数学等价地合并分块结果\n\n结论：LSE是chunked prefill从妥协方案变成可用方案的关键！"
p.font.size = Pt(20)

# Save to current directory first
out = "FA_Chunked_Prefill_LSE.pptx"
prs.save(out)

if os.path.exists(out):
    print(f"OK: {os.path.getsize(out)} bytes")
    # Copy to desktop
    import shutil

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shutil.copy(out, desktop)
    print(f"Copied to Desktop")
else:
    print("FAIL: File not created")
