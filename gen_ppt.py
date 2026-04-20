# -*- coding: utf-8 -*-
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

print("Starting PPT generation...")

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RgbColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_title_slide(title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RgbColor(0, 51, 102)
    p.alignment = PP_ALIGN.CENTER

    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11.333), Inches(1))
        tf = sub_box.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(24)
        p.alignment = PP_ALIGN.CENTER


def add_content_slide(title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RgbColor(0, 51, 102)

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.333), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = "• " + bullet
        p.font.size = Pt(22)
        p.space_after = Pt(12)


def add_diagram_slide(title, diagram_text):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RgbColor(0, 51, 102)

    diagram_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.333), Inches(5.5))
    tf = diagram_box.text_frame
    p = tf.paragraphs[0]
    p.text = diagram_text
    p.font.size = Pt(12)
    p.font.name = "Consolas"
    tf.word_wrap = True


# Slide 1: Title
add_title_slide("Flash Attention与Chunked Prefill:\n为什么需要LSE支持", "理解分块注意力与在线Softmax融合的技术原理")

# Slide 2: Background
add_content_slide(
    "背景：长序列Prefill的内存瓶颈",
    [
        "传统Prefill需要一次性加载全部KV到计算核",
        "内存复杂度：O(n²)，序列长度增长时急剧膨胀",
        "举例：32K序列需要约4GB KV缓存（FP16）",
        "Chunked Prefill：将长序列分块处理，每次只加载一个chunk",
        "内存复杂度降为O(chunk²)，可处理超长序列",
    ],
)

# Slide 3: How Chunked Prefill Works
add_diagram_slide(
    "Chunked Prefill工作方式",
    """[Chunked Prefill 流程]

输入序列: [tok1, tok2, tok3, ... tok1024]
           |----------|  (256 tokens per chunk)

Chunk1     Chunk2     Chunk3     Chunk4
  FA        FA         FA         FA
  |         |          |          |
Output1   Output2   Output3   Output4
(O1,LSE1) (O2,LSE2) (O3,LSE3) (O4,LSE4)

每次只加载一个chunk的KV，内存效率大幅提升""",
)

# Slide 4: The Core Problem
add_content_slide(
    "核心问题：如何合并分块输出？",
    [
        "错误做法：直接加权平均",
        "   O = w1*O1 + w2*O2 + w3*O3 + w4*O4",
        "   结果：数值完全错误！Softmax是非线性操作",
        "",
        "正确做法：利用LSE进行在线Softmax融合",
        "   每个Chunk FA返回：Output + LSE (Log-Sum-Exp)",
        "   通过数学公式将分块结果正确合并",
    ],
)

# Slide 5: LSE Math
add_diagram_slide(
    "LSE的数学原理",
    """[在线Softmax与LSE]

Flash Attention 使用在线Softmax算法：
  m_i = max(m_{i-1}, x_i)          // 运行最大值
  s_i = s_{i-1} + exp(x_i - m_i)  // 运行指数和
  f_i = exp(x_i - m_i) / s_i       // 归一化后的指数

关键输出：
  LSE = log(s_i) + m_i = log(Sigma exp(q*k_i))

LSE的价值：包含完整Softmax的全局信息

[LSE融合公式]

给定多个chunk的 (Output_i, LSE_i):
  LSE_max = max(LSE_1, LSE_2, ..., LSE_n)
  LSE_final = log( sum(exp(LSE_i - LSE_max)) ) + LSE_max

  Output_final = sum(exp(LSE_i - LSE_final) * Output_i)

这等价于一次性计算完整序列的Softmax结果！""",
)

# Slide 6: The Value of LSE
add_content_slide(
    "LSE支持的关键价值",
    [
        "内存效率 (分块处理) + 数值正确 (LSE融合)",
        "    |                      |",
        "    v                      v",
        "  可处理超长序列      正确计算:",
        "  (>32K tokens)      - attention after",
        "                     - logprobs采样",
        "                     - 后续decode的KV cache",
        "",
        "如果没有LSE支持:",
        "  要么：接受错误结果 (logprobs/采样错误)",
        "  要么：一次性加载全部KV (内存爆炸)",
    ],
)

# Slide 7: Architecture Diagram
add_diagram_slide(
    "Ascend NPU实现架构",
    """[Ascend NPU Chunked Prefill]

   Chunk1    Chunk2    Chunk3    Chunk4
     FA        FA        FA        FA
     |         |         |         |
   O1,LSE1   O2,LSE2   O3,LSE3   O4,LSE4
     |         |         |         |
     +---------+---------+---------+
                   |
           npu_attention_update
           (在线Softmax合并)
                   |
           Output_final + LSE_final

关键API:
- npu_fused_infer_attention_score(..., softmax_lse_flag=True)
- npu_attention_update(lse_list, out_list, update_type=0)""",
)

# Slide 8: Summary
add_content_slide(
    "总结",
    [
        "为什么FA在chunk prefill时需要LSE支持：",
        "",
        "1. 分块处理 = 内存效率",
        "   但每块独立计算，无法直接合并输出",
        "",
        "2. LSE = 在线Softmax的数学载体",
        "   包含完整的exp-sum信息，可用于正确融合",
        "",
        "3. LSE融合 = 数值正确性",
        "   将分块结果数学等价地合并成完整attention",
        "",
        "结论：LSE是chunked prefill从",
        "     省内存的妥协方案变成可用的正确方案的关键！",
    ],
)

# Save to current directory first
output_path = "D:/vllm-ascend/FA_Chunked_Prefill_LSE.pptx"
prs.save(output_path)
print(f"PPT已保存到: {output_path}")
print(f"File exists: {os.path.exists(output_path)}")
