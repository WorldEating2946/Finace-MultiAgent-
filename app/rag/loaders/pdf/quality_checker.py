"""PDF 文本抽取质量评分。

用于检测文本层损坏（如 ToUnicode 映射错误 → 乱码字符）。

评分 = 良好字符占比：
    良好 = CJK 统一表意文字 + ASCII 可打印 + CJK/全角标点；
乱码字符（CJK 部首区、数学符号、拉丁扩展、扩展 A 等）不计分。

典型值（实测）：
    干净中文年报（宁德时代）→ >0.8
    程序生成合成 PDF        → >0.9
    文本层损坏（小米 2025）→ <0.5
"""

from __future__ import annotations

import re

# 质量阈值：低于则触发 fallback / 阻断（与 loaders 约定一致）
QUALITY_THRESHOLD = 0.8

# 良好字符区间（闭合）：
#   ASCII 可打印 0x20-0x7E / CJK 标点 0x3000-0x303F
#   CJK 统一表意 0x4E00-0x9FFF / 全角形式 0xFF00-0xFFEF
_GOOD_RANGES = (
    (0x20, 0x7E),
    (0x3000, 0x303F),
    (0x4E00, 0x9FFF),
    (0xFF00, 0xFFEF),
)

# pdfminer/pdfplumber 无法映射字形时的占位记号（CID → Unicode 失败）
# 如 "(cid:18096)"。虽是 ASCII，但代表文本层损坏 → 判为坏字符。
_CID_PLACEHOLDER = re.compile(r"\(cid:\d+\)")


def quality_score(text: str) -> float:
    """计算文本抽取质量评分（0.0 ~ 1.0）。

    Args:
        text: 抽取出的文本。

    Returns:
        良好字符占比；`(cid:NNN)` 占位记为坏字符；空白不计入分母；空文本返回 0.0。
    """
    # 标记 (cid:NNN) 占位字符为坏
    cid_bad = set()
    for m in _CID_PLACEHOLDER.finditer(text):
        cid_bad.update(range(m.start(), m.end()))

    total = 0
    good = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        total += 1
        if i in cid_bad:
            continue
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _GOOD_RANGES):
            good += 1
    if total == 0:
        return 0.0
    return good / total
