"""PDF 文本质量评分单元测试。"""

from app.rag.loaders.pdf.quality_checker import QUALITY_THRESHOLD, quality_score


def test_clean_chinese_scores_high():
    text = "小米集团主要业务包括智能手机、IoT与生活消费产品。"
    assert quality_score(text) >= QUALITY_THRESHOLD


def test_clean_english_scores_high():
    text = "Xiaomi Annual Report 2025 - Consolidated Financial Statements"
    assert quality_score(text) >= QUALITY_THRESHOLD


def test_numbers_only_scores_high():
    # 财务数字（ASCII 数字/标点）是良好字符
    text = "22.3% 76.8% 1,079.2 411,082"
    assert quality_score(text) >= QUALITY_THRESHOLD


def test_garbled_unicode_scores_low():
    # 乱码：CJK 部首/数学符号/拉丁扩展（模拟 ToUnicode 损坏）
    text = "Ṗ⸶屣⊛㥄奃 㛮⸶⸳⟳␌灭劳㕉⎌ᷯ"
    assert quality_score(text) < QUALITY_THRESHOLD


def test_cid_placeholder_scores_low():
    # pdfminer 无法映射字形时的 (cid:NNN) 占位（虽是 ASCII，但代表文本层损坏）
    text = "(cid:18096) (cid:25286) 4 (cid:8494)(cid:9146)"
    assert quality_score(text) < QUALITY_THRESHOLD


def test_empty_text_scores_zero():
    assert quality_score("") == 0.0
