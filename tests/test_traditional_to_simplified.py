"""繁体中文 → 简体中文 归一化 + 自动识别测试。"""

import pytest

from app.rag.parsers.text_normalizer import (
    _NORMALIZERS,
    needs_normalize,
    normalize_text,
)

pytest.importorskip("opencc")  # 未安装 OpenCC 时整模块跳过


def test_traditional_converts_to_simplified():
    traditional = "小米集團主要業務包括智能手機、物聯網與生活消費產品。"
    simplified = normalize_text(traditional)
    assert "集團" not in simplified
    assert "集团" in simplified
    assert "手机" in simplified
    assert "物联网" in simplified


def test_simplified_stays_unchanged():
    simplified = "小米集团主要业务包括智能手机、物联网与生活消费产品。"
    assert normalize_text(simplified) == simplified


def test_empty_text_returns_empty():
    assert normalize_text("") == ""


def test_auto_detect_whether_needed():
    """自动识别：含繁体才转换，简化 / 英文跳过。"""
    assert needs_normalize("小米集團主要業務") is True
    assert needs_normalize("小米集团主要业务") is False
    assert needs_normalize("Xiaomi Annual Report 2025") is False
    assert needs_normalize("") is False


def test_loader_normalizes_and_preserves_original(tmp_path):
    """loader 归一化文本用于 embedding，原文保留在 metadata.original_text。"""
    from app.rag.loaders import load_documents

    f = tmp_path / "繁体文档.txt"
    f.write_text("小米集團主要業務包括。", encoding="utf-8")

    doc = load_documents(str(f))[0]

    assert "集團" in doc.metadata.original_text  # 原文保留（可引用）
    assert "集团" in doc.text                     # 归一化后用于 embedding
    assert "業務" not in doc.text


def test_normalizer_registry_has_zh_and_reserved_slots():
    """归一化器注册表：含 zh_t2s，预留后续语言（日/韩）扩展位置。"""
    assert "zh_t2s" in _NORMALIZERS
