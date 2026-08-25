"""文本归一化（多语言预留）。

当前：繁体中文 → 简体中文（**自动识别**是否需要转换，避免对已简化 / 英文文本做无谓转换）。
预留：日文 / 韩文 / 其他语言年报归一化，按需在 `_NORMALIZERS` 注册。

对外入口：
    from app.rag.parsers.text_normalizer import normalize_text, needs_normalize
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    """归一化器接口（预留多语言扩展：日 / 韩 / 其他）。"""

    name: str

    @abstractmethod
    def normalize(self, text: str) -> str:
        """对文本做归一化；不适用时返回原文本。"""
        raise NotImplementedError


class TraditionalToSimplified(TextNormalizer):
    """繁体中文 → 简体中文（OpenCC t2s）。"""

    name = "zh_t2s"

    def __init__(self) -> None:
        self._converter = None

    def _converter_or_none(self):
        if self._converter is None:
            try:
                from opencc import OpenCC
            except ImportError:
                return None
            self._converter = OpenCC("t2s")
        return self._converter

    def normalize(self, text: str) -> str:
        converter = self._converter_or_none()
        if converter is None:
            return text  # 未安装 OpenCC：降级不转换
        return converter.convert(text)


# 常见繁体「独有」字符（自动识别用；均为简体中不存在对应字形者）。
# 命中任一 → 需要繁→简；已简化 / 英文文本命中率≈0，可跳过转换。
_TRADITIONAL_CHARS = frozenset(
    "團業務機貨幣據訊網絡軟匯營發開產證財報潤總淨權益風險關聯審計監註併負債現買賣貴賤稅繳罰違規範標準優勢競爭戰轉級測試設計質項隊體資"
    "層討論會環業與體進選電動畫處週氣號傳區車場務際協儲備國萬億滿應園觀願確類顯鐘間雙離難響"
    "釋義識譯議訂詞語訓詳諸謝護讀變灣顧償驗認讓載連隨雖飾館飲餐馬鳥龜麗鮮歲懷態勢樂園圖導責執習練繫統條約價碼"
)

_zh_t2s = TraditionalToSimplified()

# 归一化器注册表：未来新增语言在此注册，并在 needs_normalize 中扩展识别
_NORMALIZERS: dict[str, TextNormalizer] = {
    "zh_t2s": _zh_t2s,
    # "ja": 日文用語归一化（预留）
    # "ko": 韩文归一化（预留）
}


def needs_normalize(text: str) -> bool:
    """自动识别文本是否需要繁→简转换。

    Args:
        text: 原始抽取文本。

    Returns:
        True 表示含繁体字符，需要转换。
    """
    if not text:
        return False
    return any(ch in _TRADITIONAL_CHARS for ch in text)


def normalize_text(text: str) -> str:
    """文本归一化入口：自动识别 → 应用对应归一化器。

    Args:
        text: 原始抽取文本。

    Returns:
        归一化后文本；不需要转换或依赖缺失时原样返回。
    """
    if not text:
        return text
    # 当前仅繁→简；未来按语言扩展（如 needs_ja / needs_ko）
    if needs_normalize(text):
        return _zh_t2s.normalize(text)
    return text
