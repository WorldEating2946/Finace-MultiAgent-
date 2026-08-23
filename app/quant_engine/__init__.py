"""
app/quant_engine — 量化计算引擎

本模块负责所有"硬计算"逻辑：财务指标计算、统计分析、量化模型等。
所有计算均使用 Pandas / Numpy 完成，严禁 LLM 参与数值计算。

设计原则:
    - 确定性：相同输入 → 相同输出，不依赖概率模型
    - 可验证：每个方法都有独立的单元测试
    - 纯函数优先：避免副作用，计算结果仅由输入决定
"""

from app.quant_engine.calculator import FinancialCalculator

__all__ = ["FinancialCalculator"]
