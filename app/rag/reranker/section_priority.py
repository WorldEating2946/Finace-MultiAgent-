"""年报章节优先级（PR #33）。

企业年报有天然层级：分析师查询（未来战略/营业收入/风险）最常落在
"管理层讨论及分析"等分析性章节，而非封面/目录/公司简介。

CrossEncoder 语义分数相近时，高价值章节的 chunk 应排序更前。
该信号作为 Hybrid Score Fusion 的 β 项（beta * section_priority）。

优先级取值语义：加到 final_score 的相对权重（0~1）。当前为静态表，
后续可接入公司/行业画像做动态调整（PR #34 企业知识画像）。
"""

from __future__ import annotations

# 静态默认优先级表：键 = chunk.chapter（或 section 前缀），值 = 相对权重。
_DEFAULT_PRIORITY: dict[str, float] = {
    # 分析性章节（分析师问题最常命中）——最高优先级
    "管理层讨论及分析": 0.15,
    "管理层讨论与分析": 0.15,
    "未来展望": 0.15,
    # 公司治理 / 报告类
    "董事会报告": 0.10,
    "企业管治报告": 0.10,
    "公司治理": 0.10,
    "公司治理、环境和社会": 0.10,
    # 财务数据类
    "五年财务概要": 0.10,
    "财务摘要": 0.10,
    "财务报告": 0.10,
    "财务报表": 0.10,
    "风险因素": 0.10,
}


def get_section_priority() -> dict[str, float]:
    """返回章节优先级表（拷贝，防外部修改污染默认值）。"""
    return dict(_DEFAULT_PRIORITY)
