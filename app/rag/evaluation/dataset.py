"""评测数据集模型（PR #32）。

将零散的手动评测升级为标准 Benchmark：
    - 每条 query 带 id / company / expected_sections（章节命中）/ expected_keywords（语义辅助）；
    - JSON 文件存放于 evaluation/datasets/，格式见数据集文件头注释。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """单条评测 query。"""

    id: str                         # "xiaomi_001"
    company: str                    # "xiaomi"（与向量库 company 过滤字段一致）
    query: str                      # 用户查询
    expected_sections: list[str] = Field(default_factory=list)   # 命中任一即 HIT
    expected_keywords: list[str] = Field(default_factory=list)   # 语义辅助（Phase 1 不参与计分）


class EvaluationDataset(BaseModel):
    """标准评测数据集：元数据 + items。"""

    name: str                       # "Xiaomi 2025 Annual Report"
    company: str                    # 目标公司（pipeline 过滤维度）
    items: list[DatasetItem]


def load_dataset(path: str) -> EvaluationDataset:
    """从 JSON 文件加载评测数据集。

    Args:
        path: evaluation/datasets/<name>.json 路径。

    Returns:
        EvaluationDataset。文件缺失 / 格式非法时抛异常（评测数据是硬依赖，不静默跳过）。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(raw)
