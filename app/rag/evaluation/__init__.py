"""RAG 评测体系（PR #32）：标准数据集 + 全量指标 + 回归基准。

    from app.rag.evaluation import (
        load_dataset, DatasetItem, EvaluationDataset,
        compute_metrics, RetrievalMetrics,
        RagEvaluator, PipelineBenchmark,
    )
"""

from app.rag.evaluation.dataset import DatasetItem, EvaluationDataset, load_dataset
from app.rag.evaluation.evaluator import PipelineBenchmark, RagEvaluator
from app.rag.evaluation.metrics import RetrievalMetrics, compute_metrics

__all__ = [
    "DatasetItem",
    "EvaluationDataset",
    "load_dataset",
    "RetrievalMetrics",
    "compute_metrics",
    "RagEvaluator",
    "PipelineBenchmark",
]
