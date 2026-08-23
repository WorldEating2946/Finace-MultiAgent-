"""Runtime 指标注册表（PR42b）—— 内置轻量 Counter/Histogram，零第三方依赖。

设计目标：可观测、可审计，未来接 Prometheus/Grafana 零改动。
    - Counter：单调递增计数（research_*_total / worker_*_total）。
    - Histogram：观察值分布（固定 buckets），可算 p50/p95/均值。
    - export_prometheus()：输出 Prometheus 文本格式（# HELP/# TYPE/count/sum/bucket），
      未来挂 GET /metrics 端点即暴露。
    - snapshot()：dict 视图，供测试与内部消费。

标签策略：计数器不带 research_id（避免高基数反模式）；逐任务明细走结构化日志
（observability.log_worker_event），指标回答聚合问题、日志回答"哪个 Research"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Prometheus 风格直方图固定桶（秒）。
# research_runtime_seconds：单次研究从认领到终态，10min(max_run) 内分桶。
RUNTIME_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1200, 3600)
# recovery_duration_seconds：从心跳过期（stale）到被接管续跑的恢复耗时。
RECOVERY_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600)
# task_wait_seconds：任务从创建到被 Worker CAS 认领的排队等待耗时（PR43.5 ②）。
WAIT_BUCKETS = (0.1, 0.5, 1, 5, 15, 30, 60, 120, 300, 600)


@dataclass
class Counter:
    """单调递增计数器。"""

    name: str
    help: str
    value: int = 0

    def inc(self) -> None:
        self.value += 1

    def export(self) -> list[str]:
        return [f"{self.name} {self.value}"]


@dataclass
class Histogram:
    """直方图：记录观察值，支持累计桶导出。"""

    name: str
    help: str
    buckets: tuple[float, ...] = field(default_factory=tuple)
    _values: list[float] = field(default_factory=list, repr=False)

    def observe(self, value: float) -> None:
        self._values.append(float(value))

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values)

    def _cumulative(self) -> dict[float, int]:
        """每个桶上限 → 累计观察数（Prometheus 语义：observations ≤ bound，逐桶独立计算）。"""
        return {
            bound: sum(1 for v in self._values if v <= bound) for bound in self.buckets
        }

    def export(self) -> list[str]:
        lines = []
        for bound, acc in self._cumulative().items():
            lines.append(f'{self.name}_bucket{{le="{bound:g}"}} {acc}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{self.name}_sum {self.sum:g}")
        lines.append(f"{self.name}_count {self.count}")
        return lines

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "sum": round(self.sum, 3),
            "buckets": {f"le={k:g}": v for k, v in self._cumulative().items()},
        }


@dataclass
class Gauge:
    """可设值指标（当前值语义）：并发数 / 队列长度等（PR43.5 ②）。

    与 Counter 不同：Gauge 可上升可下降，回答"现在是多少"而不是"累计多少次"。
    """

    name: str
    help: str
    value: float = 0.0

    def set(self, value: float) -> None:
        self.value = float(value)

    def export(self) -> list[str]:
        return [f"{self.name} {self.value:g}"]


class Metrics:
    """Runtime 指标注册表（PR42b 定义的 9 计数 + 2 直方图）。"""

    def __init__(self) -> None:
        self.research_started_total = Counter(
            "research_started_total", "研究任务提交总数"
        )
        self.research_completed_total = Counter(
            "research_completed_total", "研究任务成功完成总数（含崩溃恢复后完成）"
        )
        self.research_failed_total = Counter(
            "research_failed_total", "研究任务失败总数（执行异常 / 超时 / 超限）"
        )
        self.research_cancelled_total = Counter(
            "research_cancelled_total", "用户取消的研究任务总数"
        )

        self.worker_claim_total = Counter(
            "worker_claim_total", "CAS 认领成功总数（新任务 + 接管 + resume）"
        )
        self.worker_reclaim_total = Counter(
            "worker_reclaim_total", "接管 stale 孤儿任务总数（崩溃恢复）"
        )
        self.worker_fenced_total = Counter(
            "worker_fenced_total", "租约被接管后本 worker 写被拒总数（旧 worker 复活）"
        )
        self.worker_heartbeat_failure_total = Counter(
            "worker_heartbeat_failure_total", "心跳写入失败总数（异常，非 fenced）"
        )
        self.worker_timeout_total = Counter(
            "worker_timeout_total", "watchdog 运行超时判 failed 总数"
        )

        self.research_runtime_seconds = Histogram(
            "research_runtime_seconds", "单次研究从认领到终态的耗时", RUNTIME_BUCKETS
        )
        self.recovery_duration_seconds = Histogram(
            "recovery_duration_seconds",
            "崩溃恢复到被接管的耗时（心跳过期→认领）",
            RECOVERY_BUCKETS,
        )

        # ── PR43.5 ② Worker 池可观测性 ───────────────────────────
        self.worker_active_count = Gauge(
            "worker_active_count", "当前正在执行的研究任务数（并发上限 = worker_count）"
        )
        self.queue_length = Gauge(
            "queue_length", "Redis 任务队列当前长度（reaper 周期采样）"
        )
        self.task_wait_seconds = Histogram(
            "task_wait_seconds",
            "任务从创建到被 Worker CAS 认领的排队等待耗时",
            WAIT_BUCKETS,
        )
        self.worker_execution_seconds = Histogram(
            "worker_execution_seconds",
            "Worker 执行单次研究任务总耗时（含心跳/终态）",
            RUNTIME_BUCKETS,
        )

    # ── 计数 ────────────────────────────────────────────────────
    def inc(self, name: str) -> None:
        """按名称递增计数器（测试/通用入口）。"""
        counter = getattr(self, name, None)
        if isinstance(counter, Counter):
            counter.inc()
        else:
            raise KeyError(f"unknown counter: {name}")

    def observe(self, name: str, value: float) -> None:
        """按名称记录直方图观察值。"""
        hist = getattr(self, name, None)
        if isinstance(hist, Histogram):
            hist.observe(value)
        else:
            raise KeyError(f"unknown histogram: {name}")

    def set(self, name: str, value: float) -> None:
        """按名称设置 Gauge 当前值（并发数 / 队列长度）。"""
        gauge = getattr(self, name, None)
        if isinstance(gauge, Gauge):
            gauge.set(value)
        else:
            raise KeyError(f"unknown gauge: {name}")

    # ── 导出 ────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """全部指标当前值（测试与 /metrics JSON 消费）。"""
        out: dict = {}
        for c in self._counters():
            out[c.name] = c.value
        for h in self._histograms():
            out[h.name] = h.snapshot()
        for g in self._gauges():
            out[g.name] = g.value
        return out

    def export_prometheus(self) -> str:
        """Prometheus 文本格式（text/plain; version=0.0.4），未来 GET /metrics 直接暴露。"""
        lines: list[str] = []
        for c in self._counters():
            lines.append(f"# HELP {c.name} {c.help}")
            lines.append(f"# TYPE {c.name} counter")
            lines += c.export()
        for h in self._histograms():
            lines.append(f"# HELP {h.name} {h.help}")
            lines.append(f"# TYPE {h.name} histogram")
            lines += h.export()
        for g in self._gauges():
            lines.append(f"# HELP {g.name} {g.help}")
            lines.append(f"# TYPE {g.name} gauge")
            lines += g.export()
        return "\n".join(lines) + "\n"

    # ── 内部 ────────────────────────────────────────────────────
    def _counters(self) -> list[Counter]:
        return [v for v in vars(self).values() if isinstance(v, Counter)]

    def _histograms(self) -> list[Histogram]:
        return [v for v in vars(self).values() if isinstance(v, Histogram)]

    def _gauges(self) -> list[Gauge]:
        return [v for v in vars(self).values() if isinstance(v, Gauge)]
