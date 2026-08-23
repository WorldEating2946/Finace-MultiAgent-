import type { StreamEvent } from '../types'

const NODE_FRIENDLY: Record<string, string> = {
  manager: 'Manager 规划',
  research: 'Research 基本面',
  financial: 'Financial 财务',
  sentiment: 'Sentiment 舆情',
  risk: 'Risk 风险',
  report: 'Report 研报',
  health_check: '健康检查',
  evaluate_report: '质量评估',
  retry: '重试',
  rework: '修订',
  fan_out: '并行分发',
}

// SSE 节点流水（签名元素：研究"上屏"滚动带）
export default function StreamProgress({ events }: { events: StreamEvent[] }) {
  const nodes = events
    .filter((e): e is Extract<StreamEvent, { type: 'node_end' }> => e.type === 'node_end')
    .map((e) => e.node)
  const done = Array.from(new Set(nodes))

  return (
    <div>
      <div className="muted small" style={{ marginBottom: 8 }}>
        已推送 {events.length} 事件 · 完成节点 {done.length} 个
      </div>
      <div className="tape" aria-live="polite">
        {done.length === 0 ? (
          <span className="muted small">等待流水…</span>
        ) : (
          done.map((node) => (
            <span key={node} className="node done">
              {NODE_FRIENDLY[node] ?? node}
            </span>
          ))
        )}
      </div>
    </div>
  )
}
