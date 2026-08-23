// 5-Agent 分析管线图（签名元素）：Manager → ∥(Research,Financial,Sentiment) → Risk → Report
export default function Pipeline() {
  return (
    <div className="pipeline" aria-label="多 Agent 分析流程">
      <div className="pipe-stage prime">
        <span className="ps-name">Manager</span>
        <span className="ps-desc">理解意图并规划</span>
      </div>
      <span className="pipe-arrow">→</span>
      <div className="pipe-parallel">
        <div className="pipe-stage">
          <span className="ps-name">Research</span>
          <span className="ps-desc">基本面 · RAG</span>
        </div>
        <div className="pipe-stage">
          <span className="ps-name">Financial</span>
          <span className="ps-desc">财务 · 杜邦</span>
        </div>
        <div className="pipe-stage">
          <span className="ps-name">Sentiment</span>
          <span className="ps-desc">舆情 · 情绪</span>
        </div>
      </div>
      <span className="pipe-arrow">→</span>
      <div className="pipe-stage">
        <span className="ps-name">Risk</span>
        <span className="ps-desc">三维度综合风险</span>
      </div>
      <span className="pipe-arrow">→</span>
      <div className="pipe-stage prime">
        <span className="ps-name">Report</span>
        <span className="ps-desc">结构化研报</span>
      </div>
    </div>
  )
}
