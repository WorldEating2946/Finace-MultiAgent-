import { useState } from 'react'
import { runResearch, saveHistory } from '../api/agents'
import { items } from '../util'
import Explain from '../components/Explain'
import EmptyState from '../components/EmptyState'
import AgentRun from '../components/AgentRun'

type RR = {
  summary?: string
  business_model?: string
  competitive_advantages?: string[]
  key_risks_business?: string[]
  sources?: Array<{ source?: string; page?: unknown }>
}

export default function ResearchView() {
  const [company, setCompany] = useState('宁德时代')
  const [query, setQuery] = useState('')
  const [fast, setFast] = useState(true)
  const [loading, setLoading] = useState(false)
  const [steps, setSteps] = useState<{ node?: string; message?: string }[]>([])
  const [err, setErr] = useState('')
  const [res, setRes] = useState<RR | null>(null)

  async function run() {
    setErr('')
    setSteps([])
    setLoading(true)
    try {
      const r = await runResearch(company, query, fast, (step) => setSteps((prev) => [...prev, step]))
      setRes(r ?? {})
      void saveHistory(company, 'research', r ?? {})
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Explain title="Research Agent · 企业基本面">
        基于入库文档（年报/研报/政策）做自适应 RAG 检索，输出公司概况、竞争优势、经营风险与证据来源。
        —— 先到「知识库」上传该公司的文档，这里才能搜到真实内容。
      </Explain>

      <div className="card">
        <div className="row">
          <label>
            公司
            <input value={company} onChange={(e) => setCompany(e.target.value)} />
          </label>
          <label className="grow">
            研究问题（可选）
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="分析该公司的基本面与竞争力" />
          </label>
          <button onClick={run} disabled={loading || !company.trim()}>
            {loading ? '研究中…' : '运行 Research'}
          </button>
        </div>
        <div className="mode-toggle">
          <button className={'seg' + (fast ? ' active' : '')} onClick={() => setFast(true)}>⚡ 快速（单轮，秒级）</button>
          <button className={'seg' + (!fast ? ' active' : '')} onClick={() => setFast(false)}>🔬 深度（≤3 轮，更全面）</button>
        </div>
        {err && <div className="error">{err}</div>}
      </div>

      {loading && (
        <AgentRun message="Research 研究中（首次会加载模型）…" steps={steps.map((s) => s.message || s.node || '')} />
      )}

      {!res && !err && (
        <EmptyState
          icon="📚"
          title="运行 Research，查看企业基本面"
          body="填公司名（可选研究问题）点「运行 Research」。没有结果时，请先在知识库上传该公司的文档。"
        />
      )}

      {res && (
        <>
          <div className="card">
            <div className="muted small">摘要</div>
            <p style={{ margin: '8px 0' }}>{res.summary || '（空）'}</p>
            {res.business_model && (
              <>
                <div className="muted small" style={{ marginTop: 8 }}>商业模式 / 研究计划</div>
                <p style={{ margin: '6px 0' }}>{res.business_model}</p>
              </>
            )}
          </div>

          <div className="grid2">
            <div className="card">
              <div className="muted small">竞争优势</div>
              {items(res.competitive_advantages).length ? (
                <ul>{items(res.competitive_advantages).map((c, i) => <li key={i}>{c}</li>)}</ul>
              ) : <div className="empty">无</div>}
            </div>
            <div className="card">
              <div className="muted small">经营风险</div>
              {items(res.key_risks_business).length ? (
                <ul>{items(res.key_risks_business).map((c, i) => <li key={i}>{c}</li>)}</ul>
              ) : <div className="empty">无</div>}
            </div>
          </div>

          {res.sources && res.sources.length > 0 && (
            <div className="card">
              <div className="muted small">证据来源（{res.sources.length}）</div>
              <div style={{ marginTop: 8 }}>
                {res.sources.map((s, i) => (
                  <span key={i} className="tag">{s.source}{s.page ? ` · p${s.page}` : ''}</span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
