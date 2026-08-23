import { useState } from 'react'
import { runSentiment, runRisk, saveHistory } from '../api/agents'
import { asNum, asStr, items } from '../util'
import { SentimentDonut, RiskBars } from '../components/Charts'
import Explain from '../components/Explain'
import EmptyState from '../components/EmptyState'
import AgentRun from '../components/AgentRun'

type Sent = {
  searched_news_count?: number
  sentiment_distribution?: Record<string, number>
  summary?: string
  topics?: Array<Record<string, unknown>>
}
type Risk = {
  overall_risk_level?: string
  overall_score?: number
  risk_summary?: string
  dimensions?: Array<Record<string, unknown>>
  key_risks?: string[]
}

export default function SentimentRiskView() {
  const [symbol, setSymbol] = useState('300750')
  const [company, setCompany] = useState('宁德时代')
  const [days, setDays] = useState('30')
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [sent, setSent] = useState<Sent | null>(null)
  const [risk, setRisk] = useState<Risk | null>(null)

  async function run() {
    setErr('')
    setLoading(true)
    try {
      const daysN = Number(days) || 30
      const s = await runSentiment(symbol, company, daysN)
      const r = await runRisk(symbol, company, daysN)
      setSent(s?.data ?? {})
      setRisk(r?.data ?? {})
      void saveHistory(company || symbol, 'sentiment', s?.data ?? {})
      void saveHistory(company || symbol, 'risk', r?.data ?? {})
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const dist = sent?.sentiment_distribution ?? {}
  const score = asNum(risk?.overall_score)

  return (
    <div>
      <Explain title="Sentiment + Risk · 舆情与风险">
        抓取真实新闻（akshare 东方财富）→ FinBERT 情感评分 → 主题聚类；再把舆情、财务、行业信号融合成综合风险等级与维度评分。
      </Explain>

      <div className="card">
        <div className="row">
          <label>
            代码
            <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </label>
          <label>
            公司
            <input value={company} onChange={(e) => setCompany(e.target.value)} />
          </label>
          <label>
            回溯天数
            <input value={days} onChange={(e) => setDays(e.target.value)} inputMode="numeric" />
          </label>
          <button onClick={run} disabled={loading || !symbol.trim()}>
            {loading ? '分析中…' : '运行舆情 + 风险'}
          </button>
        </div>
        {err && <div className="error">{err}</div>}
      </div>

      {loading && <AgentRun message="舆情 + 风险分析中（抓新闻 → FinBERT 评分 → 风险评估）…" />}

      {!sent && !risk && !err && (
        <EmptyState
          icon="📰"
          title="运行舆情 + 风险"
          body="填代码与回溯天数，查看新闻情感分布、热点主题与综合风险等级/维度评分。"
        />
      )}

      {(sent || risk) && (
        <div className="grid2">
          <div className="card">
            <div className="muted small">新闻数 {sent?.searched_news_count ?? 0} · 舆情分布</div>
            <SentimentDonut distribution={dist} />
            {sent?.summary && <p className="small muted" style={{ marginBottom: 0 }}>{sent.summary}</p>}
          </div>
          <div className="card">
            <div className="muted small">
              风险等级 <span className="mono">{asStr(risk?.overall_risk_level)}</span> · 综合得分 <span className="mono">{score?.toFixed(2) ?? '—'}</span>
            </div>
            <RiskBars dimensions={items<Record<string, unknown>>(risk?.dimensions)} />
            {risk?.risk_summary && <p className="small muted" style={{ marginBottom: 0 }}>{risk.risk_summary}</p>}
          </div>
        </div>
      )}

      {risk && items(risk.key_risks).length > 0 && (
        <div className="card">
          <div className="muted small">关键风险</div>
          <ul>{items(risk.key_risks).map((r, i) => <li key={i}>{asStr(r)}</li>)}</ul>
        </div>
      )}
    </div>
  )
}
