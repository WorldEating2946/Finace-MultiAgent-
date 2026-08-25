import { useEffect, useState } from 'react'
import { listHistory } from '../api/agents'
import { apiFetch } from '../api/client'
import Explain from '../components/Explain'
import EmptyState from '../components/EmptyState'
import ReportView from '../components/ReportView'
import { asStr, asNum } from '../util'

type Run = {
  id: number
  company: string
  ticker?: string
  agent_type: string
  created_at: string
  result: Record<string, unknown>
}

const AGENT_META: Record<string, string> = {
  research: 'Research 基本面',
  financial: 'Financial 财务',
  sentiment: 'Sentiment 舆情',
  risk: 'Risk 风险',
  report: '研报',
}

function reveal(run: Run): string {
  const r = run.result || {}
  const s = r.summary || r.risk_summary || r.commentary || ''
  if (s) return asStr(s).slice(0, 90)
  if (run.agent_type === 'financial') return `ROE=${asNum((r.key_metrics as any)?.roe_pct)?.toFixed(2) ?? '—'}%`
  return ''
}

export default function HistoryView() {
  const [company, setCompany] = useState('宁德时代')
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [report, setReport] = useState('')
  const [genLoading, setGenLoading] = useState(false)

  async function load() {
    setLoading(true)
    setErr('')
    try {
      const r = await listHistory('', '', 100)
      setRuns(r?.data?.runs ?? [])
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    void load()
  }, [])

  // 用最新历史生成综合报告（复用：不传四 Agent 输出，后端取最新历史）
  async function generateFromHistory() {
    setGenLoading(true)
    setErr('')
    try {
      const r = await apiFetch('/api/v1/report/generate', {
        method: 'POST',
        body: JSON.stringify({ company, ticker: '', user_query: '' }),
      })
      const j = await r.json()
      if (!r.ok) throw new Error(j?.detail ?? '生成失败')
      setReport(j?.data?.markdown ?? '')
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setGenLoading(false)
    }
  }

  const recent = runs.filter((r) => r.company === company)
  return (
    <div>
      <Explain title="历史数据 · 各 Agent 运行记录">
        每个 Agent 运行后都会保存结果。综合报告生成时**自动复用最新历史**，无需重跑。
        选公司 → 「用历史生成综合报告」即可用已保存的 Research/Financial/Sentiment/Risk 拼一份研报。
      </Explain>

      <div className="card">
        <div className="row">
          <label>
            公司
            <input value={company} onChange={(e) => setCompany(e.target.value)} />
          </label>
          <button onClick={generateFromHistory} disabled={genLoading}>
            {genLoading ? '生成中…' : '用历史生成综合报告'}
          </button>
          <button className="ghost" onClick={load} disabled={loading}>
            {loading ? '刷新…' : '刷新'}
          </button>
        </div>
        {err && <div className="error">{err}</div>}
        {report && <ReportView markdown={report} />}
      </div>

      <div className="card">
        <div className="card-title">「{company}」历史记录（{recent.length}）</div>
        {recent.length === 0 ? (
          <EmptyState icon="🗂" title="暂无该公司的历史" body="先去 Research/Financial/舆情+风险 跑一次，结果会自动保存到这里。" />
        ) : (
          recent.map((r) => (
            <div key={r.id} className="svc">
              <span className="sdot ok" aria-hidden="true" />
              <span className="svc-name">{AGENT_META[r.agent_type] ?? r.agent_type}</span>
              <span className="svc-detail">{reveal(r) || '（已完成）'}</span>
              <span className="svc-ms mono muted">{new Date(r.created_at).toLocaleTimeString()}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
