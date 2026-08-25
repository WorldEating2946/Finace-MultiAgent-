import { useState } from 'react'
import { streamAnalyze } from '../api/analyze'
import { saveHistory } from '../api/agents'
import type { StreamEvent } from '../types'
import StreamProgress from '../components/StreamProgress'
import ReportView from '../components/ReportView'
import Pipeline from '../components/Pipeline'
import EmptyState from '../components/EmptyState'

// 总览：多 Agent 全链路 —— Manager → ∥(Research,Financial,Sentiment) → Risk → Report
export default function OverviewView() {
  const [company, setCompany] = useState('宁德时代')
  const [ticker, setTicker] = useState('300750')
  const [query, setQuery] = useState('全面分析该公司的财务健康状况与发展前景')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [report, setReport] = useState('')
  const [reportPath, setReportPath] = useState('')
  const [running, setRunning] = useState(false)
  const [err, setErr] = useState('')

  async function run() {
    setEvents([])
    setReport('')
    setReportPath('')
    setErr('')
    setRunning(true)
    try {
      await streamAnalyze({ company, ticker, user_query: query }, (evt) => {
        setEvents((prev) => [...prev, evt])
        if (evt.type === 'done') {
          setReport(evt.markdown)
          setReportPath(evt.html_path)
          void saveHistory(company, 'report', { markdown: evt.markdown, report_id: evt.report_id })
        }
        if (evt.type === 'error') setErr(evt.message)
      })
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const idle = !running && events.length === 0 && !report

  return (
    <div>
      <Pipeline />

      <div className="card">
        <div className="row">
          <label>
            公司
            <input value={company} onChange={(e) => setCompany(e.target.value)} placeholder="如 宁德时代" />
          </label>
          <label>
            代码
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="如 300750" />
          </label>
          <label className="grow">
            提问（可选）
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="想分析什么？" />
          </label>
          <button onClick={run} disabled={running || !company.trim()}>
            {running ? '分析中…' : '开始分析'}
          </button>
        </div>

        {idle && (
          <EmptyState
            icon="🔍"
            title="一键生成投研报告"
            body="填好公司与代码，点「开始分析」——系统会并行调动 Research / Financial / Sentiment，再汇入 Risk 评估并输出结构化研报。下方实时显示各节点进度。"
          />
        )}
        {err && <div className="error">{err}</div>}
        {events.length > 0 && <StreamProgress events={events} />}
        {reportPath && <div className="muted small" style={{ marginTop: 8 }}>HTML：{reportPath}</div>}
      </div>

      {report && <ReportView markdown={report} />}
    </div>
  )
}
