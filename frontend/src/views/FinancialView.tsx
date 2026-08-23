import { useState } from 'react'
import { runFinancial, saveHistory } from '../api/agents'
import { items, asNum } from '../util'
import { FinancialTrend } from '../components/Charts'
import Explain from '../components/Explain'
import EmptyState from '../components/EmptyState'
import AgentRun from '../components/AgentRun'

type F = {
  key_metrics?: Record<string, unknown>
  dupont?: Record<string, unknown>
  yoy_history?: Array<Record<string, unknown>>
  commentary?: string
}

function Stat({ label, value, dir }: { label: string; value: string; dir?: 'pos' | 'neg' }) {
  return (
    <div className="stattile">
      <div className="lab">{label}</div>
      <div className={'val' + (dir ? ` ${dir}` : '')}>{value}</div>
    </div>
  )
}

export default function FinancialView() {
  const [ticker, setTicker] = useState('300750')
  const [company, setCompany] = useState('宁德时代')
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [res, setRes] = useState<F | null>(null)

  async function run() {
    setErr('')
    setLoading(true)
    try {
      const r = await runFinancial(ticker, company)
      setRes(r?.data ?? {})
      void saveHistory(company || ticker, 'financial', r?.data ?? {})
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const km = res?.key_metrics ?? {}

  return (
    <div>
      <Explain title="Financial Agent · 财务分析">
        从真实数据源（akshare）取年报，计算核心指标（ROE/利润率/同比）、杜邦拆解与历年的增速趋势。
        —— 填股票代码即可，无需入库文档。
      </Explain>

      <div className="card">
        <div className="row">
          <label>
            代码
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} />
          </label>
          <label>
            公司
            <input value={company} onChange={(e) => setCompany(e.target.value)} />
          </label>
          <button onClick={run} disabled={loading || !ticker.trim()}>
            {loading ? '计算中…' : '运行 Financial'}
          </button>
        </div>
        {err && <div className="error">{err}</div>}
      </div>

      {loading && <AgentRun message="Financial 分析中（获取数据 → 计算 → 点评）…" />}

      {!res && !err && (
        <EmptyState
          icon="💰"
          title="运行 Financial，查看财报指标"
          body="填股票代码（如 300750）点「运行 Financial」，会得到 ROE、净利润率、同比增速、杜邦拆解与趋势图。"
        />
      )}

      {res && (
        <>
          <div className="grid2">
            <Stat label="ROE" value={`${asNum(km.roe_pct)?.toFixed(2) ?? '—'}%`} dir={asNum(km.roe_pct)! >= 0 ? 'pos' : 'neg'} />
            <Stat label="净利润率" value={`${asNum(km.net_profit_margin_pct)?.toFixed(2) ?? '—'}%`} dir={asNum(km.net_profit_margin_pct)! >= 0 ? 'pos' : 'neg'} />
            <Stat label="营收同比" value={`${asNum(km.revenue_yoy_pct)?.toFixed(2) ?? '—'}%`} dir={asNum(km.revenue_yoy_pct)! >= 0 ? 'pos' : 'neg'} />
            <Stat label="净利同比" value={`${asNum(km.net_profit_yoy_pct)?.toFixed(2) ?? '—'}%`} dir={asNum(km.net_profit_yoy_pct)! >= 0 ? 'pos' : 'neg'} />
          </div>

          <div className="card">
            <div className="muted small">历年同比增速趋势</div>
            {items(res.yoy_history).length ? <FinancialTrend history={res.yoy_history ?? []} /> : <div className="empty">无同比历史</div>}
          </div>

          {res.commentary && (
            <div className="card">
              <div className="muted small">CFO 专业点评</div>
              <p style={{ margin: '8px 0' }}>{res.commentary}</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
