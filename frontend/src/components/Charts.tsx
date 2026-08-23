import type { CSSProperties } from 'react'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  BarChart,
  Bar,
} from 'recharts'

// 图表配色全部用 CSS 变量 → 随深/浅主题自动切换
const tick = { fill: 'var(--text2)', fontSize: 11 }
const tooltipStyle: CSSProperties = {
  backgroundColor: 'var(--bg2)',
  border: '1px solid var(--border)',
  borderRadius: 8,
  color: 'var(--text1)',
  fontSize: 12,
}
const legendStyle: CSSProperties = { fontSize: 12, color: 'var(--text2)' }

const SENTIMENT_COLORS = ['var(--chart-pos)', 'var(--chart-neg)', 'var(--chart-neu)'] // pos/neg/neutral

// ── 舆情分布（甜甜圈；pos/neg/neutral，标签次级编码）──────────
export function SentimentDonut({ distribution }: { distribution: Record<string, number> }) {
  const data = [
    { name: '看多', key: 'positive', value: distribution?.positive ?? 0 },
    { name: '看空', key: 'negative', value: distribution?.negative ?? 0 },
    { name: '中性', key: 'neutral', value: distribution?.neutral ?? 0 },
  ].filter((d) => d.value > 0)
  if (data.length === 0) return <div className="empty">暂无舆情评分</div>
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={52} outerRadius={82} paddingAngle={2}>
            {data.map((d) => {
              const idx = ['positive', 'negative', 'neutral'].indexOf(d.key)
              return <Cell key={d.key} fill={SENTIMENT_COLORS[idx]} stroke="var(--bg1)" strokeWidth={2} />
            })}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={legendStyle} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── 财务趋势（同比增速折线；revenue vs net_profit，两条系列带图例）──
export function FinancialTrend({ history }: { history: Array<Record<string, unknown>> }) {
  if (!history || history.length === 0) return <div className="empty">暂无同比历史</div>
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history} margin={{ left: 0, right: 12, top: 8, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="period" tick={tick} stroke="var(--border)" />
          <YAxis tick={tick} stroke="var(--border)" width={46} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={legendStyle} />
          <Line type="monotone" dataKey="revenue_growth_pct" name="营收同比%" stroke="var(--chart-rev)" strokeWidth={2} dot={{ r: 3 }} />
          <Line type="monotone" dataKey="net_profit_growth_pct" name="净利同比%" stroke="var(--chart-profit)" strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── 风险维度（单一量纲得分柱状；标题命名实体，无需图例）──
export function RiskBars({ dimensions }: { dimensions: Array<Record<string, unknown>> }) {
  if (!dimensions || dimensions.length === 0) return <div className="empty">暂无风险维度</div>
  const data = dimensions.map((d) => ({ name: String(d.name ?? ''), score: Number(d.score ?? 0) }))
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: 0, right: 12, top: 8, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={tick} stroke="var(--border)" />
          <YAxis tick={tick} stroke="var(--border)" width={42} domain={[0, 1]} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
          <Bar dataKey="score" name="得分" fill="var(--accent)" radius={[4, 4, 0, 0]} maxBarSize={48} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
