import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import ThemeToggle from '../components/ThemeToggle'
import OverviewView from '../views/OverviewView'
import ResearchView from '../views/ResearchView'
import FinancialView from '../views/FinancialView'
import SentimentRiskView from '../views/SentimentRiskView'
import KnowledgeView from '../views/KnowledgeView'
import SystemStatusView from '../views/SystemStatusView'
import HistoryView from '../views/HistoryView'

type TabKey = 'overview' | 'research' | 'financial' | 'sentiment' | 'knowledge' | 'history' | 'system'

const TABS: Array<{ key: TabKey; label: string; ico: string }> = [
  { key: 'overview', label: '总览', ico: '◉' },
  { key: 'research', label: 'Research', ico: 'R' },
  { key: 'financial', label: 'Financial', ico: 'F' },
  { key: 'sentiment', label: '舆情+风险', ico: 'S' },
  { key: 'knowledge', label: '知识库', ico: 'K' },
  { key: 'history', label: '历史', ico: '🕘' },
  { key: 'system', label: '系统状态', ico: '⛁' },
]

const TITLES: Record<TabKey, string> = {
  overview: '多 Agent 全链路',
  research: 'Research · 企业基本面（RAG）',
  financial: 'Financial · 财务分析',
  sentiment: 'Sentiment + Risk · 舆情与风险',
  knowledge: '知识库 · 入库与检索',
  history: '历史数据 · 运行记录与复用',
  system: '系统就绪状态',
}

export default function Dashboard() {
  const { user, logout } = useAuth()
  const [tab, setTab] = useState<TabKey>('overview')

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">FINACEAGENT</div>
        {TABS.map((t) => (
          <div
            key={t.key}
            className={'navitem' + (tab === t.key ? ' active' : '')}
            onClick={() => setTab(t.key)}
          >
            <span className="ico">{t.ico}</span>
            {t.label}
          </div>
        ))}
        <div className="spacer" />
        <div className="userbox">
          <span className="muted small">
            {user?.username}
            {user?.is_admin ? ' · admin' : ''}
          </span>
          <ThemeToggle />
          <button className="ghost small" onClick={() => logout()}>
            退出
          </button>
        </div>
      </aside>

      <div className="main">
        <div className="pagehead">
          <div>
            <h2>{TITLES[tab]}</h2>
            <div className="sub">{tab} · FinaceAgent 多 Agent 投研</div>
          </div>
        </div>

        {/* 所有视图常驻挂载：切换 Tab 只隐藏，不卸载 → 各 Agent 结果/状态保留 */}
        <div className={'tabpane' + (tab === 'overview' ? '' : ' hidden')}><OverviewView /></div>
        <div className={'tabpane' + (tab === 'research' ? '' : ' hidden')}><ResearchView /></div>
        <div className={'tabpane' + (tab === 'financial' ? '' : ' hidden')}><FinancialView /></div>
        <div className={'tabpane' + (tab === 'sentiment' ? '' : ' hidden')}><SentimentRiskView /></div>
        <div className={'tabpane' + (tab === 'knowledge' ? '' : ' hidden')}><KnowledgeView /></div>
        <div className={'tabpane' + (tab === 'history' ? '' : ' hidden')}><HistoryView /></div>
        <div className={'tabpane' + (tab === 'system' ? '' : ' hidden')}><SystemStatusView /></div>
      </div>
    </div>
  )
}
