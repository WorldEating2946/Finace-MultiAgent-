import { useState, type ReactNode } from 'react'
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

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: 'overview', label: '总览' },
  { key: 'research', label: 'Research' },
  { key: 'financial', label: 'Financial' },
  { key: 'sentiment', label: '舆情+风险' },
  { key: 'knowledge', label: '知识库' },
  { key: 'history', label: '历史' },
  { key: 'system', label: '系统状态' },
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

// 线性 SVG 图标（Feather 风格，跟随 currentColor）
function Ico({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  )
}

const ICONS: Record<TabKey, ReactNode> = {
  overview: <Ico d="M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z" />,
  research: <Ico d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />,
  financial: <Ico d="M3 21h18M7 16V9M12 16V5M17 16v-5" />,
  sentiment: <Ico d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
  knowledge: <Ico d="M12 3C7 3 3 4.3 3 6s4 3 9 3 9-1.3 9-3-4-3-9-3zM21 6v12c0 1.7-4 3-9 3s-9-1.3-9-3V6M21 12c0 1.7-4 3-9 3s-9-1.3-9-3" />,
  history: <Ico d="M12 7v5l3 2M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3 9M3 3v6h6" />,
  system: <Ico d="M22 12h-4l-3 9L9 3l-3 9H2" />,
}

function Mark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="22" height="22" rx="6" stroke="var(--accent)" strokeWidth="1.5" />
      <path
        d="M6.5 15l3-3.4 2.7 2.3 5-5.1"
        stroke="var(--accent)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14.2 8.8h4v4.4"
        stroke="var(--accent)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function Dashboard() {
  const { user, logout } = useAuth()
  const [tab, setTab] = useState<TabKey>('overview')
  const initial = (user?.username || '?').charAt(0).toUpperCase()
  const role = user?.is_admin ? '管理员' : '分析员'

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <Mark />
          <span>FINACEAGENT</span>
          <span className="mono small" style={{ marginLeft: 'auto', color: 'var(--text3)' }}>v1.0</span>
        </div>

        {TABS.map((t) => (
          <div
            key={t.key}
            className={'navitem' + (tab === t.key ? ' active' : '')}
            onClick={() => setTab(t.key)}
          >
            <span className="ico">{ICONS[t.key]}</span>
            {t.label}
          </div>
        ))}

        <div className="spacer" />

        <div className="userbox">
          <div className="ub-left">
            <span className="avatar">{initial}</span>
            <span className="ub-name">
              <span className="n">{user?.username}</span>
              <span className="r">{role}</span>
            </span>
          </div>
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
