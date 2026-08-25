import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

function Mark() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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

// 市场数据带：静态高度序列（禁用随机，保证确定性 & 干净）
const BAND = [38, 56, 34, 62, 44, 70, 40, 58, 36, 66, 48, 60]

export default function LoginPage() {
  const { login } = useAuth()
  const nav = useNavigate()
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setErr('')
    setLoading(true)
    try {
      await login(identifier, password)
      nav('/', { replace: true })
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-wrap">
      <aside className="auth-hero">
        <div className="hero-brand">
          <Mark />
          <span>FINACEAGENT</span>
        </div>

        <h1>
          把一支投研团队
          <br />
          <em>装进一个终端</em>
        </h1>
        <p className="hero-sub">
          LLM · LangGraph · RAG · 真实金融数据 · 中文情感——输入目标公司，自动产出六章结构化研报，全程可审计、可降级、可复用。
        </p>

        <div className="hero-band" aria-hidden="true">
          {BAND.map((h, i) => (
            <span key={i} className="hb" style={{ height: `${h}%` }} />
          ))}
        </div>

        <ul className="hero-feats">
          <li>
            <i className="dot live" />
            A股 · 沪深 · 实时行情
          </li>
          <li>
            <i className="dot" />
            RAG 研报 · 证据索引归因
          </li>
          <li>
            <i className="dot" />
            多 Agent 并行 · 全链路可观测
          </li>
        </ul>
      </aside>

      <main className="auth-panel">
        <form className="auth-card" onSubmit={onSubmit}>
          <div className="auth-card-head">
            <h1>登录</h1>
            <p className="muted small">投研工作台 · FinaceAgent</p>
          </div>

          <label>
            用户名 / 邮箱
            <input
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              autoComplete="username"
              placeholder="请输入用户名"
            />
          </label>

          <label>
            密码
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              placeholder="••••••"
            />
          </label>

          {err && <div className="error">{err}</div>}

          <button type="submit" disabled={loading} className="block">
            {loading ? '登录中…' : '登录'}
          </button>

          <div className="auth-foot muted small">没有账号？请联系管理员开通</div>
        </form>
      </main>
    </div>
  )
}
