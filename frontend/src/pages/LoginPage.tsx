import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const { login } = useAuth()
  const nav = useNavigate()
  const [identifier, setIdentifier] = useState('admin')
  const [password, setPassword] = useState('admin123')
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
      <form className="auth-card" onSubmit={onSubmit}>
        <div className="auth-brand">FinaceAgent</div>
        <h1>投研工作台</h1>
        <p className="muted small">多 Agent 金融智能分析 — 登录</p>
        <label>
          用户名 / 邮箱
          <input
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            autoComplete="username"
            placeholder="admin"
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
        <button type="submit" disabled={loading}>
          {loading ? '登录中…' : '登录'}
        </button>
        <div className="muted small">admin/admin123 · analyst/analyst123 · demo/demo123</div>
      </form>
    </div>
  )
}
