import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function RequireAuth({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth()
  if (!ready) return <div className="center muted">加载中…</div>
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}
