import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import type { User } from '../types'
import { getCurrentUser, getStoredRefreshToken, tryRefresh } from '../api/client'
import { login as loginApi, logout as logoutApi } from '../api/auth'

interface AuthCtx {
  user: User | null
  ready: boolean
  login: (identifier: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const Ctx = createContext<AuthCtx | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)

  // 启动时若本地有 refresh_token，则自动续期一次恢复登录态
  useEffect(() => {
    async function bootstrap() {
      if (getStoredRefreshToken()) {
        const ok = await tryRefresh()
        if (ok) setUser(getCurrentUser())
      }
      setReady(true)
    }
    void bootstrap()
  }, [])

  async function login(identifier: string, password: string) {
    const tokens = await loginApi(identifier, password) // 内部已 setSession + 持久化 refresh
    setUser(tokens.user)
  }
  async function logout() {
    try {
      await logoutApi() // 内部已 setSession(null) + 清 localStorage
    } finally {
      setUser(null)
    }
  }

  return <Ctx.Provider value={{ user, ready, login, logout }}>{children}</Ctx.Provider>
}

export function useAuth(): AuthCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAuth 必须在 AuthProvider 内使用')
  return ctx
}
