// 轻量 API 客户端：持有 token，401 时用 refresh_token 自动续期并重试。
import type { Tokens } from '../types'

// ── 会话存储（模块级单例）──────────────────────────────
let accessToken: string | null = null
let refreshToken: string | null = null
let currentUser: Tokens['user'] | null = null

const REFRESH_STORAGE_KEY = 'finaceagent_refresh_token'

export function setSession(tokens: Tokens | null): void {
  if (tokens) {
    accessToken = tokens.access_token
    refreshToken = tokens.refresh_token
    currentUser = tokens.user
    localStorage.setItem(REFRESH_STORAGE_KEY, tokens.refresh_token)
  } else {
    accessToken = null
    refreshToken = null
    currentUser = null
    localStorage.removeItem(REFRESH_STORAGE_KEY)
  }
}

export function getAccessToken(): string | null {
  return accessToken
}
export function getCurrentUser(): Tokens['user'] | null {
  return currentUser
}
export function getStoredRefreshToken(): string | null {
  return refreshToken ?? localStorage.getItem(REFRESH_STORAGE_KEY)
}

// ── 刷新 token（唯一能续期 access 的入口）───────────────
export async function tryRefresh(): Promise<boolean> {
  const rt = getStoredRefreshToken()
  if (!rt) return false
  const res = await fetch('/api/v1/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: rt }),
  })
  if (!res.ok) return false
  const tokens = (await res.json()) as Tokens
  setSession(tokens)
  return true
}

// ── 从响应安全提取错误信息（兼容非 JSON 500，如代理/未启动后端）──
export async function errorMessage(res: Response): Promise<string> {
  let msg = `HTTP ${res.status}`
  try {
    const j = await res.json()
    msg = j?.detail ?? j?.message ?? msg
  } catch {
    try {
      const t = (await res.text()).slice(0, 200)
      if (t) msg = t
    } catch {
      /* 保持 HTTP 状态 */
    }
  }
  return msg
}

// ── 自动带 Bearer 的 fetch ──────────────────────────────
export async function apiFetch(
  path: string,
  options: RequestInit = {},
  retry = true,
): Promise<Response> {
  const headers = new Headers(options.headers)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }

  const res = await fetch(path, { ...options, headers })

  // access 过期 → 用 refresh 续期一次再重试
  if (res.status === 401 && retry && (await tryRefresh())) {
    return apiFetch(path, options, false)
  }
  return res
}
