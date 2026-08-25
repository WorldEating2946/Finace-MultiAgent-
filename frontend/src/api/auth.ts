// 认证 API 封装
import type { Tokens, User } from '../types'
import { apiFetch, setSession, errorMessage } from './client'

export async function login(identifier: string, password: string): Promise<Tokens> {
  const res = await apiFetch('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ identifier, password }),
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  const tokens = (await res.json()) as Tokens
  setSession(tokens)
  return tokens
}

export async function logout(): Promise<void> {
  await apiFetch('/api/v1/auth/logout', { method: 'POST' })
  setSession(null)
}

export async function me(): Promise<User> {
  const res = await apiFetch('/api/v1/auth/me')
  if (!res.ok) throw new Error(await errorMessage(res))
  return (await res.json()) as User
}
