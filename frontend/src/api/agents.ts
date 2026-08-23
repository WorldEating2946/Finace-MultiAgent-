// 各 Agent 独立端点封装 + 知识库（上传/检索）
// 走 apiFetch（自动带 Bearer + 401 自动 refreshtoken）
import { apiFetch, getAccessToken, errorMessage } from './client'

export type UploadEvent = {
  type?: string
  node?: string
  embedded?: number
  total?: number
  chunk_count?: number
  message?: string
  data?: unknown
}

function parseSse(frame: string): UploadEvent | null {
  let event = 'message'
  let dataStr = ''
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
  }
  if (!dataStr) return null
  try {
    const d = JSON.parse(dataStr) as UploadEvent
    return { ...d, type: d.type || event }
  } catch {
    return null
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await errorMessage(res))
  return (await res.json()) as T
}

// ── Financial ─────────────────────────────────────────────
export async function runFinancial(ticker: string, company = '') {
  const r = await apiFetch('/api/v1/financial', {
    method: 'POST',
    body: JSON.stringify({ ticker, company }),
  })
  return json<any>(r)
}

// ── Research（SSE 流式：节点步骤 + 最终结果）──────────────
export async function runResearch(
  company: string,
  query = '',
  fast = true,
  onProgress?: (step: { node?: string; message?: string }) => void,
): Promise<any> {
  const token = getAccessToken()
  const res = await fetch('/api/v1/research/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ company, query, fast }),
  })
  if (res.status === 401) throw new Error('登录已过期，请重新登录')
  if (!res.ok || !res.body) throw new Error(await errorMessage(res))

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let result: any = null
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const evt = parseSse(frame)
      if (evt) {
        if (evt.type === 'step') onProgress?.({ node: evt.node, message: evt.message })
        if (evt.type === 'error') throw new Error(evt.message ?? 'Research 失败')
        if (evt.type === 'done') result = evt
      }
    }
  }
  return result?.data ?? null
}

// ── Sentiment + Risk ──────────────────────────────────────
export async function runSentiment(symbol: string, companyName: string, days = 30) {
  const r = await apiFetch('/api/v1/sentiment', {
    method: 'POST',
    body: JSON.stringify({ symbol, company_name: companyName, days }),
  })
  return json<any>(r)
}
export async function runRisk(symbol: string, companyName: string, days = 30) {
  const r = await apiFetch('/api/v1/risk', {
    method: 'POST',
    body: JSON.stringify({ symbol, company_name: companyName, days }),
  })
  return json<any>(r)
}
export async function runSentimentRiskFull(symbol: string, companyName: string, days = 30) {
  const r = await apiFetch('/api/v1/sentiment-risk/full', {
    method: 'POST',
    body: JSON.stringify({ symbol, company_name: companyName, days }),
  })
  return json<any>(r)
}

// ── 知识库 ────────────────────────────────────────────────
export async function searchKnowledge(query: string, company: string, topK = 5) {
  const r = await apiFetch(
    `/api/v1/knowledge/search?query=${encodeURIComponent(query)}&company=${encodeURIComponent(company)}&top_k=${topK}`,
  )
  return json<any>(r)
}

export async function getSystemStatus() {
  const r = await apiFetch('/api/v1/health/status')
  return json<any>(r)
}

// ── 历史（保存 / 列出 Agent 运行结果，供综合报告复用）──────
export async function saveHistory(company: string, agentType: string, result: unknown, ticker = '') {
  const r = await apiFetch('/api/v1/history', {
    method: 'POST',
    body: JSON.stringify({ company, agent_type: agentType, result, ticker }),
  })
  return json<any>(r)
}

export async function listHistory(company = '', agentType = '', limit = 100) {
  const params = new URLSearchParams()
  if (company) params.set('company', company)
  if (agentType) params.set('agent_type', agentType)
  params.set('limit', String(limit))
  const r = await apiFetch(`/api/v1/history?${params.toString()}`)
  return json<any>(r)
}

export async function getKnowledgeCompanies() {
  const r = await apiFetch('/api/v1/knowledge/companies')
  return json<any>(r)
}

export async function uploadDocument(
  file: File,
  company: string,
  sourceType = '',
  onProgress?: (e: UploadEvent) => void,
): Promise<UploadEvent> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('company', company)
  fd.append('source_type', sourceType)
  const token = getAccessToken()
  const res = await fetch('/api/v1/knowledge/upload', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: fd,
  })
  if (res.status === 401) throw new Error('登录已过期，请重新登录')
  if (!res.ok || !res.body) throw new Error(`上传失败 (${res.status})`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let result: UploadEvent = { type: 'message' }
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const evt = parseSse(frame)
      if (evt) {
        onProgress?.(evt)
        if (evt.type === 'done' || evt.type === 'error') {
          if (evt.type === 'error') throw new Error(evt.message ?? '入库失败')
          return evt
        }
      }
    }
  }
  return result
}
