// 投研分析 SSE 流式消费（fetch + ReadableStream，因 EventSource 无法带 Bearer 头）
import type { StreamEvent } from '../types'
import { getAccessToken } from './client'

export interface AnalyzeParams {
  company: string
  ticker?: string
  user_query?: string
}

// 逐帧解析 SSE（event: <type>\ndata: <json>）并回调 onEvent
function parseFrame(frame: string): StreamEvent | null {
  let event = 'message'
  let dataStr = ''
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
  }
  if (!dataStr) return null
  try {
    const evt = JSON.parse(dataStr) as StreamEvent
    // 帧里的 event 字段覆盖 data.type
    return { ...evt, type: (evt.type as string) || event } as StreamEvent
  } catch {
    return null
  }
}

export async function streamAnalyze(
  params: AnalyzeParams,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const token = getAccessToken()
  if (!token) throw new Error('未认证，请先登录')

  const res = await fetch('/api/v1/analyze/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(params),
  })
  if (res.status === 401) throw new Error('登录已过期，请重新登录')
  if (!res.ok || !res.body) throw new Error(`分析请求失败 (${res.status})`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const evt = parseFrame(frame)
      if (evt) onEvent(evt)
      if (evt?.type === 'done' || evt?.type === 'error') return
    }
  }
}
