import { useEffect, useState } from 'react'
import { getSystemStatus } from '../api/agents'
import Explain from '../components/Explain'
import { asStr } from '../util'

type Svc = { name: string; ok: boolean; detail: string; ms: number }

export default function SystemStatusView() {
  const [status, setStatus] = useState<{ overall?: boolean; checked_at?: string; services?: Svc[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  async function load() {
    setLoading(true)
    setErr('')
    try {
      const r = await getSystemStatus()
      setStatus(r?.data ?? {})
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const svcs = status?.services ?? []
  const overall = status?.overall

  return (
    <div>
      <Explain title="系统就绪状态">
        查看各 API / 依赖的可用情况：数据库、Milvus、akshare、Embedding、Reranker。
        绿色 = 可用，红色 = 异常（含原因与时延）。首次 Research 会加载 Embedding 模型，此处会显示设备（GPU/CPU）与是否已加载。
      </Explain>

      <div className="card">
        <div className="row">
          <div className="muted small">
            {overall === true ? '✅ 全部就绪' : overall === false ? '⚠️ 部分未就绪' : '—'}
            {status?.checked_at ? ` · ${new Date(status.checked_at).toLocaleTimeString()}` : ''}
          </div>
          <button className="ghost small" onClick={load} disabled={loading}>
            {loading ? '检测中…' : '刷新'}
          </button>
        </div>
        {err && <div className="error">{err}</div>}

        {svcs.map((s) => (
          <div key={s.name} className="svc">
            <span className={`sdot ${s.ok ? 'ok' : 'bad'}`} aria-hidden="true" />
            <span className="svc-name mono">{s.name}</span>
            <span className="svc-detail muted">{asStr(s.detail)}</span>
            <span className="svc-ms mono muted">{s.ms}ms</span>
          </div>
        ))}
      </div>
    </div>
  )
}
