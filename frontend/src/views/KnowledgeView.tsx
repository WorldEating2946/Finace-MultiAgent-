import { useEffect, useState, type ChangeEvent } from 'react'
import { uploadDocument, searchKnowledge, getKnowledgeCompanies } from '../api/agents'
import { asStr } from '../util'
import Explain from '../components/Explain'

type Chunk = { chunk_id?: string; text?: string; source?: string; page?: unknown; chapter?: unknown }

const SOURCE_TYPES = ['annual_report', 'research_report', 'policy', 'news']

export default function KnowledgeView() {
  // ── 上传 ──
  const [file, setFile] = useState<File | null>(null)
  const [uCompany, setUCompany] = useState('宁德时代')
  const [srcType, setSrcType] = useState('annual_report')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<{ embedded: number; total: number } | null>(null)
  const [uploadMsg, setUploadMsg] = useState('')

  // ── 检索 ──
  const [q, setQ] = useState('宁德时代的主营业务')
  const [sCompany, setSCompany] = useState('宁德时代')
  const [searching, setSearching] = useState(false)
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [companies, setCompanies] = useState<Array<{ name: string; ingested: boolean; source_types?: Array<{ type: string; count: number }> }>>([])
  const [err, setErr] = useState('')

  async function loadCompanies() {
    try {
      const r = await getKnowledgeCompanies()
      setCompanies(r?.data?.companies ?? [])
    } catch {
      setCompanies([])
    }
  }
  useEffect(() => {
    void loadCompanies()
  }, [])

  async function doUpload() {
    setErr('')
    setProgress(null)
    setUploadMsg('')
    if (!file) return
    setUploading(true)
    try {
      const evt = await uploadDocument(file, uCompany, srcType, (e) => {
        if (e.type === 'upload_progress' && e.total) setProgress({ embedded: e.embedded ?? 0, total: e.total })
      })
      setUploadMsg(`已入库 ${evt.chunk_count ?? 0} 个 chunk → ${uCompany}`)
    } catch (ex) {
      setErr((ex as Error).message)
      setProgress(null)
    } finally {
      setUploading(false)
    }
  }

  async function doSearch() {
    setErr('')
    setSearching(true)
    try {
      const r = await searchKnowledge(q, sCompany, 5)
      setChunks(r?.data?.chunks ?? [])
    } catch (ex) {
      setErr((ex as Error).message)
    } finally {
      setSearching(false)
    }
  }

  return (
    <div>
      <Explain title="知识库 · 入库与检索">
        上传文档（PDF/TXT/MD）→ 切片 → BGE-M3 嵌入入库（GPU 优先）。Research 的 RAG 检索依赖这里的文档，
        所以先给目标公司上传资料，再回 Research / 总览跑分析。入库在下方实时显示进度条。
      </Explain>

      <div className="card">
        <div className="card-title">已入库公司 · 哪些能查（✓ 有文档可检索 / ✗ 没文档，Research 会空）</div>
        {companies.length === 0 ? (
          <div className="muted small">暂无入库数据，先上传文档。</div>
        ) : (
          companies.map((c) => (
            <div key={c.name} className="svc">
              <span className={`sdot ${c.ingested ? 'ok' : 'bad'}`} aria-hidden="true" />
              <span className="svc-name">{c.name}</span>
              <span className="svc-detail">
                {c.ingested && (c.source_types ?? []).length > 0 ? (
                  (c.source_types ?? []).map((st) => (
                    <span key={st.type} className="tag">{st.type} · {st.count}</span>
                  ))
                ) : (
                  <span className="muted">未入库</span>
                )}
              </span>
            </div>
          ))
        )}
        <div className="muted small" style={{ marginTop: 10 }}>
          ✓ = Research/总览 能查到该公司的真实内容；✗ = 需先上传该公司的年报/研报/政策。
        </div>
      </div>

      <div className="card">
        <div className="muted small" style={{ marginBottom: 10 }}>离线文档入库（BGE-M3 嵌入 → 向量库）</div>
        <div className="row">
          <label className="grow">
            文件（.md/.txt/.pdf）
            <input
              type="file"
              accept=".md,.txt,.pdf"
              onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            公司
            <input value={uCompany} onChange={(e) => setUCompany(e.target.value)} />
          </label>
          <label>
            类型
            <select value={srcType} onChange={(e) => setSrcType(e.target.value)}>
              {SOURCE_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <button onClick={doUpload} disabled={uploading || !file}>
            {uploading ? '入库中…' : '上传入库'}
          </button>
        </div>
        {progress && (
          <div className="muted small" style={{ marginTop: 10 }}>
            嵌入进度 {progress.embedded} / {progress.total} · {Math.round((progress.embedded / (progress.total || 1)) * 100)}%
            <div className="progressbar">
              <div className="progressbar-fill" style={{ width: `${(progress.embedded / (progress.total || 1)) * 100}%` }} />
            </div>
          </div>
        )}
        {uploadMsg && <div className="muted small">✓ {uploadMsg}</div>}
      </div>

      <div className="card">
        <div className="row">
          <label className="grow">
            检索问题
            <input value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          <label>
            公司
            <input value={sCompany} onChange={(e) => setSCompany(e.target.value)} />
          </label>
          <button onClick={doSearch} disabled={searching || !q.trim()}>
            {searching ? '检索中…' : '语义检索'}
          </button>
        </div>
        {err && <div className="error">{err}</div>}
        {chunks.length > 0 && (
          <div style={{ marginTop: 10 }}>
            {chunks.map((c) => (
              <div key={c.chunk_id} className="card" style={{ marginTop: 8, padding: '12px 14px' }}>
                <div className="muted small">{[c.source, c.page ? `p${c.page}` : '', c.chapter ? `ch.${c.chapter}` : ''].filter(Boolean).join(' · ')}</div>
                <div style={{ marginTop: 4 }}>{asStr(c.text)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
