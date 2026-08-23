import { useEffect, useState } from 'react'

// Agent 运行中的进度显示：转圈 + 已耗时 + 可选步骤清单（Research 用真实节点步骤）
export default function AgentRun({
  message = '正在运行…',
  steps = [],
}: {
  message?: string
  steps?: string[]
}) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="agentrun">
      <div className="ar-head">
        <span className="spinner" aria-hidden="true" />
        <span className="ar-msg">{message}</span>
        <span className="ar-time muted mono">{(elapsed / 60).toFixed(0)}m{elapsed % 60}s</span>
      </div>
      {steps.length > 0 && (
        <ul className="ar-steps">
          {steps.map((s, i) => (
            <li key={i} className={i === steps.length - 1 ? 'current' : 'done'}>
              <span className="dot">{i === steps.length - 1 ? '…' : '✓'}</span>
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
