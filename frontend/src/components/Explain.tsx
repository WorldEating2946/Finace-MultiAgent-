import type { ReactNode } from 'react'

// 每个 Agent 视图顶部的「这个 Agent 做什么」解释卡
export default function Explain({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="explainer">
      <div className="ex-title">{title}</div>
      <div className="ex-body">{children}</div>
    </div>
  )
}
