import type { ReactNode } from 'react'

// 空状态图标：单色线描（避免彩色 emoji 削弱专业感）。
// 兼容旧视图传入的 emoji 字符串，先映射为 SVG，未命中则按文本渲染。
function Ico({ d }: { d: string }) {
  return (
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  )
}

const ICONS: Record<string, ReactNode> = {
  '🔍': <Ico d="M11 11m-7 0a7 7 0 1 0 14 0a7 7 0 1 0-14 0M21 21l-4.3-4.3" />, // 搜索
  '💰': <Ico d="M3 21h18M7 17V9M12 17V5M17 17v-4" />, // 财务
  '📰': <Ico d="M4 5h12v14H4zM18 7h3v12h-14M8 9h4M8 13h4M8 17h4" />, // 新闻
  '🗂': <Ico d="M12 7v5l3 2M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3 9M3 3v6h6" />, // 历史
  '📚': <Ico d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />, // 研究
}

export default function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: string
  title: string
  body?: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      {icon && <div className="esico">{ICONS[icon] ?? icon}</div>}
      <div className="estitle">{title}</div>
      {body && <div className="esbody">{body}</div>}
      {action && <div className="esaction">{action}</div>}
    </div>
  )
}
