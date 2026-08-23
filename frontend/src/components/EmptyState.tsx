import type { ReactNode } from 'react'

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
      {icon && <div className="esico">{icon}</div>}
      <div className="estitle">{title}</div>
      {body && <div className="esbody">{body}</div>}
      {action && <div className="esaction">{action}</div>}
    </div>
  )
}
