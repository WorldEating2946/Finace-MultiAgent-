// 防御式取值工具
export function items<T = string>(v: unknown): T[] {
  if (Array.isArray(v)) return v as T[]
  if (v == null) return []
  if (typeof v === 'string' && v) return [v as T]
  return []
}

export function asStr(v: unknown): string {
  if (v == null) return ''
  return typeof v === 'string' ? v : String(v)
}

export function asNum(v: unknown): number | null {
  if (v == null) return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}
