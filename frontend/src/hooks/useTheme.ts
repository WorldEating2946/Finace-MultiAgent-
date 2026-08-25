import { useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'
const KEY = 'finaceagent_theme'

function systemTheme(): Theme {
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: light)').matches) {
    return 'light'
  }
  return 'dark'
}

// 初始主题：本地记忆 > 系统偏好（默认跟随系统）
export function initialTheme(): Theme {
  const stored = localStorage.getItem(KEY) as Theme | null
  return stored === 'dark' || stored === 'light' ? stored : systemTheme()
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(KEY, theme)
  }, [theme])

  const toggle = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  return { theme, toggle }
}
