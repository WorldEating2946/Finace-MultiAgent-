import { useTheme } from '../hooks/useTheme'

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  )
}

export default function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      className="ghost small themebtn"
      onClick={toggle}
      title={`当前${theme === 'dark' ? '深色' : '浅色'}，点击切换`}
      aria-label="切换深浅主题"
    >
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
      <span style={{ marginLeft: 5 }}>{theme === 'dark' ? '浅色' : '深色'}</span>
    </button>
  )
}
