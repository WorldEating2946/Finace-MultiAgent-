import { useTheme } from '../hooks/useTheme'

export default function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      className="ghost small themebtn"
      onClick={toggle}
      title={`当前${theme === 'dark' ? '深色' : '浅色'}，点击切换`}
      aria-label="切换深浅主题"
    >
      {theme === 'dark' ? '☀ 浅色' : '☾ 深色'}
    </button>
  )
}
