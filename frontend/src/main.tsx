import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { initialTheme } from './hooks/useTheme'

// 渲染前设好初始主题（本地记忆 > 系统偏好），避免明暗闪烁
document.documentElement.dataset.theme = initialTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
