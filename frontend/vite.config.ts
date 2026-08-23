import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发代理：/api 与 /health 转发到后端（app.main:app @8000），避免跨域。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
