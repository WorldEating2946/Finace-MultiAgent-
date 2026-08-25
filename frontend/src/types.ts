// 后端 DTO 类型（对应 app/schemas/auth.py 与 app/api/analyze_stream.py）

export interface User {
  id: number
  username: string
  email: string
  full_name?: string | null
  is_active: boolean
  is_admin: boolean
  created_at: string
}

export interface Tokens {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user: User
}

// SSE 事件（app/api/analyze_stream.py _sse 帧）
export type StreamEvent =
  | { type: 'run_start'; company: string; ticker: string }
  | { type: 'node_end'; node: string; status: string; summary: string }
  | {
      type: 'report_generated'
      report_id: string
      html_path: string
      markdown_path: string
    }
  | {
      type: 'done'
      report_id: string
      markdown: string
      html_path: string
      markdown_path: string
    }
  | { type: 'error'; message: string }
