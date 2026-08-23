import ReactMarkdown from 'react-markdown'

export default function ReportView({ markdown }: { markdown: string }) {
  return (
    <div className="report">
      <ReactMarkdown>{markdown}</ReactMarkdown>
    </div>
  )
}
