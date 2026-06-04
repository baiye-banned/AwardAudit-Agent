import assert from 'node:assert/strict'
import test from 'node:test'

import { askAgent } from '../src/api/analysis.js'


test('askAgent sends session, question and history to backend', async () => {
  const calls = []

  global.fetch = async (url, options) => {
    calls.push({ url, options })
    return {
      ok: true,
      async json() {
        return { answer: 'ok', charts: [], audit_result: { report: '审核报告' } }
      },
    }
  }

  const result = await askAgent({
    sessionId: 'session-1',
    question: '请分析销售额',
    history: [{ role: 'user', content: '你好' }],
  })

  assert.equal(result.answer, 'ok')
  assert.equal(result.audit_result.report, '审核报告')
  assert.equal(calls[0].url, '/api/chat')
  assert.equal(calls[0].options.method, 'POST')

  const body = JSON.parse(calls[0].options.body)
  assert.equal(body.session_id, 'session-1')
  assert.equal(body.question, '请分析销售额')
  assert.equal(body.history.length, 1)
})
