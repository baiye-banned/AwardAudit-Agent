const API_BASE = '/api'


export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })

  return readJsonResponse(response)
}


export async function askAgent({ sessionId, question, history }) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      history,
    }),
  })

  return readJsonResponse(response)
}


async function readJsonResponse(response) {
  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || '请求失败，请稍后重试')
  }

  return data
}

