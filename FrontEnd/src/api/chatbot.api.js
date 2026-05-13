import axiosClient from './axiosClient'
import useAuthStore from '@/store/authStore'

const _CHATBOT_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080'

/**
 * Low-level SSE streaming helper.
 * Calls onToken for each text token, onDone when the stream closes.
 * Throws on HTTP errors; AbortError is swallowed (caller controls cancel).
 */
async function _streamPost(path, body, { onToken, onDone, onStatus, signal } = {}) {
  const token = useAuthStore.getState().token
  const resp = await fetch(`${_CHATBOT_BASE}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  })

  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop()
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const payload = JSON.parse(line.slice(6))
          if (payload.done) onDone?.(payload)
          else if (payload.token) onToken?.(payload.token)
          else if (payload.status) onStatus?.(payload.status)
        } catch { /* ignore malformed lines */ }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') throw err
  }
}

export const chatbotApi = {
  // ── Session list & history (Java recruitment-service via /api/chatbot/**) ──

  getSessions: (params) =>
    axiosClient.get('/api/chatbot/sessions', { params }),

  /** Cursor-based: params = { limit, beforeId? } */
  getSessionHistory: (sessionId, params) =>
    axiosClient.get(`/api/chatbot/sessions/${sessionId}`, { params }),

  // ── HR Chat (Python chatbot-service via /chatbot/**) ──

  createHRSession: (hrId, positionId, mode) =>
    axiosClient.post('/chatbot/hr/session', {
      hr_id: String(hrId),
      position_id: positionId,
      mode,
    }),

  sendHRMessage: (sessionId, query, hrId, positionId, mode) =>
    axiosClient.post('/chatbot/hr/chat', {
      session_id: sessionId,
      query,
      hr_id: String(hrId),
      position_id: positionId,
      mode,
    }),

  /** SSE streaming. callbacks: { onToken, onDone, signal } */
  streamHRMessage: (sessionId, query, hrId, positionId, mode, callbacks) =>
    _streamPost('/chatbot/hr/chat/stream', {
      session_id: sessionId,
      query,
      hr_id: String(hrId),
      position_id: positionId,
      mode,
    }, callbacks),

  // ── Candidate Chat (Python chatbot-service via /chatbot/**) ──

  createCandidateSession: (userId, positionId = null) =>
    axiosClient.post('/chatbot/candidate/session', {
      user_id: String(userId),
      position_id: positionId,
    }),

  sendCandidateMessage: (sessionId, query, candidateId, cvId = null) =>
    axiosClient.post('/chatbot/candidate/chat', {
      session_id: sessionId,
      query,
      candidate_id: String(candidateId),
      cv_id: cvId,
    }),

  /** SSE streaming. callbacks: { onToken, onDone, signal } */
  streamCandidateMessage: (sessionId, query, candidateId, cvId, callbacks) =>
    _streamPost('/chatbot/candidate/chat/stream', {
      session_id: sessionId,
      query,
      candidate_id: String(candidateId),
      cv_id: cvId,
    }, callbacks),
}
