import axiosClient from './axiosClient'

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
}
