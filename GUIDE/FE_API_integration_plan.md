# FE ↔ BE API Integration Plan

## Tổng quan

Mục tiêu: thay thế toàn bộ `mockData.js` bằng API thực từ backend. Plan chia 5 phase theo thứ tự ưu tiên.

---

## Bug Critical — Phải fix trước mọi thứ

**`axiosClient.js` đang có `baseURL: 'http://localhost:8080/api/v1'`.**
API Gateway KHÔNG có prefix `/api/v1` trong bất kỳ route nào (routes là `/positions/**`, `/cv/**`, `/chatbot/**`...).
→ Mọi API call hiện tại đều bị 404 khi gọi BE thực.

**Fix ngay trong Phase 1:**
```js
// axiosClient.js
const axiosClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8080',
  // ...
})
```

---

## Phân tích Gap — Field Mapping

### Position

| Mock field | Real API field | Ghi chú |
|---|---|---|
| `name` | `title` | Rename |
| `level` | `seniority` | Rename |
| `language` | `skills` | Rename, kiểu `String[]` |
| `minFitScore` | `minimumFitScore` | Rename |
| `internalCount` | `internalCount` | ✅ Confirmed có trong response |
| `externalCount` | `externalCount` | ✅ Confirmed có trong response |

### Candidate / CandidateCV

| Mock field | Real API field | Ghi chú |
|---|---|---|
| `type` | `sourceType` | `INTERNAL` / `EXTERNAL` |
| `interviewDate` | `interviewSchedule` | Rename |
| `analysis.feedback` | `analysis.aiAssessment` | → hiển thị ở "Reason for Match" |
| `analysis.skillMatch` | Bỏ | Gộp vào `aiAssessment` |
| `analysis.skillMiss` | Bỏ | Gộp vào `aiAssessment` |
| `driveFileUrl` | `driveUrl` | → hiển thị ở nút "View CV" |

### Auth

| Mock | Real | Ghi chú |
|---|---|---|
| Hardcoded fake token | `POST /auth/login` | Phone + password |
| Fake user object | `data.account` | id, name, email, phone, role |
| No refresh | `data.refreshToken` | Lưu vào localStorage |

---

## Thiếu sót ở BE (phải thêm trước khi làm FE tương ứng)

| # | Thiếu gì | Cần cho Phase | Giải pháp |
|---|---|---|---|
| 1 | `GET /cv/me` cho Candidate | Phase 4 - CVPage | Thêm vào `CandidateCVController`, query theo `X-User-Id` header + `sourceType = EXTERNAL` + `positionId IS NULL` |
| 2 | Python chatbot-service nhận path `/chatbot/**` | Phase 5 - Chatbot | Thêm prefix `/chatbot` vào FastAPI app trong `chatbot-service/app/main.py` (xem chi tiết ở Phase 5) |

---

## Cấu trúc API Layer mới

```
src/api/
├── axiosClient.js       (sửa baseURL)
├── auth.api.js          (mới)
├── position.api.js      (mới)
├── candidate.api.js     (mới)
├── upload.api.js        (mới)
├── chatbot.api.js       (mới)
└── mockData.js          (xóa sau khi migration xong)
```

---

## Phase 1 — Auth + axiosClient Fix

### Mục tiêu
- Fix baseURL bug
- Kết nối Login.jsx với `auth-service` thực
- Lưu JWT và handle refresh token

### BE Endpoint
```
POST /auth/login
Body:     { Phone: string, password: string }
Response: { data: { accessToken, refreshToken, account: { id, name, email, phone, role } } }

POST /auth/refresh-token
Body:     { refreshToken: string }
Response: { data: { accessToken, refreshToken } }
```

### Việc cần làm

**1. Fix `axiosClient.js`**
- Đổi `baseURL` thành `http://localhost:8080`
- Thêm logic refresh token trong interceptor 401:
  - Gọi `POST /auth/refresh-token`
  - Nếu thành công: cập nhật token mới, retry request gốc
  - Nếu thất bại: logout + redirect `/login`

**2. Tạo `src/api/auth.api.js`**
```js
import axiosClient from './axiosClient'

export const authApi = {
  login: (phone, password) =>
    axiosClient.post('/auth/login', { Phone: phone, password }),
  logout: () => axiosClient.post('/auth/logout'),
  refreshToken: (token) =>
    axiosClient.post('/auth/refresh-token', { refreshToken: token }),
}
```

**3. Cập nhật `Login.jsx`**
- Thay 2 fake button bằng form nhập Phone + Password (Ant Design Form)
- Gọi `authApi.login()` → map `account.role` để navigate đúng route
- Hiển thị error message nếu login fail

**4. Cập nhật `authStore.js`**
- Thêm `refreshToken` vào state và persist vào localStorage

---

## Phase 2 — HR Positions Page

### Mục tiêu
`PositionsPage.jsx` dùng API thực, tất cả CRUD + toggle + score đều gọi BE.

### BE Endpoints
```
GET    /positions?keyword=&isActive=&page=0&size=20
POST   /positions                            (multipart: title, seniority, skills, minimumFitScore, file)
PUT    /positions/{id}                       (body: { title, seniority, skills })
DELETE /positions                            (body: { ids: number[] })
PATCH  /positions/{id}/minimum-fit-score     (body: { score: number })
PATCH  /positions/{id}/toggle-active
GET    /positions/jd/{id}/text
```

### Tạo `src/api/position.api.js`
```js
import axiosClient from './axiosClient'

export const positionApi = {
  filter: (params) => axiosClient.get('/positions', { params }),
  create: (formData) => axiosClient.post('/positions', formData),
  update: (id, data) => axiosClient.put(`/positions/${id}`, data),
  deleteMany: (ids) => axiosClient.delete('/positions', { data: { ids } }),
  updateMinScore: (id, score) =>
    axiosClient.patch(`/positions/${id}/minimum-fit-score`, { score }),
  toggleActive: (id) => axiosClient.patch(`/positions/${id}/toggle-active`),
  getJDText: (id) => axiosClient.get(`/positions/jd/${id}/text`),
}
```

### Cập nhật `PositionsPage.jsx`
- Thay `fetchPositions` → `positionApi.filter({ keyword, isActive, page, size })`
- Fix field names: `title`, `seniority`, `skills`, `minimumFitScore`, `internalCount`, `externalCount`
- Thêm server-side pagination từ `PageResponse.totalElements`
- Thay tất cả mutations mock → `positionApi.*`
- `create` dùng `new FormData()` append từng field + file JD
- Khi click số candidates (internal/external) → navigate `/hr/candidates?positionId={id}&sourceType=INTERNAL|EXTERNAL`

---

## Phase 3 — HR Candidates Page

### Mục tiêu
`CandidatesPage.jsx` dùng API thực, full recruitment pipeline hoạt động.

### BE Endpoints
```
GET   /cv/candidates?keyword=&positionId=&recruitmentStage=&sourceType=&cvStatus=&page=0&size=20
GET   /cv/{cvId}
POST  /cv/{cvId}/schedule-interview    (body: { interviewDate, customMessage })
POST  /cv/{cvId}/reschedule-interview  (body: { interviewDate, customMessage })
POST  /cv/{cvId}/send-offer            (body: { benefit, salary, startDate, offerExpirationDate, additionalNote })
PATCH /cv/{cvId}/stage                 (body: { stage: 'ACCEPTED' | 'REJECTED' })

POST  /upload/hr/cv                    (multipart: positionId, files[])
```

### Tạo `src/api/candidate.api.js`
```js
import axiosClient from './axiosClient'

export const candidateApi = {
  filter: (params) => axiosClient.get('/cv/candidates', { params }),
  getById: (cvId) => axiosClient.get(`/cv/${cvId}`),
  getMyCV: () => axiosClient.get('/cv/me'),               // Candidate only
  deleteMany: (ids) => axiosClient.delete('/cv', { data: { ids } }),
  updateInfo: (cvId, params) =>
    axiosClient.put(`/cv/${cvId}`, null, { params }),     // name, email as query params
  scheduleInterview: (cvId, data) =>
    axiosClient.post(`/cv/${cvId}/schedule-interview`, data),
  rescheduleInterview: (cvId, data) =>
    axiosClient.post(`/cv/${cvId}/reschedule-interview`, data),
  sendOffer: (cvId, data) =>
    axiosClient.post(`/cv/${cvId}/send-offer`, data),
  updateStage: (cvId, stage) =>
    axiosClient.patch(`/cv/${cvId}/stage`, { stage }),
}
```

### Tạo `src/api/upload.api.js`
```js
import axiosClient from './axiosClient'

export const uploadApi = {
  hrUploadCVs: (positionId, files) => {
    const fd = new FormData()
    fd.append('positionId', positionId)
    files.forEach(f => fd.append('files', f))
    return axiosClient.post('/upload/hr/cv', fd)
  },
  candidateUploadCV: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return axiosClient.post('/upload/candidate/cv', fd)
  },
}
```

### Cập nhật `CandidatesPage.jsx`
- Thay `fetchCandidates` → `candidateApi.filter(params)` với server-side pagination
- Thay `fetchPositions` → `positionApi.filter({})` cho dropdown filter
- Fix field names: `sourceType`, `interviewSchedule`, `aiAssessment`, `driveUrl`
- Thay tất cả mutations → gọi `candidateApi.*`
- Khi navigate từ PositionsPage với `?positionId=&sourceType=` → đọc URL params để pre-fill filter

---

## Phase 4 — Candidate Portal

### 4.1 CareerPage

**BE Endpoint:** `GET /positions?isActive=true&size=50`

**Cập nhật:**
- Thay `fetchActivePositions()` → `positionApi.filter({ isActive: true, size: 50 })`
- Fix field: `title`, `skills`, `seniority`

### 4.2 CVPage

**Cần thêm BE endpoint trước:**
```java
// CandidateCVController.java
@GetMapping("/me")
@PreAuthorize("hasRole('CANDIDATE')")
public ResponseEntity<ApiResponse<CandidateCVResponse>> getMyCv(
    @RequestHeader("X-User-Id") String userId) {
  return ResponseEntity.ok(
    ApiResponse.success(candidateCVService.getMasterCvByCandidate(userId))
  );
}
// Service: query by candidateId = userId AND sourceType = EXTERNAL AND positionId IS NULL
```

**BE Endpoints:**
```
GET    /cv/me                          (Candidate only — cần thêm vào BE)
POST   /upload/candidate/cv            (multipart: file)
PUT    /cv/{cvId}?name=&email=
DELETE /cv                             (body: { ids: [cvId] })
```

**Cập nhật `CVPage.jsx`:**
- Thay mock fetch → `candidateApi.getMyCV()` (queryKey: `['myCV']`)
- Upload mới → `uploadApi.candidateUploadCV(file)` → invalidate `['myCV']`
- Update info → `candidateApi.updateInfo(cvId, { name, email })`
- Delete → `candidateApi.deleteMany([cvId])` → invalidate `['myCV']`

---

## Phase 5 — Chatbot

### Vấn đề routing — Cần fix BE trước

**Hiện trạng:** API Gateway route `/chatbot/**` → Python chatbot-service (port 8085).
Python service có routes `/hr/session`, `/hr/chat`, `/candidate/session`, `/candidate/chat`.
Nhưng gateway forward full path → service nhận `/chatbot/hr/chat` trong khi route chỉ có `/hr/chat` → **404**.

**Fix trong `chatbot-service/app/main.py`:**
```python
# Thêm root_path để FastAPI nhận cả prefix /chatbot
app = FastAPI(root_path="/chatbot")
```
Hoặc đặt prefix cho toàn bộ router:
```python
app.include_router(hr_router, prefix="/chatbot/hr")
app.include_router(candidate_router, prefix="/chatbot/candidate")
```

### BE Endpoints (sau khi fix prefix)
```
# Python chatbot-service (qua gateway /chatbot/**)
POST /chatbot/hr/session
Body:     { hr_id, position_id, mode: "INTERNAL" | "EXTERNAL" }
Response: { session_id }

POST /chatbot/hr/chat
Body:     { session_id, query, hr_id, position_id, mode }
Response: { answer, metadata }         ← JSON, không phải SSE (v1)

POST /chatbot/candidate/session
Body:     { user_id, position_id? }
Response: { session_id }

POST /chatbot/candidate/chat
Body:     { session_id, query, candidate_id, cv_id? }
Response: { answer, metadata }

# Java recruitment-service (qua gateway /api/chatbot/**)
GET /api/chatbot/sessions?page=0&size=20
GET /api/chatbot/sessions/{sessionId}
```

> **Note SSE:** Chat hiện trả JSON, không phải SSE streaming. Giữ nguyên JSON cho v1 (dùng loading state). Nâng cấp lên SSE streaming trong sprint riêng sau khi integration xong, vì cần sửa cả Python service (LangGraph `.astream()` + `StreamingResponse`) và thêm custom `useStreamingChat` hook ở FE.

### Tạo `src/api/chatbot.api.js`
```js
import axiosClient from './axiosClient'

export const chatbotApi = {
  // Session list + history (Java service)
  getSessions: (params) => axiosClient.get('/api/chatbot/sessions', { params }),
  getSessionHistory: (sessionId) =>
    axiosClient.get(`/api/chatbot/sessions/${sessionId}`),

  // HR Chat (Python service qua /chatbot/**)
  createHRSession: (hrId, positionId, mode) =>
    axiosClient.post('/chatbot/hr/session', { hr_id: hrId, position_id: positionId, mode }),
  sendHRMessage: (sessionId, query, hrId, positionId, mode) =>
    axiosClient.post('/chatbot/hr/chat', { session_id: sessionId, query, hr_id: hrId, position_id: positionId, mode }),

  // Candidate Chat (Python service qua /chatbot/**)
  createCandidateSession: (userId, positionId) =>
    axiosClient.post('/chatbot/candidate/session', { user_id: userId, position_id: positionId }),
  sendCandidateMessage: (sessionId, query, candidateId, cvId) =>
    axiosClient.post('/chatbot/candidate/chat', { session_id: sessionId, query, candidate_id: candidateId, cv_id: cvId }),
}
```

### Cập nhật `HRChatbotPage.jsx`
- Sidebar: load session list từ `chatbotApi.getSessions()`
- Chọn session cũ → `chatbotApi.getSessionHistory(sessionId)` để restore messages
- New chat: `chatbotApi.createHRSession(user.id, positionId, mode)` → lưu `sessionId`
- Gửi message:
  1. Append user message vào UI ngay
  2. Set loading state
  3. `chatbotApi.sendHRMessage(...)` → nhận `answer` → append vào UI
- Mode mapping: "Internal" → `"HR_MODE"`, "External" → `"CANDIDATE_MODE"`

### Candidate Chatbot (embedded trong CareerPage)
- Mỗi PositionCard có nút Chat → mở drawer chatbot
- `chatbotApi.createCandidateSession(user.id)` khi open drawer. Scope cho Candidate chatbot là CV của candidate và positionId của các position active. 
- Gửi message → `chatbotApi.sendCandidateMessage(...)`

---

## Thứ tự thực hiện

```
Phase 1: Auth + baseURL fix    → Login hoạt động với real JWT
    ↓
Phase 2: HR Positions          → CRUD positions, toggle, score sync
    ↓
[BE: thêm GET /cv/me]
[BE: fix chatbot-service prefix /chatbot]
    ↓
Phase 3: HR Candidates         → Full recruitment pipeline
    ↓
Phase 4: Candidate Portal      → CareerPage + CVPage
    ↓
Phase 5: Chatbot               → HR Chatbot + Candidate Chatbot
```

---

## Decisions đã chốt

| # | Vấn đề | Quyết định |
|---|---|---|
| baseURL | axiosClient có `/api/v1` thừa | Fix về `http://localhost:8080` |
| Chatbot routing | Gateway forward full path, Python route thiếu prefix | Thêm prefix `/chatbot` vào Python FastAPI |
| Chat response | JSON hay SSE | JSON cho v1, SSE là enhancement sau |
| DELETE batch | Body hay query param | Request body `{ ids: number[] }` |
| internalCount/externalCount | Có trong PositionResponse không | ✅ Có, dùng để navigate filter candidates |
| aiAssessment | Display ở đâu | Cột "Reason for Match" trong Candidates table |
| driveUrl | Display ở đâu | Nút "View CV" trong Candidates table |
