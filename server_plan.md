# Server Modification Plan

## I. Modify Database Entities & Schemas

### 1. Position Table (`Positions.java`)
- [x] Rename `name` -> `title`
- [x] Rename `level` -> `seniority`
- [x] Rename `language` -> `skills` (type `List<String>` or similar)
  - *Example:* `title`: Senior Fullstack Engineer, `seniority`: Senior, `skills`: Java, ReactJs, Golang.
- [x] Add `minimum_fit_score` (Double/Float) - Allow HR to modify in recruitment service, then sync -> chatbot service.

### 2. Candidate CV Table (`CandidateCV.java`)
- [x] Change `source_type` logic: `HR` -> `INTERNAL`, `CANDIDATE` -> `EXTERNAL`
- [x] Remove columns: `scored_at`, `cv_path`
- [x] Update Recruitment Stage enum (`RecruitmentStage.java` or similar): `APPLIED`, `INTERVIEW_SCHEDULED`, `INTERVIEWED`, `REJECTED`, `OFFER`, `ACCEPTED`
- [x] Add `applied_date` (Date/Timestamp)
- [x] Add `interview_schedule` (Date/Timestamp or String)
- [x] Rename `parsed_at` -> `created_at`

### 3. CV Analysis Table (`CVAnalysis.java`)
- [x] Rename `feedback` -> `ai_assessment`
- [x] Remove column: `analyzed_at`
- [x] **Decision on `skill_miss` / `skill_match`**:
  - *Analysis:* **Nên bỏ**. Nếu giữ lại thì tốn token và phải format prompt phức tạp thêm. HR/Candidate thường quan tâm đến đánh giá tổng quan hơn. Ta có thể gom các thông tin về skill match/miss ngắn gọn vào trực tiếp nội dung của `ai_assessment`. Điều này vừa giúp schema gọn gàng, tiết kiệm cost khi call LLM, mà vẫn giữ được giá trị cho người đọc.

---

## II. Refactor Core Methods

### 1. Update JD (Position) & Update CV Methods
- [x] **Update Position:** Giữ method update metadata (`title`, `seniority`, `skills`, `minimumFitScore`). Bỏ chức năng update file JD — nếu cần đổi JD, HR đóng position và tạo mới.
- [x] **Update Candidate CV:** Bỏ chức năng update file CV. Chỉ giữ API update thông tin cơ bản `name`, `email`.

### 2. New Methods — Tổng quan
*(Chi tiết tại Phase 3 bên dưới)*

#### Role: HR
- [ ] Filter CV (unified: keyword, positionId, stage, sourceType, cvStatus)
- [ ] Send Mail + Auto-update Stage (schedule, reschedule, offer)
- [ ] Manual Stage update (ACCEPTED, REJECTED)
- [ ] Filter Position (unified: keyword, isActive)
- [ ] Update `minimumFitScore` + sync chatbot
- [ ] Toggle Active Status cho Position

#### Role: Candidate
- [x] Apply logic (qua chatbot — `finalize_application`).

#### Notification
- [ ] System Notification cho user (Lưu DB, fetch qua HTTP bình thường, không cần WebSocket realtime). *(Deferred)*

---

## III. Thiết kế đã chốt — Phase 3

### Quyết định đã confirm

| # | Vấn đề | Quyết định |
|---|---|---|
| Q1 | Sync `minimumFitScore` | **Phương án A**: Recruitment gọi `PATCH` sang Chatbot Service khi HR update. Chatbot cache in-memory (`dict` Python). |
| Q2 | Guardrail `finalize_application` | Dùng **score thuần** `(technicalScore + experienceScore) / 2 >= minimumFitScore`. `overallStatus` để AI tự sinh ra qua prompt, không dùng làm logic guard. |
| Q3 | Transition `INTERVIEW_SCHEDULED → INTERVIEWED` | **Tự động bằng Spring Scheduler** — `@Scheduled` job chạy mỗi 15 phút, quét các CV có `interviewSchedule < now()` và `stage = INTERVIEW_SCHEDULED`. Không có manual "mark as interviewed". |
| Q4 | Filter Candidate | Filter có **cả 2 field** độc lập: `recruitmentStage` (APPLIED, INTERVIEW_SCHEDULED, ...) **và** `cvStatus` (EMBEDDED, FAILED). Cả 2 đều optional. |

### State Machine — Recruitment Stage

```
APPLIED ──── [scheduleInterview()]       ──► INTERVIEW_SCHEDULED
        └─── [updateStage(REJECTED)]     ──► REJECTED

INTERVIEW_SCHEDULED ── [rescheduleInterview()] ──► INTERVIEW_SCHEDULED (giữ nguyên)
                    └── [@Scheduled auto]       ──► INTERVIEWED

INTERVIEWED ── [sendOffer()]             ──► OFFER
            └── [updateStage(REJECTED)]  ──► REJECTED

OFFER ── [updateStage(ACCEPTED)]         ──► ACCEPTED
      └── [updateStage(REJECTED)]        ──► REJECTED
```

> **`updateStage()` chỉ chấp nhận:** `ACCEPTED`, `REJECTED`. Các transition khác do action method hoặc scheduler thực hiện.

### Sync `minimumFitScore` — Flow Phương án A

```
HR update score
     │
     ▼
PATCH /positions/{id}/minimum-fit-score   (recruitment-service)
     │  1. Update MySQL
     │  2. Fire-and-forget REST (try-catch, không block response HR)
     ▼
PATCH /internal/positions/{id}/minimum-fit-score   (chatbot-service)
     │
     ▼
position_score_cache[position_id] = new_score   (Python in-memory dict)
```

> Nếu chatbot-service down → chỉ log warning, cache sẽ sync lại lần HR update tiếp theo.
> Khi chatbot restart → preload cache bằng cách gọi `GET /internal/positions/scores` từ recruitment.

### Guardrail `finalize_application` — Score-based

```python
# Thay thế _APPLY_ALLOWED_STATUSES (hardcode) bằng score động
avg_score = (technical_score + experience_score) / 2
min_score = position_score_cache.get(position_id, 70)  # default 70

if avg_score < min_score:
    block(f"Điểm {avg_score:.1f} < ngưỡng tối thiểu {min_score}")
```

> `overallStatus` (EXCELLENT_MATCH, GOOD_MATCH, POTENTIAL, POOR_FIT) vẫn tồn tại trong `aiAssessment` để LLM tự sinh cho user đọc, nhưng **không tham gia vào guard logic**.

---

## IV. Execution Steps

### Phase 1 — Hoàn thành ✅
- [x] 1. Sửa Entities (`Positions`, `CandidateCV`, `CVAnalysis`).
- [x] 2. Cập nhật các Enums liên quan (`RecruitmentStage`, `SourceType`).
- [x] 3. Chạy DB Migration (Hibernate auto-ddl).
- [x] 4. Sửa các Repositories đang query dựa trên cột cũ.
- [x] 5. Refactor `PositionService` (bỏ update file JD, fix filter).

### Phase 2 — Python Services & Vector DB ✅
- [x] 6. Đồng bộ Embedding Service (`cv_consumer.py`, `jd_consumer.py`): field mới, Enum `INTERNAL`/`EXTERNAL`.
- [x] 7. Đồng bộ Chatbot Service: Qdrant filter, `aiAssessment`, `recruitment_api.py`, JD payload builder.
- [ ] 8. **Thực thi Qdrant Migration:** Chạy `migrate_qdrant_source_type.py`.

---

### Phase 3 Sprint A — Core Candidate Actions 🔥 *(Ưu tiên 1)*
> **Mục tiêu:** HR filter candidate, schedule/reschedule interview, send offer, và update stage cuối (accepted/rejected).

- [x] **A1.** Thêm `@Query` unified filter vào `CandidateCVRepository`:
  - Filter theo `keyword` (LIKE name/email), `positionId`, `recruitmentStage`, `sourceType`, `cvStatus`.
  - Tất cả tham số nullable, kết hợp AND.
- [x] **A2.** Thêm method `filterCandidates(keyword, positionId, stage, sourceType, cvStatus, page, size)` vào `CandidateCVService` trả về `PageResponse<CandidateCVResponse>`.
- [x] **A3.** Xóa `EmailType.REJECTION` khỏi enum. Xóa case trong `NotificationService`. Xóa file template `email/rejection.html`.
- [x] **A4.** Thêm 3 methods vào `CandidateCVService` (mỗi method: validate stage → update DB → gọi mail):
  - `scheduleInterview(cvId, interviewDate, customMessage)` → `APPLIED → INTERVIEW_SCHEDULED`, set `interviewSchedule`.
  - `rescheduleInterview(cvId, interviewDate, customMessage)` → giữ `INTERVIEW_SCHEDULED`, cập nhật `interviewSchedule`.
  - `sendOffer(cvId, benefit, salary, startDate, offerExpirationDate, additionalNote)` → `INTERVIEWED → OFFER`.
- [x] **A5.** Thêm method `updateStage(cvId, RecruitmentStage newStage)` — validate chỉ cho phép `ACCEPTED`, `REJECTED`.
- [x] **A6.** Expose endpoints trong `CandidateCVController`:
  - `GET /candidates` (unified filter)
  - `POST /candidates/{id}/schedule-interview`
  - `POST /candidates/{id}/reschedule-interview`
  - `POST /candidates/{id}/send-offer`
  - `PATCH /candidates/{id}/stage`

---

### Phase 3 Sprint B — Auto Stage Scheduler *(Ưu tiên 2)*
> **Mục tiêu:** Tự động chuyển `INTERVIEW_SCHEDULED → INTERVIEWED` sau khi qua ngày phỏng vấn.

- [x] **B1.** Thêm query vào `CandidateCVRepository`: tìm CV có `recruitmentStage = INTERVIEW_SCHEDULED` và `interviewSchedule <= now()`.
- [x] **B2.** Tạo `InterviewAutoTransitionScheduler.java` — `@Scheduled(fixedDelay = 900_000)` (15 phút):
  - Gọi query B1.
  - Batch-update `recruitmentStage = INTERVIEWED`.
  - Log số lượng record được chuyển.

---

### Phase 3 Sprint C — Position Filter & Toggle Active ✅
> **Mục tiêu:** Gộp các method filter position thành 1, thêm toggle active.

- [x] **C1.** Xóa `getPositions()`, `searchPositions()`, `getAllPositions()` khỏi `PositionService`. Thêm `filterPositions(keyword, isActive, page, size)` dùng JPQL dynamic (`:param IS NULL OR ...`).
- [x] **C2.** Thêm method `toggleActiveStatus(positionId)` vào `PositionService`:
  - `isActive = true` → set `false`, `closedAt = now()`.
  - `isActive = false` → set `true`, `closedAt = null`, `openedAt = now()`.
- [x] **C3.** Cập nhật `PositionController`:
  - `GET /positions` (thay thế 3 endpoints cũ)
  - `PATCH /positions/{id}/toggle-active`

---

### Phase 3 Sprint D — MinimumFitScore Sync & Guardrail ✅
> **Mục tiêu:** Score ngưỡng được HR điều chỉnh real-time, đồng bộ sang Chatbot, thay thế guard hardcode.

- [x] **D1.** Thêm `updateMinimumFitScore(positionId, score)` vào `PositionService` — update MySQL + fire-and-forget gọi chatbot.
- [x] **D2.** Expose `PATCH /positions/{id}/minimum-fit-score` trong `PositionController`.
- [x] **D3.** Expose `GET /internal/chatbot/positions/scores` trong recruitment — trả về `Map<Integer, Double>` (positionId → score) để chatbot preload.
- [x] **D4.** Trong `chatbot-service`: tạo `position_score_cache.py` (singleton `dict`), expose `PATCH /internal/positions/{id}/minimum-fit-score`.
- [x] **D5.** Thêm startup hook trong chatbot: gọi `GET /internal/chatbot/positions/scores` từ recruitment để preload cache sau khi service start.
- [x] **D6.** Refactor `finalize_application` trong `candidate_tools.py` + `reasoning.py`:
  - Xóa `_APPLY_ALLOWED_STATUSES`.
  - Guard mới: `avg_score = (technical_score + experience_score) / 2 >= position_score_cache.get(position_id, 70)`.
