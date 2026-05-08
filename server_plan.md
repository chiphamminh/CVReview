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
- [ ] **Update Position:**
  - *Analysis:* **NÊN GIỮ method update metadata, nhưng BỎ chức năng update file JD**. Việc thay đổi file JD gốc sẽ làm sai lệch tiêu chí đánh giá của các Candidate đã apply trước đó. Nếu cần đổi file JD, HR nên đóng Job đó và tạo Job mới. Các field metadata như `title`, `seniority`, `skills`, `minimum_fit_score` có thể cho phép sửa đổi linh hoạt.
- [ ] **Update Candidate CV:**
  - *Analysis:* **ĐỒNG Ý, nên bỏ chức năng update file CV**. CV nộp vào là một "snapshot". Không thể để ứng viên hay HR đổi file CV giữa chừng trong quá trình tuyển dụng. Chỉ nên giữ lại API update để sửa các field như `name`, `email` trong trường hợp hệ thống parse sai thông tin từ CV.

### 2. New Methods in Recruitment Service

#### Role: HR
- [ ] Send Mail Actions:
  - [ ] Offer mail
  - [ ] Interview schedule mail
  - [ ] Pre-schedule mail
- [ ] Update `recruitment_stage` cho Candidate.
- [ ] Update `minimum_fit_score` cho Position:
  - [ ] Gửi request sang Chatbot Service để sync mức điểm tối thiểu (dùng cho việc filter Candidate hoặc chặn apply).
- [ ] Update Searching/Filtering:
  - [ ] Sửa lại logic filter CV, Position theo các field mới (`title`, `seniority`, `recruitment_stage`, v.v.).

#### Role: Candidate
- [ ] Apply logic.

#### Notification
- [ ] System Notification cho user (Lưu DB, fetch qua HTTP bình thường, không cần WebSocket realtime).

---

## III. Execution Steps
- [x] 1. Sửa Entities (`Positions`, `CandidateCV`, `CVAnalysis`).
- [x] 2. Cập nhật các Enums liên quan (như `RecruitmentStage`, `SourceType`).
- [x] 3. Chạy DB Migration (hoặc update schema nếu dùng Hibernate auto-ddl).
- [x] 4. Sửa các Repositories đang query dựa trên cột cũ.
- [x] 5. Refactor `PositionService` (chỉnh sửa method update JD, logic filter, sync `minimum_fit_score`).
- [ ] 6. Refactor CV/Analysis Services (xóa logic update file, thêm logic update info cơ bản).
- [ ] 7. Code các service chức năng gửi Mail & Notification.
