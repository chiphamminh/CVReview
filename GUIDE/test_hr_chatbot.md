# Kịch bản Kiểm thử Tích hợp (Integration Test Flow) — HR Chatbot

Tài liệu này cung cấp các luồng kịch bản (Test Cases) chi tiết để bạn kiểm tra độ chính xác của **Intent Routing**, **Hybrid Retrieval** và **State Management** trên 2 chế độ (Mode) khác nhau của HR Chatbot.

---

## 🛠 1. Chuẩn bị (Pre-requisites)
- Đảm bảo hệ thống backend đang chạy (đã thấy bạn bật `chatbot-service` và `embedding-service`).
- Đảm bảo đã chọn một **Position (Vị trí tuyển dụng)** cụ thể trên giao diện FrontEnd.
- Đảm bảo Qdrant và Database có sẵn dữ liệu cho vị trí đó:
  - Vài CV do HR tự upload (để test `HR_MODE`).
  - Vài CV do ứng viên chủ động nộp (để test `CANDIDATE_MODE`).

---

## 💼 2. Test Flow 1: HR_MODE (Nguồn CV do HR Upload)
*Lưu ý: Bạn phải chạy tuần tự từ trên xuống dưới để test khả năng ghi nhớ ngữ cảnh (State) của Chatbot.*

### Khúc 1: AGGREGATE (Kiểm tra Bypass Qdrant, trả lời siêu nhanh)
*Mục đích: Đảm bảo Bot không search vector vô ích, chỉ gọi `get_cv_summary`.*
1. **Q:** *"Vị trí này hiện tại có tổng cộng bao nhiêu CV đã được upload?"*
2. **Q:** *"Thống kê nhanh cho tôi số lượng ứng viên pass và fail theo tiêu chuẩn AI."*

### Khúc 2: RANK / FILTER (Kiểm tra Query Expansion & Hybrid Retrieval)
*Mục đích: Đảm bảo Bot hiểu từ đồng nghĩa (expansion) và trả về top CV chính xác.*
1. **Q:** *"Tìm cho tôi top 3 ứng viên xuất sắc nhất phù hợp với yêu cầu của JD."*
2. **Q:** *"Trong số này, có ứng viên nào mạnh về Frontend như ReactJS hoặc Vue không?"*
3. **Q:** *"Lọc ra những ứng viên có điểm số AI đánh giá trên 80 điểm."*

### Khúc 3: DETAIL (Kiểm tra Virtual Full CV & Pinned Fetch)
*Mục đích: Đảm bảo Bot không search lại mà dùng chính `active_cv_ids` từ bước trên, nối các section lại để đọc.*
1. **Q:** *"Cho tôi xem thông tin chi tiết về kinh nghiệm làm việc của ứng viên đứng đầu."* (Hoặc gọi tên cụ thể).
2. **Q:** *"Bạn đánh giá sao về kỹ năng mềm và học vấn của [Tên ứng viên]?"*

### Khúc 4: COMPARE (Kiểm tra đối chiếu nhiều CV)
*Mục đích: Đảm bảo Bot gom đủ context của nhiều người và so sánh.*
1. **Q:** *"Hãy so sánh chi tiết điểm mạnh và điểm yếu giữa ứng viên [Tên A] và [Tên B]."*
2. **Q:** *"Giữa 2 bạn này, ai có kinh nghiệm làm dự án thực tế sát với yêu cầu JD hơn?"*

### Khúc 5: FIND_MORE (Kiểm tra Exclude Filter)
*Mục đích: Đảm bảo Bot query Qdrant nhưng loại trừ (must_not) các CV đã hiển thị.*
1. **Q:** *"Ngoài những bạn trên, hệ thống còn ứng viên nào khác có kỹ năng tương tự không?"*
2. **Q:** *"Tìm thêm cho tôi 2 ứng viên khác để tôi xem thêm."*

### Khúc 6: ACTION (Kiểm tra chặn gửi Email & Confirmation)
*Mục đích: Đảm bảo Bot KHÔNG gửi email bừa bãi mà phải hỏi ý kiến HR.*
1. **Q:** *"Hãy gửi email mời phỏng vấn cho [Tên ứng viên]."*
   - 👉 *Kỳ vọng:* Bot dừng lại, in ra màn hình xác nhận: *"Bạn có chắc muốn gửi email tới..."*
2. **Q:** *"Đồng ý, gửi đi."*
   - 👉 *Kỳ vọng:* Tool `send_interview_email` chính thức được gọi.

---

## 📩 3. Test Flow 2: CANDIDATE_MODE (Nguồn Inbound Apply)
*Lưu ý: Chuyển sang chế độ Candidate Mode trên UI trước khi test.*

### Khúc 1: AGGREGATE
1. **Q:** *"Có bao nhiêu người đã chủ động nộp đơn vào vị trí này?"*
2. **Q:** *"Trong số các đơn nộp, bao nhiêu người bị đánh giá rớt (fail)?"*

### Khúc 2: RANK / FILTER
1. **Q:** *"Liệt kê danh sách các ứng viên nộp đơn có kinh nghiệm quản lý hoặc làm Leader."*
2. **Q:** *"Tìm những bạn sinh viên mới ra trường hoặc Junior nộp đơn vào vị trí này."*

### Khúc 3: ACTION
1. **Q:** *"Gửi thư từ chối (reject) cho những ứng viên bị đánh giá dưới 75 điểm."*
   - 👉 *Kỳ vọng:* Bot liệt kê danh sách những người < 75đ và hỏi bạn xác nhận trước khi gửi thư hàng loạt.
2. **Q:** *"Xác nhận gửi."*

---

## 🎯 4. Tiêu chí Đánh giá (Pass/Fail)
Sau khi test, bạn để ý các log ở Terminal của `chatbot-service` (Port 8085):
- Nhìn thấy log `[HR Retrieve] strategy=AGGREGATE → Bypass Qdrant` 👉 **PASS Bước 13**.
- Nhìn thấy log `[HR Retrieve] strategy=COMPARE → Pinned scroll fetch for 2 CV(s)` 👉 **PASS Bước 12**.
- Xem prompt gửi cho Gemini không bị dính đoạn `No CV data found...` khi gọi Intent AGGREGATE/ACTION 👉 **PASS Bước 14**.
