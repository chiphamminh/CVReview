# Kịch bản Kiểm thử Tích hợp (Integration Test Flow) — Candidate Chatbot

Tài liệu này cung cấp các luồng kịch bản (Test Cases) chi tiết để bạn kiểm tra **Intent Routing**, **Hybrid Retrieval** và quá trình **Apply Job** dành riêng cho luồng của Ứng viên (Candidate Chatbot).

---

## 🛠 1. Chuẩn bị (Pre-requisites)
- Đảm bảo hệ thống backend đang chạy bình thường.
- Đăng nhập vào giao diện UI dưới quyền một **Ứng viên (Candidate)**.
- Ứng viên này cần có sẵn một **CV đã upload** lên hệ thống (để test `CV_ANALYSIS` và `JD_SEARCH`).
- Trên hệ thống cần có vài **Vị trí tuyển dụng (Job Positions)** đang mở (đã được parse JD lên Qdrant).

---

## 🧑‍💻 2. Kịch bản Test Liên tục (Candidate Workflow)
*Mô phỏng hành trình tự nhiên của một ứng viên: Vào đọc CV của mình -> Nhờ tìm việc phù hợp -> Hỏi chi tiết việc -> Nộp đơn -> Kiểm tra trạng thái.*

### Khúc 1: CV_ANALYSIS (Phân tích CV cá nhân)
*Mục đích: Đảm bảo Bot CHỈ fetch dữ liệu từ CV của chính ứng viên đang đăng nhập, không lẫn dữ liệu người khác.*
1. **Q:** *"Dựa vào CV tôi đã tải lên, hãy tóm tắt những kỹ năng công nghệ mạnh nhất của tôi."*
2. **Q:** *"Kinh nghiệm làm việc của tôi hiện tại phù hợp với định hướng Frontend hay Backend hơn?"*
3. **Q:** *"CV của tôi có điểm yếu nào cần cải thiện không?"*

### Khúc 2: JD_SEARCH (Tìm việc phù hợp - Hybrid Retrieval)
*Mục đích: Test luồng Full Pipeline (Expansion -> Hybrid Search -> Rerank) để match CV của ứng viên với tập JD.*
1. **Q:** *"Với kỹ năng hiện tại của tôi, hệ thống đang có những công việc nào phù hợp để ứng tuyển không?"*
2. **Q:** *"Tìm cho tôi công việc liên quan đến Spring Boot hoặc Java Backend, yêu cầu dưới 2 năm kinh nghiệm."*
   - 👉 *Kỳ vọng:* Query Expansion sẽ kích hoạt để bắt các cụm từ đồng nghĩa, kết quả trả về JD khớp nhất và Bot phân tích tại sao lại hợp.
3. **Q:** *"Bạn đánh giá mức độ phù hợp của tôi với vị trí [Tên một Job vừa trả về] là bao nhiêu %?"*

### Khúc 3: JD_ANALYSIS & JD_CONVERSE (Hỏi đáp chuyên sâu về JD)
*Mục đích: Đảm bảo Bot chuyển intent sang JD-only, đọc thông tin đặc tả từ file JD của công ty.*
1. **Q:** *"Vị trí [Tên Job] này yêu cầu những trách nhiệm công việc cụ thể là gì?"* (Intent `JD_ANALYSIS`)
2. **Q:** *"Công việc này có cho phép làm Remote không? Chế độ phúc lợi như thế nào?"* (Intent `JD_CONVERSE`)
3. **Q:** *"Mức lương khởi điểm cho vị trí này khoảng bao nhiêu?"* (Intent `JD_CONVERSE`)

### Khúc 4: APPLY (Nộp đơn ứng tuyển - Tool Calling & Confirmation)
*Mục đích: Đảm bảo Bot không tự ý apply mà phải chờ xác nhận từ user. Kiểm tra cơ chế Tool gọi xuống API.*
1. **Q:** *"Tôi muốn nộp đơn ứng tuyển vào vị trí [Tên Job]."*
   - 👉 *Kỳ vọng:* Bot dừng lại, tổng hợp thông tin (vị trí nào, dùng CV nào) và hỏi: *"Bạn có chắc chắn muốn nộp đơn vào vị trí này không?"*
2. **Q:** *"Tôi đồng ý, hãy nộp giúp tôi."*
   - 👉 *Kỳ vọng:* Bot gọi tool `finalize_application` và báo ứng tuyển thành công.

### Khúc 5: STATUS_CHECK (Tra cứu trạng thái - Bypass Qdrant)
*Mục đích: Đảm bảo Bot không tốn tiền gọi Qdrant mà lấy thẳng kết quả từ Database SQL.*
1. **Q:** *"Cho tôi biết trạng thái của tất cả các đơn ứng tuyển tôi đã nộp."*
2. **Q:** *"Đơn ứng tuyển của tôi vào vị trí [Tên Job] đã được HR xem hay chấm điểm chưa?"*

---

## 🎯 3. Tiêu chí Đánh giá (Pass/Fail trên Terminal)
Khi chạy luồng này, hãy mở song song terminal của `chatbot-service` (Port 8085) để quan sát log routing:
- Khi hỏi Khúc 1, log phải báo `strategy=CV_ANALYSIS` (Chỉ load CV chunks).
- Khi hỏi Khúc 2, log phải báo chạy qua `query_expansion` và `strategy=JD_SEARCH`.
- Khi hỏi Khúc 4, Bot phải gọi được vòng lặp confirmation trước khi log báo thực thi tool apply.
- Khi hỏi Khúc 5, log báo `strategy=STATUS_CHECK → Bypass Qdrant` 👉 **PASS**.
