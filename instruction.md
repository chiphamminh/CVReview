# CV Review System - System Instruction

> **Tài liệu hướng dẫn và mô tả kiến trúc hệ thống CV Review**
> Nguồn tham chiếu cho đội ngũ phát triển, AI Agent và các bên liên quan.

---

## 1. Tổng quan dự án (Project Overview)
**CV Review System** là một nền tảng tự động hóa tuyển dụng tích hợp AI, được thiết kế nhằm mục đích kết nối ứng viên và doanh nghiệp một cách thông minh, tối ưu hóa quy trình sàng lọc và tuyển dụng.

Hệ thống bao gồm hai giao diện/không gian chính:
- **Hệ thống nội bộ (Internal Portal):** Dành cho doanh nghiệp quản lý quy trình tuyển dụng.
- **Trang web ứng viên (Candidate Portal):** Dành cho ứng viên tìm kiếm việc làm và nộp hồ sơ.

### Phân quyền và Vai trò (Roles)
- **Role HR (Nhân sự):**
  - Quản lý vị trí tuyển dụng (CRUD positions, quản lý đợt tuyển).
  - Upload JD (Job Description) và CV của ứng viên (để lưu trữ và phân tích).
  - Cung cấp **Dashboard tổng hợp** về điểm số (score) và phân tích Candidate pool analysis.
  - Tương tác với hệ thống qua **HR Chatbot** (hỗ trợ tìm kiếm/sàng lọc ứng viên, gửi email phỏng vấn, và tự động tạo ra **bộ câu hỏi phỏng vấn** dựa trên kết quả matching giữa CV và JD).
- **Role Candidate (Ứng viên):**
  - Upload CV cá nhân để hệ thống lưu trữ và phân tích.
  - Sử dụng **Candidate Chatbot** để tìm kiếm vị trí phù hợp dựa trên năng lực.
  - Nhận tư vấn từ AI về độ phù hợp (Skill matching) và trực tiếp nộp đơn ứng tuyển qua chat.
- **Role Admin (Quản trị viên):**
  - Quản lý toàn bộ hệ thống (người dùng, quyền truy cập, v.v.).
  - **Quản lý cấu hình AI:** Thay đổi và quản lý Prompt cho Chatbot, cấu hình các model LLM sử dụng, và quản lý logs hệ thống để debug/theo dõi hiệu suất.

---

## 2. Tech Stack (Công nghệ sử dụng)
Hệ thống áp dụng kiến trúc **Microservices** bất đồng bộ nhằm đảm bảo tính mở rộng và khả năng chịu lỗi.

### Frontend
- **ReactJS:** Xây dựng giao diện cho cả Internal Portal (HR, Admin) và Candidate Portal.

### Backend & Core Services
- **Java 21 & Spring Boot:** Xây dựng các service lõi (`recruitment-service`, `auth-service`, `api-gateway`). Sử dụng **Virtual Threads** để tối ưu hóa hiệu suất xử lý I/O (ví dụ: long-polling với dịch vụ parse file).
- **Python 3 & FastAPI:** Đảm nhiệm các tác vụ liên quan đến AI (`chatbot-service`, `embedding-api`).

### AI & Data Processing
- **RAG (Retrieval-Augmented Generation):** Pipeline kết hợp LLM và Vector Database để trả lời câu hỏi và phân tích ngữ cảnh.
- **LLM Models:** Gemini 2.5 Flash / Gemini-1.5-Flash (Scoring, Chat reasoning), Llama-3.3-70b via Groq (Fast reasoning).
- **Embedding Model:** BAAI/bge-small-en-v1.5 (384 dimensions) xử lý vector hóa văn bản (JD & CV).
- **LangGraph:** Quản lý state và flow suy luận phức tạp của Chatbot.
- **LlamaParse API:** Trích xuất và parse dữ liệu từ file PDF/Word sang định dạng Markdown.

### Databases & Message Broker
- **Relational DB:** MySQL (truy xuất qua Hibernate/JPA).
- **Vector DB:** Qdrant (lưu trữ và tìm kiếm vector nhúng để tính độ tương đồng nhanh chóng).
- **Message Broker:** RabbitMQ xử lý luồng sự kiện bất đồng bộ (Parsing -> Embedding -> AI Scoring) với DLQ (Dead Letter Queue) và Retry mechanisms.

### Storage & Real-time
- **File Storage:** Google Drive, Cloudflare R2, Local Storage.
- **Real-time Comms:** Server-Sent Events (SSE) để stream kết quả AI trả về cho client.

---

## 3. Quy tắc thiết kế (Design Rules)
1. **Kiến trúc Microservices độc lập:** Các domain được tách biệt rõ ràng (`api-gateway`, `auth`, `recruitment`, `embedding`, `chatbot`). Giao tiếp chủ yếu qua REST API cho tác vụ đồng bộ và RabbitMQ cho tác vụ bất đồng bộ.
2. **Shared Context qua Common Library:** Tất cả DTOs, Exception Handlers toàn cục, Security Configs và hằng số (Constants) phải được đặt trong module `common-library` để tránh lặp code (D.R.Y).
3. **Tối ưu hóa Pipeline phân tích bằng 2 giai đoạn (Two-Stage Pipeline):** 
   - Thay vì gửi hàng loạt CV/JD vào LLM gây quá tải và chi phí cao, hệ thống sử dụng quy trình 2 bước:
     - **Bước 1 - Retrieval (Lọc thô - Chi phí thấp):** Dùng Qdrant Vector Search (Cosine Similarity) kết hợp Metadata Filtering để lọc ra Top 5-10 ứng viên/JDs phù hợp nhất.
     - **Bước 2 - Reranking / Scoring (Chấm điểm chi tiết - LLM):** Chỉ gửi Top đã lọc từ Bước 1 cùng prompt cho Gemini để thực hiện chấm điểm đa chiều (Technical, Experience) và tạo feedback.
4. **Mô hình Dữ liệu Bất biến cho CV (Single Source of Truth):**
   - Ứng viên chỉ có **đúng 1 Master CV** (không gắn với Position nào). Master CV này mới được vectorize và lưu vào Qdrant cùng với Metadata được extract 1 lần duy nhất lúc upload.
   - Khi ứng tuyển, hệ thống tạo ra **Application CV** kế thừa từ Master CV. Application CV **không** được embed lại vào Qdrant để tiết kiệm chi phí và tránh trùng lặp dữ liệu vector.
5. **Phân tách Trách nhiệm Service (Bounded Context):** 
   - `recruitment-service`: Chịu trách nhiệm quản lý dữ liệu (MySQL), extract metadata từ CV (thông qua GeminiExtractionService) và quản lý đợt tuyển dụng.
   - `chatbot-service`: Chịu trách nhiệm duy nhất cho các luồng tương tác AI, bao gồm Retrieval (tương tác với Qdrant) và Reranking/Scoring (gọi Gemini) khi người dùng (HR/Candidate) đưa ra yêu cầu.
   - *(Note: Nhờ kiến trúc Two-Stage Pipeline, việc chấm điểm batch toàn bộ Pool ứng viên không còn cần thiết, giúp loại bỏ hoàn toàn `ai-service` cũ).*
6. **Phân tách State của Chatbot:** 
   - Lịch sử chat (Chat history) và Session được quản lý và lưu ở MySQL (recruitment-service). AI LLM chỉ nhận tối đa 20 lượt chat gần nhất (Sliding Window) làm context để tránh quá tải token.
7. **Atomic Updates:** Mỗi pull request hoặc code commit chỉ nên tập trung vào một nhiệm vụ hoặc trách nhiệm duy nhất (Single Responsibility Principle).

---

## 4. Quy tắc bắt buộc (Mandatory Rules)

### Quy tắc về Code (Engineering Excellence)
1. **Developer-Centric Commenting:** Chỉ viết comment để giải thích "TẠI SAO" (Why) cho những logic phức tạp hoặc rule nghiệp vụ. Tuyệt đối không dùng comment kiểu AI như "Updated logic for X", "Fixed this line". Sử dụng Javadoc/Docstring chuẩn mực.
2. **Radical Codebase Cleanup (Dọn dẹp code rác):** Ngay sau khi cập nhật logic mới, phải xóa bỏ hoàn toàn code cũ (phương thức, biến, class không còn dùng). Tuyệt đối không để lại "zombie code" (code bị comment out).
3. **Robust Error Handling:** Mọi logic quan trọng phải được bắt lỗi (try-catch) và ném ra các **Custom Exceptions** cụ thể thay vì dùng Exception chung chung.
4. **Strict Typing:** Đảm bảo an toàn kiểu dữ liệu. Phải định nghĩa rõ kiểu trả về và tham số truyền vào, tránh dùng kiểu động (`any`) ở mức tối đa có thể.

### Quy tắc về Hệ thống & Nghiệp vụ (System Guardrails)
5. **Guardrail Nộp Đơn (Application Guardrail):** Function `finalize_application` trong Candidate Chatbot **KHÔNG BAO GIỜ** được phép thực thi nếu điểm phù hợp (score) **dưới 70 điểm**. Chatbot bắt buộc phải từ chối, giải thích thiếu sót kỹ năng và gợi ý lộ trình học tập.
6. **Bảo mật Internal APIs:** Tất cả các endpoints có tiền tố `/internal/*` bắt buộc không được lộ ra ngoài qua API Gateway. Các endpoint này chỉ giao tiếp nội bộ trong mạng Docker và phải được bảo vệ bằng header xác thực nội bộ (ví dụ: `X-Internal-Service`).
7. **Cập nhật CV:** Khi Candidate upload CV mới, tất cả `Application CVs` và `CVAnalysis` cũ đang ứng tuyển đều phải được **Soft-Delete**. Ứng viên phải nộp lại đơn ứng tuyển với CV mới.

---

## 5. Folder Structure (Cấu trúc thư mục)

> Tổng quan cấu trúc thư mục toàn bộ dự án — cả FrontEnd (React/Vite) và BackEnd (Java Spring Boot + Python FastAPI Microservices).

```
CVReview/
├── FrontEnd/                          # React + Vite SPA (HR Portal & Candidate Portal)
│   └── src/
│       ├── api/
│       │   ├── axiosClient.js         # Axios instance với interceptor JWT
│       │   └── mockData.js            # Mock data cho dev/test
│       ├── assets/                    # Static assets (icons, images)
│       ├── components/
│       │   ├── candidate/             # Components dành riêng cho Candidate Portal
│       │   ├── chatbot/               # Chatbot Drawer UI (HR & Candidate)
│       │   ├── common/                # Shared components (Button, Badge, Spinner…)
│       │   ├── modals/                # Modal dialogs (Upload, SendOffer, Confirm…)
│       │   ├── tables/                # Data table components
│       │   └── upload/                # Upload CV/JD components + progress tracking
│       ├── hooks/                     # Custom React hooks
│       ├── layouts/
│       │   ├── HRLayout.jsx           # Layout chung cho HR Portal (sidebar, navbar)
│       │   └── CandidateLayout.jsx    # Layout chung cho Candidate Portal
│       ├── pages/
│       │   ├── Login.jsx              # Trang đăng nhập chung
│       │   ├── hr/
│       │   │   ├── PositionsPage.jsx  # Quản lý vị trí tuyển dụng
│       │   │   ├── CandidatesPage.jsx # Danh sách & sàng lọc ứng viên
│       │   │   └── HRChatbotPage.jsx  # Giao diện HR Chatbot
│       │   └── candidate/
│       │       ├── CareerPage.jsx     # Trang tìm kiếm vị trí (Candidate)
│       │       └── CVPage.jsx         # Trang quản lý Master CV
│       ├── routes/                    # React Router config + ProtectedRoute
│       ├── store/
│       │   └── authStore.js           # Zustand store quản lý auth state
│       ├── types/                     # JSDoc type definitions
│       ├── utils/                     # Helper functions (format, validate…)
│       ├── App.jsx
│       └── main.jsx
│
└── BackEnd/                           # Microservices Architecture
    ├── docker-compose.yml             # Orchestration toàn bộ services
    ├── docker-compose-build.yml       # Build & push images lên registry
    │
    ├── common-library/                # [Spring Boot] Shared module dùng chung
    │   └── src/main/java/.../commonlibrary/
    │       ├── dto/                   # Shared DTOs & event payloads (RabbitMQ)
    │       ├── exception/             # Global exception handlers & custom exceptions
    │       └── utils/                 # Shared utility classes
    │
    ├── api-gateway/                   # [Spring Cloud Gateway] Entry point duy nhất
    │   └── src/main/java/.../apigateway/
    │       ├── config/                # Route config, CORS, filter chains
    │       ├── controller/            # Health check endpoints
    │       └── security/              # JWT validation filter
    │
    ├── auth-service/                  # [Spring Boot] Xác thực & phân quyền
    │   └── src/main/java/.../authservice/
    │       ├── config/                # Security config, Bean config
    │       ├── controller/            # Auth endpoints (login, register, refresh)
    │       ├── dto/                   # Request/Response DTOs
    │       ├── models/                # User, Role entities
    │       ├── repository/            # JPA Repositories
    │       ├── security/              # JWT provider, UserDetailsService
    │       ├── services/              # Business logic
    │       └── utils/                 # Helper utilities
    │
    ├── recruitment-service/           # [Spring Boot] Service lõi - Quản lý nghiệp vụ tuyển dụng
    │   └── src/main/java/.../recruitmentservice/
    │       ├── client/                # Feign clients gọi sang embedding-service
    │       ├── config/                # RabbitMQ, JPA, SSE, async config
    │       ├── controller/            # REST Controllers:
    │       │   ├── PositionController            # CRUD vị trí tuyển dụng
    │       │   ├── CandidateCVController         # Upload & quản lý Master CV
    │       │   ├── UploadCVController            # Batch upload CV (HR)
    │       │   ├── ProcessingBatchController     # Quản lý batch processing
    │       │   ├── ChunkingController            # Chunking & indexing pipeline
    │       │   ├── AnalysisController            # CV Analysis results
    │       │   ├── ChatbotInternalController     # Internal APIs cho chatbot-service
    │       │   ├── ChatbotPublicController       # Public chatbot session APIs
    │       │   ├── AdminCVController             # Admin CV management
    │       │   └── AdminAnalyticsController      # Dashboard analytics
    │       ├── dto/                   # Request/Response DTOs & Event payloads
    │       ├── exception/             # Domain-specific custom exceptions
    │       ├── listener/              # RabbitMQ event listeners (parse, embed, score)
    │       ├── models/                # JPA Entities (Position, CV, Application, ChatSession…)
    │       ├── repository/            # JPA Repositories
    │       ├── scheduler/             # Scheduled jobs (cleanup, retry…)
    │       ├── security/              # Internal API auth filter (X-Internal-Service)
    │       ├── services/              # Business logic services
    │       ├── sse/                   # Server-Sent Events emitter (streaming progress)
    │       └── utils/                 # Helpers (file, date, string…)
    │
    ├── embedding-service/             # [Python FastAPI] Vector hóa CV & JD → Qdrant
    │   └── app/
    │       ├── main.py                # FastAPI app entry point
    │       ├── config.py              # Env config (Qdrant, RabbitMQ, model)
    │       ├── routers/
    │       │   ├── cv.py              # Endpoints nhận CV chunks để embed
    │       │   └── jd.py              # Endpoints nhận JD chunks để embed
    │       ├── services/
    │       │   ├── embedding.py       # BAAI/bge embedding model wrapper
    │       │   ├── qdrant.py          # Qdrant upsert / delete operations
    │       │   ├── rabbitmq/          # RabbitMQ consumers (CV & JD queues)
    │       │   └── http_client.py     # HTTP client gọi recruitment-service
    │       ├── models/                # Pydantic schemas
    │       ├── worker_cv.py           # Entry point worker consume CV queue
    │       └── worker_jd.py           # Entry point worker consume JD queue
    │
    └── chatbot-service/               # [Python FastAPI + LangGraph] RAG Chatbot Engine
        └── app/
            ├── main.py                # FastAPI app entry point
            ├── config.py              # Env config (Qdrant, Gemini, Groq, RabbitMQ)
            ├── api/routes/
            │   ├── chat.py            # Shared chat endpoint (session-based)
            │   ├── hr_chat.py         # HR-specific chat routing
            │   ├── candidate_chat.py  # Candidate-specific chat routing
            │   └── health.py          # Health check endpoint
            ├── models/                # Pydantic schemas (ChatRequest, ChatResponse…)
            ├── services/
            │   ├── embedding.py       # Embedding wrapper (query vectorization)
            │   ├── qdrant.py          # Qdrant search & filter operations
            │   ├── retriever.py       # Two-Stage Pipeline: Vector Search → Rerank
            │   ├── reranker.py        # LLM-based reranking & scoring
            │   └── recruitment_api.py # HTTP client → recruitment-service internal APIs
            └── rag/
                ├── prompts.py         # Tất cả LLM prompts (HR & Candidate)
                ├── intent.py          # Intent classification logic
                ├── shared/            # Shared RAG utilities & context formatters
                ├── hr/
                │   ├── hr_graph.py    # LangGraph graph definition cho HR flow
                │   ├── hr_tools.py    # Tool functions (search, email, interview Qs)
                │   ├── router.py      # Intent router → node dispatcher
                │   ├── state.py       # HR graph state schema
                │   └── nodes/         # LangGraph node implementations
                └── candidate/
                    ├── candidate_graph.py  # LangGraph graph cho Candidate flow
                    ├── candidate_tools.py  # Tool functions (search, apply)
                    ├── router.py           # Intent router → node dispatcher
                    ├── state.py            # Candidate graph state schema
                    └── nodes/              # LangGraph node implementations
```
