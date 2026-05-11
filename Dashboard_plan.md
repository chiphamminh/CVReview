# HR Dashboard — Bản Thiết Kế Hoàn Chỉnh

> Tác giả: AI Design Review | Ngày: 2026-05-11  
> Phạm vi: HR Dashboard (`/hr/dashboard`) — Phân tích & trực quan hóa dữ liệu tuyển dụng

---

## 1. Phân tích Hiện trạng Hệ thống

### Backend đã có sẵn
| Endpoint | Dữ liệu trả về | Trạng thái |
|----------|----------------|------------|
| `GET /admin/analytics/cv-traffic?days=30` | `totalCv`, `successCv`, `failedCv`, `processingCv` | ✅ Sẵn sàng |
| `GET /admin/analytics/processing-time?days=30` | Buckets thời gian xử lý batch theo quy mô | ✅ Sẵn sàng |
| `GET /internal/chatbot/positions/{id}/cv-statistics` | `total`, `scored`, `passed`, `failed` per position | ✅ Sẵn sàng (internal) |

### Backend cần bổ sung
| Endpoint mới | Mục đích | Độ ưu tiên |
|---|---|---|
| `GET /admin/analytics/overview` | KPI tổng hợp toàn hệ thống | 🔴 Cao |
| `GET /admin/analytics/score-distribution` | Phân phối điểm số theo dải | 🔴 Cao |

### Frontend hiện tại
- Dashboard là placeholder `<div>HR Dashboard</div>` tại `AppRoutes.jsx:19`
- **Chưa có** chart library nào được cài đặt
- **Chưa có** `analytics.api.js` wrapper

---

## 2. Quyết định Kiến trúc

### 2.1 Real-time vs. Manual Refresh

**Quyết định: Không dùng real-time (SSE/WebSocket). Dùng Auto-refresh + Manual refresh button.**

**Lý do:**
- Dashboard analytics là dữ liệu lịch sử tổng hợp — không thay đổi từng giây như upload progress.
- CV được upload theo batch, không phải liên tục. Refresh mỗi 60 giây là đủ.
- SSE đã được dùng cho upload progress tracking — dùng thêm cho dashboard sẽ tốn resource không cần thiết.
- Độ phức tạp tăng không tương xứng với giá trị thực tế đem lại.

**Giải pháp triển khai:**
```
React Query (TanStack Query) với:
  - staleTime: 60_000ms (1 phút)
  - refetchInterval: 60_000ms (auto-refresh ngầm mỗi 1 phút)
  - Manual refresh: nút "Refresh" trigger invalidateQueries
```

### 2.2 Chart Library

**Quyết định: Dùng `@ant-design/plots` (G2Plot wrapper)**

**Lý do:**
- Project đang dùng Ant Design 6.x — nhất quán về style, không cần custom theme.
- Hỗ trợ **Funnel Chart** natively — recharts không có.
- Hỗ trợ Column, Histogram, Pie, và tất cả chart types cần dùng.
- Bundle size chấp nhận được cho SPA.

---

## 3. Thiết kế Dashboard — Layout & Widgets

### 3.1 Layout Tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│  HR Dashboard          [Time Filter: 7D | 30D | 90D] [🔄 Refresh]│
├──────────┬──────────┬──────────┬──────────────────────────────────┤
│ KPI Card │ KPI Card │ KPI Card │           KPI Card               │
│ Total CVs│ Avg Score│Time Saved│        Success Rate              │
├──────────┴──────────┴──────────┴──────────────────────────────────┤
│                                   │                               │
│   Score Distribution              │   Processing Funnel           │
│   (Column Chart)                  │   (Funnel Chart)              │
│                                   │                               │
├───────────────────────────────────┴───────────────────────────────┤
│                  CV Status Breakdown (Donut Chart)                 │
│          (Embedded | Processing | Failed | Pending)               │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi tiết Từng Widget

### Widget 1 — KPI Cards (Row 1)

**Mục đích:** Con số tổng quan đập vào mắt ngay khi mở dashboard.

#### Card 1: Total CVs Processed
- **Metric:** Tổng CV đã qua AI pipeline thành công
- **Data source:** `/admin/analytics/cv-traffic` → `successCv`
- **Display:** Số lớn + so sánh với kỳ trước (% thay đổi) — cần `days` window
- **Icon:** `FileDoneOutlined` (xanh lá)

#### Card 2: Average Matching Score
- **Metric:** Điểm matching trung bình của toàn bộ CV đã được chấm điểm
- **Data source:** `/admin/analytics/overview` (endpoint mới) → `avgMatchingScore`
- **Display:** Số thập phân (VD: `73.4`) + thanh progress nhỏ
- **Icon:** `StarOutlined` (vàng)
- **Lưu ý:** Query từ bảng `CVAnalysis` → `AVG(score) WHERE score IS NOT NULL`

#### Card 3: Time Saved (Estimated)
- **Metric:** Số giờ HR đã tiết kiệm được nhờ AI
- **Công thức:** `ceil(successCv × 5 / 60)` giờ (5 phút/CV thủ công)
- **Data source:** Tính toán client-side từ `successCv`
- **Display:** Số giờ + note nhỏ "(~5 min/CV estimated)"
- **Icon:** `ClockCircleOutlined` (tím)
- **Lưu ý:** Đây là con số marketing, cần label rõ "Estimated" để tránh hiểu nhầm.

#### Card 4: Success Match Rate
- **Metric:** % CV đạt ngưỡng điểm ≥ 70 (cùng ngưỡng guardrail của hệ thống)
- **Data source:** `/admin/analytics/overview` → `successMatchRate`
- **Công thức backend:** `(count WHERE score >= 70) / (count WHERE score IS NOT NULL) × 100`
- **Display:** Phần trăm + progress circle
- **Icon:** `CheckCircleOutlined` (xanh dương)

---

### Widget 2 — Candidate Score Distribution (Row 2, Left)

**Mục đích:** Cho HR thấy chất lượng pool ứng viên đang phân bổ ở đâu — nhiều ứng viên trung bình hay xuất sắc?

**Chart type:** Column Chart (Bar đứng)

**Data source:** `GET /admin/analytics/score-distribution` (endpoint mới)

**Response structure cần thiết:**
```json
{
  "buckets": [
    { "range": "0–20",  "label": "Rất yếu",   "count": 12 },
    { "range": "21–40", "label": "Yếu",        "count": 34 },
    { "range": "41–60", "label": "Trung bình", "count": 87 },
    { "range": "61–80", "label": "Khá",        "count": 156 },
    { "range": "81–100","label": "Xuất sắc",   "count": 43 }
  ]
}
```

**Visual spec:**
- Trục X: Dải điểm (5 cột)
- Trục Y: Số lượng CV
- Màu sắc gradient: đỏ → vàng → xanh lá (theo chất lượng)
- Đường kẻ ngang ở `score = 70` (ngưỡng pass) để HR thấy bao nhiêu CV "đủ điều kiện"
- Tooltip: hiện số lượng + % tổng khi hover

**Tại sao không dùng Histogram?**
Histogram phù hợp khi data liên tục dày đặc. Ở đây chỉ có 5 bucket cố định → Column Chart rõ hơn, dễ đọc hơn.

---

### Widget 3 — Processing Funnel (Row 2, Right)

**Mục đích:** Minh họa quy trình xử lý CV từ upload đến shortlist — thể hiện sự chuyên nghiệp của pipeline.

**Chart type:** Funnel Chart

**Data source:** Kết hợp `/admin/analytics/cv-traffic` + `/admin/analytics/overview`

**Các bước trong funnel:**
```
Stage 1: Uploaded     → totalCv (từ cv-traffic)
Stage 2: Parsed       → successCv (CV parse thành công, không bị failed)  
Stage 3: AI Scored    → scored (CV đã có điểm trong CVAnalysis)
Stage 4: Shortlisted  → passed (CV có score >= 70)
```

**Visual spec:**
- Funnel dọc, màu xanh gradient từ trên xuống
- Mỗi bước hiện: tên stage + số lượng + % so với bước trên (conversion rate)
- Tooltip: hiện số tuyệt đối + tỷ lệ drop-off

**Lưu ý quan trọng:**  
`scored` count cần được expose trong `/admin/analytics/overview` vì hiện tại chỉ có per-position statistics qua internal API.

---

### Widget 4 — CV Status Breakdown (Row 3)

**Mục đích:** Bức tranh nhanh về trạng thái xử lý của toàn bộ CV trong hệ thống.

**Chart type:** Donut Chart (Pie có lỗ giữa)

**Data source:** `/admin/analytics/cv-traffic`

**Segments:**
| Trạng thái | Màu | Data field |
|---|---|---|
| Processed (Success) | Xanh lá `#52c41a` | `successCv` |
| Processing | Xanh dương `#1677ff` | `processingCv` |
| Failed | Đỏ `#ff4d4f` | `failedCv` |

**Visual spec:**
- Legend bên phải với số lượng tuyệt đối
- Center label: tổng `totalCv`
- Không cần interaction phức tạp — static legend là đủ

---

## 5. Các Widget bị loại bỏ (và lý do)

### ❌ Processing Time Chart (từ `/admin/analytics/processing-time`)
**Lý do loại:** Endpoint này trả về thời gian xử lý batch theo quy mô (1-10 CV, 11-20 CV...). Đây là thông tin kỹ thuật/DevOps, không phải thông tin nghiệp vụ HR cần. HR không quan tâm batch 20 CV mất bao nhiêu giây. Endpoint này phù hợp hơn cho Admin Dashboard (nếu có).

### ❌ Real-time Live Feed
**Lý do loại:** Thêm complexity (WebSocket/SSE connection management) mà không đem lại giá trị thực tế cho HR. Dashboard analytics không phải monitoring tool.

---

## 6. Backend — Endpoints Cần Bổ Sung

### 6.1 `GET /admin/analytics/overview`

**Controller:** `AdminAnalyticsController.java`

**Response DTO:** `OverviewResponse.java`
```java
@Builder
public class OverviewResponse {
    private long totalCvsProcessed;   // CVAnalysis count (scored)
    private double avgMatchingScore;  // AVG(score) from CVAnalysis
    private double successMatchRate;  // % score >= 70
    private long totalPositions;      // Active positions count
    private int days;
}
```

**Service query logic:**
```java
// From CVAnalysis repository:
long scored = cvAnalysisRepo.countByScoreIsNotNull();
double avg = cvAnalysisRepo.averageScore();
long passed = cvAnalysisRepo.countByScoreGreaterThanEqual(70.0);
double rate = scored > 0 ? (passed * 100.0 / scored) : 0.0;
```

### 6.2 `GET /admin/analytics/score-distribution`

**Controller:** `AdminAnalyticsController.java`

**Response DTO:** `ScoreDistributionResponse.java`
```java
@Builder
public class ScoreDistributionResponse {
    private List<ScoreBucket> buckets;

    @Builder
    public static class ScoreBucket {
        private String range;   // "0-20"
        private String label;   // "Rất yếu"
        private long count;
    }
}
```

**Query approach:** 5 JPQL queries với BETWEEN clause trên `CVAnalysis.score`.

---

## 7. Frontend — Cấu trúc Files

```
FrontEnd/src/
├── api/
│   └── analytics.api.js          # NEW: Wrapper cho /admin/analytics/*
├── pages/hr/
│   └── HRDashboardPage.jsx        # NEW: Thay thế placeholder
├── components/
│   └── dashboard/                 # NEW: Folder chứa chart components
│       ├── KPICard.jsx            # Reusable card component
│       ├── ScoreDistributionChart.jsx
│       ├── ProcessingFunnelChart.jsx
│       └── CVStatusDonut.jsx
└── hooks/
    └── useAnalytics.js            # NEW: React Query hooks cho analytics data
```

---

## 8. Implementation Roadmap

### Phase 1 — Backend (Ưu tiên cao)
1. Tạo `CVAnalysisRepository` query methods: `averageScore()`, `countByScoreIsNotNull()`, `countByScoreGreaterThanEqual(double)`, `countByScoreRange(double, double)`
2. Tạo `OverviewAnalyticsService` aggregating data
3. Thêm 2 endpoints vào `AdminAnalyticsController`: `/overview` và `/score-distribution`
4. Thêm `scored` count vào `CvTrafficResponse` (hoặc expose qua `/overview`)

### Phase 2 — Frontend Setup
1. `npm install @ant-design/plots` trong `FrontEnd/`
2. Tạo `analytics.api.js` wrapper
3. Tạo `useAnalytics.js` custom hook với React Query

### Phase 3 — Dashboard UI
1. Tạo `HRDashboardPage.jsx` với layout grid (Ant Design `Row`/`Col`)
2. Build 4 `KPICard` components
3. Build `ScoreDistributionChart` (Column Chart)
4. Build `ProcessingFunnelChart` (Funnel Chart)
5. Build `CVStatusDonut` (Donut Chart)
6. Wire time filter (7D/30D/90D) với `days` param

### Phase 4 — Polish
1. Loading skeleton states cho mỗi card/chart
2. Empty state khi chưa có data
3. Error boundary nếu API fail
4. Responsive layout cho màn hình nhỏ hơn

---

## 9. Tổng kết Quyết định Thiết kế

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Real-time | ❌ Không | Data analytics không cần second-by-second update |
| Refresh strategy | ✅ Auto (60s) + Manual button | Balance giữa freshness và resource |
| Chart library | `@ant-design/plots` | Nhất quán với Ant Design, có Funnel Chart |
| Funnel steps | 4 bước | Upload → Parsed → Scored → Shortlisted |
| Pass threshold | 70 điểm | Nhất quán với guardrail của Candidate Chatbot |
| KPI Time frame | Theo `days` filter | Cho phép HR so sánh theo tuần/tháng/quý |
| Score buckets | 5 dải cố định | Đủ granular, dễ đọc hơn histogram liên tục |
