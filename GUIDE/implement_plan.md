# CV Review Chatbot — RAG Refactor & Implementation Plan

> **Mục đích file này:** Tài liệu kế hoạch triển khai chi tiết cho việc refactor và nâng cấp RAG pipeline.  
> **Dùng làm context carrier:** Khi mở conversation mới, paste toàn bộ file này + các file code liên quan vào đầu conversation để AI tiếp tục đúng chỗ.  
> **Cập nhật lần cuối:** Sprint 0 (Planning complete)

---

## 1. Bối cảnh & Vấn đề cần giải quyết

### Hệ thống hiện tại
- **CV Review System** với 2 chatbot: HR Chatbot và Candidate Chatbot
- Pipeline hiện tại: LangGraph workflow, Qdrant vector search, Gemini LLM, Cross-encoder reranker
- Tech stack: Python/FastAPI, LangGraph, Qdrant, Gemini Flash/Pro, BGE embedding, BGE reranker

### Issue #1 — Linear Graph không có Intent-Aware Routing
**Triệu chứng:** Mọi query đều đi qua toàn bộ pipeline kể cả các node không liên quan.  
**Ví dụ cụ thể:** HR hỏi "so sánh 2 ứng viên này" → vẫn chạy Qdrant retrieval, JD fetch — hoàn toàn thừa.  
**Root cause:** Graph là linear chain cứng nhắc, không có routing logic dựa trên intent.

### Issue #2 — Single-Stage Dense Vector Retrieval bỏ sót candidate
**Triệu chứng:** Lọc từ 100 CV → top 5-10 bị miss candidate tốt.  
**Ví dụ cụ thể:** CV viết "Spring Framework" bị miss khi HR tìm "Spring Boot" vì cosine score thấp hơn threshold.  
**Root cause:** Dense vector giỏi semantic nhưng kém exact skill matching. Không có keyword search bổ trợ.

---

## 2. Giải pháp tổng thể

### 2.1 Kiến trúc mới

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         Hard-rule Intent Router          │  ← THÊM MỚI (zero LLM cost)
│  Pattern matching + Entity extraction    │
└─────────────────────────────────────────┘
    │
    ├── ACTION / AGGREGATE / COMPARE / DETAIL
    │       │
    │       ▼ (bypass Qdrant hoàn toàn)
    │   Pipeline tương ứng → build_prompt → LLM → save → format
    │
    └── RANK / FILTER / FIND_MORE
            │
            ▼
    ┌───────────────────┐
    │  Query Expansion  │  ← THÊM MỚI (1 LLM Flash call)
    │  Synonym expand   │
    └───────────────────┘
            │
            ▼
    ┌──────────────────────────────┐
    │      Hybrid Retrieval         │  ← NÂNG CẤP
    │  Dense Search (async)         │
    │  + Keyword Search (async)     │
    │  → RRF Merge                  │
    └──────────────────────────────┘
            │
            ▼
    Cross-encoder Rerank (giữ nguyên)
            │
            ▼
    build_prompt → LLM → save → format
```

### 2.2 Nguyên tắc thiết kế
- Router là **pure function** — không side effects, không API call
- LLM Flash chỉ được gọi khi intent **cần Qdrant** (RANK, FILTER, FIND_MORE, JD_SEARCH)
- Hard-rule có **priority order** — match đầu tiên wins, không evaluate tiếp
- HR Router và Candidate Router **tách biệt hoàn toàn** — tránh conflict

---

## 3. Intent Taxonomy

### 3.1 HR Chatbot Intents

| Intent | Trigger Signal | Qdrant? | LLM Expansion? | Pipeline |
|--------|---------------|---------|----------------|----------|
| `ACTION` | gửi email, đồng ý, confirm, từ chối, offer | ✗ | ✗ | SQL + Tool |
| `AGGREGATE` | bao nhiêu, thống kê, tổng số, how many | ✗ | ✗ | SQL only |
| `COMPARE` | so sánh, compare, vs + active_cv_ids có sẵn | ✗ | ✗ | Pinned fetch |
| `DETAIL` | chi tiết, thông tin về, tell me about + active_cv_ids có sẵn | ✗ | ✗ | Pinned fetch |
| `FIND_MORE` | còn ai, thêm, khác, next, more | ✓ + exclude | ✓ | Hybrid + exclude_ids |
| `RANK` | top N, tìm, liệt kê, tốt nhất, phù hợp nhất | ✓ | ✓ | Hybrid full |
| `FILTER` | có skill X, điểm >= Y, lọc theo | ✓ | ✓ | SQL-first + Hybrid |

### 3.2 Candidate Chatbot Intents

| Intent | Trigger Signal | Qdrant? | LLM Expansion? | Pipeline |
|--------|---------------|---------|----------------|----------|
| `STATUS_CHECK` | đã apply chưa, trạng thái, đơn của tôi | ✗ | ✗ | SQL only |
| `APPLY` | nộp đơn, apply, ứng tuyển | ✗ | ✗ | Tool direct |
| `CV_ANALYSIS` | CV của tôi, kỹ năng của tôi, profile của tôi | ✓ CV only | ✗ | CV chunks |
| `JD_CONVERSE` | lương, benefits, quy trình, culture, remote | ✓ JD only | ✗ | JD chunks |
| `JD_SEARCH` | vị trí nào phù hợp, tìm việc, recommend | ✓ CV + JD | ✓ | Full pipeline |
| `JD_ANALYSIS` | vị trí này yêu cầu gì, phân tích JD | ✓ JD only | ✗ | JD chunks |

### 3.3 Router Priority Order (HR)

```
Priority 1 — Session state check (không cần pattern match):
  → pending_emails tồn tại + confirm phrase → ACTION(confirm_email)
  → active_cv_ids tồn tại + COMPARE pattern → COMPARE
  → active_cv_ids tồn tại + DETAIL pattern  → DETAIL

Priority 2 — Hard keyword patterns:
  → AGGREGATE patterns → AGGREGATE (skip Qdrant)
  → ACTION patterns    → ACTION (skip Qdrant)
  → FIND_MORE patterns → FIND_MORE (Qdrant + exclude)

Priority 3 — Default semantic:
  → RANK / FILTER      → Query Expansion → Hybrid Retrieval
```

---

## 4. Query Expansion (LLM Flash)

**Trigger:** Chỉ khi intent thuộc {RANK, FILTER, FIND_MORE, JD_SEARCH}

**Input:** query string + extracted skill_keywords từ Router

**Output (structured JSON):**
```json
{
  "expanded_query": "chuỗi đã expand với synonyms — dùng để embed dense vector",
  "skill_variants": ["Spring Boot", "Spring Framework", "Spring MVC", "..."]
}
```

**Hai output có vai trò khác nhau:**
- `expanded_query` → embed thành vector cho **dense search**
- `skill_variants` → dùng cho Qdrant `MatchAny` filter trên **`skills` metadata field**

**Fallback:** Nếu LLM call fail hoặc timeout > 2s:
- `expanded_query` = original query
- `skill_variants` = skill_keywords từ entity extraction
- Log warning, pipeline tiếp tục bình thường

---

## 5. Hybrid Retrieval

### 5.1 Flow

```
expanded_query
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
Dense Search (async)                  Keyword Search (async)
Qdrant cosine similarity              Qdrant MatchAny on `skills` field
embed(expanded_query)                 filter: skills MatchAny skill_variants
limit = top_n * 4                     limit = top_n * 3
score_threshold = adaptive            score_threshold = 0.0
      │                                        │
      └──────────────┬─────────────────────────┘
                     ▼
              RRF Merge
              weight: 0.6 dense + 0.4 keyword
                     │
                     ▼
            Cross-encoder Rerank
            group by cvId (Max Score)
            áp dụng max_chunks_per_id = 6 để tránh tràn prompt
            → top_n results
```

### 5.2 RRF Formula
```
RRF_score(doc) = Σ weight_i / (60 + rank_i)
```
Không cần normalize score giữa 2 hệ thống — đây là lý do chọn RRF thay vì weighted sum.

### 5.3 FIND_MORE Variant
Hybrid retrieval chạy bình thường nhưng thêm `must_not` vào cả 2 Qdrant queries:
```
must_not: [FieldCondition(key="cvId", match=MatchAny(any=active_cv_ids))]
```
Qdrant xử lý exclusion ở tầng index — không cần filter lại ở Python layer.

### 5.4 Prerequisite
- `skills` field đã tồn tại trong Qdrant payload (đã confirm) — không cần re-index
- Field name phải nhất quán giữa indexing code và query filter

---

## 6. Pipeline Nodes (Bypass Retrieval)

### COMPARE / DETAIL Pipeline
- Input: `active_cv_ids` từ session state
- Action: Pinned fetch trực tiếp bằng `cvId` filter, không rerank
- JD context: empty (không inject JD block vào prompt)
- Build prompt nhận biết intent → thêm instruction "Compare side by side"

### ACTION Pipeline
- Input: `sql_metadata` đã có trong state
- Action: bypass Qdrant hoàn toàn, `cv_context = []`, `jd_context = []`
- LLM chỉ nhận sql_metadata + query để call tool
- Email confirmation flow (pending_emails): Cần tạo `action_id` (UUID) khi lưu vào state, và check `action_id` trước khi send để tránh double-send race condition.

### AGGREGATE Pipeline
- Input: query + position_id
- Action: gọi `recruitment_api.get_cv_statistics()` trực tiếp
- Có thể skip LLM call nếu data đủ rõ ràng (chỉ cần format output)

### "Virtual Full CV" cho COMPARE/DETAIL
Vì Qdrant không có section "full CV summary", khi pinned fetch:
- Fetch **tất cả sections** của mỗi cvId (không cap chunk quá thấp)
- Python layer ghép theo thứ tự: SUMMARY → EXPERIENCE → SKILLS → EDUCATION → PROJECTS
- Kết quả là "virtual full CV" đủ để LLM đọc và compare

---

## 7. Cấu trúc thư mục sau refactor

```
app/rag/
    hr/
        state.py              # HRChatState TypedDict (tách từ hr_graph.py)
        router.py             # Hard-rule Intent Router (MỚI)
        nodes/
            session.py        # load_hr_session_history_node
            scope.py          # load_candidate_scope_node
            expansion.py      # query_expansion_node (MỚI)
            retrieval.py      # retrieve_hr_context_node (refactored)
            prompts.py        # build_hr_prompts_node
            reasoning.py      # llm_hr_reasoning_node
            scoring.py        # hr_scoring_node
            persistence.py    # save_hr_turn_node
            formatting.py     # format_hr_response_node
        helpers/
            context_formatter.py   # _format_cv_context, _format_jd_context, _format_sql_metadata
            candidate_resolver.py  # _resolve_candidates_by_name
            history_formatter.py   # _format_history
            cv_assembler.py        # ghép sections → virtual full CV (MỚI)
        graph.py              # create_hr_graph(), HRChatbot class

    candidate/
        state.py              # CandidateState TypedDict (tách từ candidate_graph.py)
        router.py             # Candidate Intent Router (MỚI, Sprint 2)
        nodes/
            session.py
            expansion.py      # dùng chung logic với hr/nodes/expansion.py
            retrieval.py
            scoring.py
            prompts.py
            reasoning.py
            persistence.py
            formatting.py
        graph.py

    shared/
        expansion.py          # query_expansion core logic (dùng chung cả 2 chatbot)
        hybrid_retrieval.py   # dense + keyword + RRF merge (MỚI)
        intent_types.py       # Enum IntentType, PipelineStrategy

    # Files giữ nguyên (không đổi):
    intent.py                 # IntentClassifier (Candidate dùng, sẽ replace Sprint 2)
    prompts.py                # Prompt templates
```

---

## 8. State Schema — Fields thêm mới

### HRChatState (thêm vào `hr/state.py`)
```python
pipeline_strategy: str          # "SEMANTIC"|"COMPARE"|"FILTER"|"ACTION"|"AGGREGATE"|"FIND_MORE"
query_intent:      str          # "RANK"|"COMPARE"|"DETAIL"|"ACTION"|"AGGREGATE"|"FILTER"|"FIND_MORE"
query_entities:    Dict         # extracted entities: skill_keywords, score_threshold, top_n, candidate_names
expanded_query:    Optional[str] # output của LLM expansion
skill_variants:    List[str]    # expanded skill list cho keyword search
```

### CandidateState (thêm vào `candidate/state.py`)
```python
pipeline_strategy: str
query_intent:      str          # "JD_SEARCH"|"JD_ANALYSIS"|"CV_ANALYSIS"|"APPLY"|"STATUS_CHECK"|"JD_CONVERSE"
query_entities:    Dict
expanded_query:    Optional[str]
skill_variants:    List[str]
```

---

## 9. Graph Edges — Thay đổi

### HR Graph
**Trước:**
```
load_session → load_scope → retrieve → build_prompt → llm → save → format
```

**Sau:**
```
load_session → load_scope → query_intelligence → [conditional branch]
                                  │
                    ┌─────────────┼──────────────────┐
                    ▼             ▼                   ▼
              ACTION/AGGR    COMPARE/DETAIL      RANK/FILTER/FIND_MORE
                    │             │                   │
                    │             │              query_expansion
                    │             │                   │
                    └─────────────┴──────────→ retrieve_hr_context
                                                      │
                                              build_hr_prompts
                                                      │
                                              llm_hr_reasoning
                                                      │
                                          [conditional: hr_scoring?]
                                                      │
                                               save_hr_turn
                                                      │
                                             format_hr_response
```

### Candidate Graph
Tương tự HR Graph. `classify_intent_node` + `should_retrieve` edge bị **replace** bởi `query_intelligence_node` trong Sprint 2.

---

## 10. Implementation Sprints

### Sprint 1 — Fix Issue #1: HR Graph Routing (ƯU TIÊN CAO)

**Bước 1: Tách `hr/state.py`**
- Move `HRChatState` TypedDict ra khỏi `hr_graph.py`
- Thêm 4 fields mới: `pipeline_strategy`, `query_intent`, `query_entities`, `expanded_query`, `skill_variants`
- Không thay đổi logic — pure move + extend

**Bước 2: Tách `hr/helpers/`**
- `_format_history` → `history_formatter.py`
- `_format_cv_context`, `_format_jd_context`, `_format_sql_metadata` → `context_formatter.py`
- `_resolve_candidates_by_name` → `candidate_resolver.py`
- `detect_hr_query_intent`, `_extract_top_n_from_query`, `_is_cv_count_query` → **KHÔNG tách**, sẽ bị replace bởi Router

**Bước 3: Tách `hr/nodes/`**
- Move từng node ra file riêng (thứ tự từ cuối graph lên để ít dependency nhất):
  - `formatting.py`, `persistence.py`, `scoring.py`, `session.py` (Cập nhật `session.py` query DB không limit để lấy functionCall gần nhất, chống mất cache state sau 20 turns).
  - `scope.py`, `prompts.py`, `reasoning.py`, `retrieval.py`

**Bước 4: Build `hr/router.py`**
- Hard-rule Router theo priority order (Section 3.3)
- Entity extraction bằng regex trong cùng bước
- Output: `pipeline_strategy` + `query_entities`
- Thêm `query_intelligence_node` wrapper function

**Bước 5: Wire Router vào `hr/graph.py`**
- Inject `query_intelligence_node` sau `load_candidate_scope`
- Thêm conditional edges
- Cập nhật `build_hr_prompts_node` nhận `pipeline_strategy` để adapt prompt
- Integration test từng intent path

**Deliverable Sprint 1:** HR graph route đúng COMPARE/DETAIL/ACTION/AGGREGATE — không còn chạy Qdrant thừa.

---

### Sprint 2 — Fix Issue #2: Hybrid Retrieval + Candidate Router

**Bước 6: Build `shared/hybrid_retrieval.py`**
- Dense search function (async)
- Keyword search function dùng `skills` MatchAny (async)
- RRF merge function

**Bước 7: Build `shared/expansion.py`**
- LLM Flash call với structured output
- Fallback strategy khi timeout

**Bước 8: Build `hr/nodes/expansion.py`**
- `query_expansion_node` wrapper
- Chỉ trigger khi `pipeline_strategy` thuộc {RANK, FILTER, FIND_MORE}

**Bước 9: Refactor `hr/nodes/retrieval.py`**
- Thay dense-only retrieval bằng hybrid_retrieval
- Wire FIND_MORE exclusion logic

**Bước 10: Build `candidate/router.py`**
- Tương tự HR Router nhưng intent taxonomy riêng
- Replace `classify_intent_node` + `should_retrieve` edge

**Bước 11: Wire Candidate graph**
- Inject Candidate Router
- Wire `query_expansion_node` cho JD_SEARCH intent
- Thêm tool validation logic trong `reasoning.py` (kiểm tra `scored_jobs` không rỗng trước khi Apply, validate `position_id`).

**Deliverable Sprint 2:** Retrieval quality cải thiện, candidate không bị miss do vocabulary mismatch. Candidate graph có routing đúng.

---

### Sprint 3 — Polish & Edge Cases

**Bước 12:** `cv_assembler.py` — ghép sections thành virtual full CV cho COMPARE/DETAIL

**Bước 13:** AGGREGATE pipeline — skip LLM call, format thẳng từ SQL data

**Bước 14:** Adapt `build_prompt` nodes nhận `pipeline_strategy` để:
- COMPARE/DETAIL: bỏ JD block, thêm comparison instruction
- ACTION: chỉ inject sql_metadata
- AGGREGATE: format stats template

**Bước 15:** Integration test toàn bộ intent paths cả 2 chatbot

---

## 11. Breaking Changes cần handle

| Thay đổi | Impact | Mitigation |
|----------|--------|------------|
| `HRChatState` move sang file mới | Import path thay đổi | Update tất cả import trong graph.py và node files |
| `detect_hr_query_intent` bị xóa | Không còn dùng | Replace bởi Router, xóa sau Sprint 1 |
| `should_retrieve` edge bị xóa (Candidate) | Graph structure thay đổi | Replace bởi conditional edges từ Router |
| `classify_intent_node` bị replace | Intent field vẫn giữ trong state | Router set cùng field name để backward compatible |
| Dense-only → Hybrid retrieval | Kết quả có thể thay đổi | Test với known queries trước khi deploy |

---

## 12. Quyết định kỹ thuật đã chốt

| Quyết định | Lựa chọn | Lý do |
|-----------|----------|-------|
| Skill synonyms | LLM expansion (không hardcode) | IT landscape thay đổi liên tục, hardcode không scale |
| Intent classification | Hard-rule pattern (không LLM) | 40% query HR không cần Qdrant — LLM là overkill |
| Query expansion | 1 LLM Flash call | Cover synonym tự động, ~50-100ms trade-off chấp nhận được |
| HR vs Candidate Router | Tách riêng | Tránh conflict, intent taxonomy khác nhau |
| Full CV | Ghép sections runtime | Không cần re-index Qdrant, zero migration cost |
| RRF weights | 0.6 dense + 0.4 keyword | Semantic vẫn là primary signal, keyword là bổ trợ |

---

## 13. Trạng thái hiện tại (cập nhật khi progress)

- [x] Planning & Architecture design hoàn thành
- [ ] Sprint 1 — Bước 1: Tách hr/state.py
- [ ] Sprint 1 — Bước 2: Tách hr/helpers/
- [ ] Sprint 1 — Bước 3: Tách hr/nodes/
- [ ] Sprint 1 — Bước 4: Build hr/router.py
- [ ] Sprint 1 — Bước 5: Wire Router vào hr/graph.py
- [ ] Sprint 2 — Bước 6-11
- [ ] Sprint 3 — Bước 12-15

---

## 14. Hướng dẫn dùng file này khi chuyển conversation

Khi mở conversation mới với AI, paste theo thứ tự:
1. File `implement_plan.md` này (toàn bộ)
2. Các file code hiện tại liên quan đến bước đang làm
3. Câu mở đầu: *"Tôi đang implement theo plan này, hiện tại đang ở Bước X. Tiếp tục từ đây."*

Cập nhật checklist ở Section 13 sau mỗi bước hoàn thành.
