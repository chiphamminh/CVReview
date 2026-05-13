# RAG Optimization Plan — CVReview Chatbot System

> **Last updated:** 2026-05-12  
> **Scope:** `chatbot-service` (Python/LangGraph) + `recruitment-service` (Java/MySQL)

---

## 0. Hiểu đúng kiến trúc (quan trọng)

Hệ thống dùng **Two-Stage Pipeline** — LLM **không** nhận toàn bộ CV/JD:

```
Stage 1 — RAG Filter (Qdrant, cheap):
  Query vector → search CV/JD collection → top-K chunks
  → Cross-Encoder rerank → top 5 unique CVs (hoặc JDs)

Stage 2 — LLM Scoring (chỉ trên kết quả đã lọc):
  Gemini nhận top 5-7 CVs → technical_score + experience_score + gap_advice
  → ⚠️ CONFIRMED BUG: đang gọi N lần tuần tự (1 call/CV), không phải batch
  → 5 CVs × ~6s/call = 30s chỉ riêng scoring — root cause chính của 2 phút
```

**Fix đơn giản nhất, impact lớn nhất:** Parallelize scoring calls với `asyncio.gather()`.

---

## 1. Intent Map (Final — Confirmed)

### 1.1 HR Chatbot

| Intent | RAG cần? | Bottleneck | Trigger mẫu |
|--------|----------|-----------|-------------|
| **SEARCH_CANDIDATES** | ✅ Full hybrid | Scoring + Retrieval | "Top 5 Java dev phù hợp nhất" |
| **FIND_MORE** | ✅ Hybrid + exclude | Retrieval + Scoring | "Tìm thêm, loại những người đã mời" |
| **SEND_INTERVIEW** | ❌ Tool call only | Tool HTTP × N | "Mời top 3, A: 9h T3, B: 10h T3, C: 2h T4" |
| **GENERATE_QUESTIONS** | ❌ Internal API | LLM output length | "Tạo câu hỏi phỏng vấn cho Nguyễn A" |

**Đã loại bỏ:** CANDIDATE_DETAIL (→ follow-up của SEARCH), STATUS_QUERIES (→ dashboard), COMPARE (→ follow-up tự nhiên).

### 1.2 Candidate Chatbot

| Intent | RAG cần? | Bottleneck | Trigger mẫu |
|--------|----------|-----------|-------------|
| **FIND_JOBS** | ✅ Full hybrid | Scoring + Retrieval | "Tìm việc phù hợp với tôi" |
| **CV_GAP_ANALYSIS** | ✅ CV + JD (no rerank) | LLM reasoning | "CV tôi thiếu gì cho vị trí Senior Java?" |
| **APPLY** | ❌ Tool call only | Tool HTTP | "Tôi muốn apply vị trí này" |

**Đã loại bỏ:** STATUS_CHECK (→ UI stage tags), JOB_DETAIL (→ candidate xem JD card).

---

## 2. Data Design Decisions

### 2.1 gap_advice thay thế learning_path

`learning_path` sai ngữ nghĩa với Senior. Đổi thành `gap_advice` — AI tự calibrate theo seniority detect từ CV:

| Level | Output |
|-------|--------|
| Junior (0–3 năm) | Structured roadmap: "Học Spring Boot → Docker → 60 ngày" |
| Mid (3–6 năm) | Gap bridging: "Có Spring nhưng thiếu Kafka event-driven" |
| Senior (6+ năm) | Strategic advice: "Thiếu K8s production scale — CKA cert" |

### 2.2 Nơi lưu gap_advice

```
FIND_JOBS Turn 1:
  AI scoring → gap_advice cho từng job → lưu trong chat_history.functionCall
  UI hiển thị trong chat. KHÔNG lưu vào cv_analysis table.

APPLY (score ≥ 70/minimum_fit_score):
  → Tạo cv_analysis record: technical_score, experience_score, overall_status
  → KHÔNG có gap_advice (candidate đã qualify)

APPLY (score < 70) hoặc CV_GAP_ANALYSIS:
  → Guardrail block / gap analysis
  → gap_advice calibrated theo seniority, lưu trong chat history
  → KHÔNG tạo cv_analysis record
```

**Quyết định:** Drop column `learning_path` khỏi `cv_analysis` table. `gap_advice` chỉ sống trong `chat_history.functionCall`.

### 2.3 cv_analysis chỉ tạo khi APPLY thành công

Mọi AI analysis trước đó (scoring, gap_advice) chỉ lưu tại `functionCall` trong `chat_history`. `cv_analysis` record được tạo duy nhất khi `finalize_application` tool call thành công.

---

## 3. Optimized Graph Path per Intent

### 3.1 Nguyên tắc

> Node đắt tiền (Qdrant, Cross-Encoder, LLM scoring) phải có **điều kiện skip rõ ràng** dựa trên session state. Router phân tích context TRƯỚC khi dispatch.

### 3.2 Session State (tối giản — chỉ những gì cần thiết)

```python
# Lưu trong chat_history.functionCall, restore tại load_session node
{
  # Routing state
  "prev_strategy":    "SEARCH_CANDIDATES",
  "conv_state":       "IDLE",           # IDLE | AWAITING_CONFIRM
  "pending_action":   "SEND_INTERVIEW", # chỉ khi conv_state = AWAITING_CONFIRM
  "pending_params":   {...},            # params đã parse, chờ confirm

  # HR-specific
  "shown_cv_ids":     [12, 45, 78],     # cho FIND_MORE exclude
  "ranked_cv_list":   [{"rank": 1, "cvId": 12, "name": "Nguyễn A"}, ...],

  # Candidate-specific
  "scored_jobs":      [...],            # cache Turn 1 scoring
  "last_jd_id":       3,               # JD context cho CV_GAP_ANALYSIS
}
```

### 3.3 HR Chatbot — Đường đi

```
load_session → route_hr_intent
    │
    ├─ SEARCH_CANDIDATES
    │   expansion* → retrieve_hybrid → scoring(batch) → prompts → llm → save
    │   *skip nếu query ≤ 5 words với skill keywords rõ ràng
    │
    ├─ FIND_MORE
    │   SKIP expansion
    │   retrieve_hybrid(exclude_ids=shown_cv_ids) → scoring(batch) → prompts → llm → save
    │
    ├─ SEND_INTERVIEW
    │   SKIP expansion, retrieve, scoring
    │   [entity resolve: "người thứ 2" → cvId từ ranked_cv_list]
    │   prompts → llm(tool: send_interview_email × N parallel) → save
    │
    └─ GENERATE_QUESTIONS
        SKIP expansion, Qdrant
        fetch full cv+jd text (internal API) → prompts → llm → save
```

### 3.4 Candidate Chatbot — Đường đi

```
load_session → route_candidate_intent
    │
    ├─ FIND_JOBS
    │   ├─ New search (no scored_jobs cache OR new skill keywords):
    │   │   expansion → retrieve_hybrid(CV+JD) → scoring(batch) → prompts → llm → save
    │   │
    │   └─ Follow-up (scored_jobs cache hit, query ngắn, không skill mới):
    │       SKIP expansion, retrieve, scoring
    │       prompts(scored_jobs từ cache) → llm → save
    │
    ├─ CV_GAP_ANALYSIS
    │   [fallback nếu last_jd_id null: → route sang FIND_JOBS trước]
    │   SKIP expansion, scoring
    │   retrieve(cv_id + last_jd_id, no rerank) → prompts(gap template) → llm → save
    │
    └─ APPLY
        SKIP expansion, retrieve, scoring
        [guardrail: score ≥ threshold từ scored_jobs cache]
        → Pass: llm(tool: finalize_application) → save
        → Fail: prompts(gap_advice by seniority) → llm → save
```

### 3.5 Skip Flag Decision

```python
def decide_path_flags(state: dict) -> dict:
    strategy       = state["pipeline_strategy"]
    scored_jobs    = state.get("scored_jobs")
    has_new_skills = bool(state.get("query_entities", {}).get("skill_keywords"))
    short_query    = len(state.get("query", "").split()) <= 12

    flags = {"skip_expansion": False, "skip_retrieval": False, "skip_scoring": False}

    if strategy == "APPLY":
        flags.update(skip_expansion=True, skip_retrieval=True, skip_scoring=True)

    elif strategy == "FIND_JOBS":
        is_followup = scored_jobs and not has_new_skills and short_query
        if is_followup:
            flags.update(skip_expansion=True, skip_retrieval=True, skip_scoring=True)

    elif strategy == "CV_GAP_ANALYSIS":
        flags.update(skip_expansion=True, skip_scoring=True)

    elif strategy == "FIND_MORE":
        flags["skip_expansion"] = True

    elif strategy in ("SEND_INTERVIEW", "GENERATE_QUESTIONS"):
        flags.update(skip_expansion=True, skip_retrieval=True, skip_scoring=True)

    return flags
```

---

## 4. Conversation State Machine (Minimal)

Chỉ implement 3 categories thực sự gây pain, bỏ qua những case phức tạp không xứng giá trị:

### Categories được implement

| Cat | Case | Fix |
|-----|------|-----|
| **1+2** | Confirm/Reject sau AWAITING_CONFIRM | State machine Layer 0 |
| **4** | Continuation ("tiếp", "3 người nữa") | Layer 3 context fallback |
| **5** | Ordinal reference ("người thứ 2") | ranked_cv_list resolution |

### Categories bỏ qua (complexity > value)

| Cat | Case | Lý do bỏ |
|-----|------|----------|
| 3 | Modification pre-confirm | Bảo user nói lại từ đầu là UX acceptable |
| 6 | Pronoun ("họ", "người đó") | Quá ambiguous, dễ sai hơn bỏ qua |
| 7 | Reply to LLM clarification | Rare, LLM tự handle qua history context |
| 8 | Selection từ list ("cái 2") | Merge vào ordinal resolution |

### Implementation — LLM tự set flag (không dùng regex heuristic)

**Vấn đề với regex detection:** LLM paraphrase confirmation theo nhiều cách khác nhau → regex miss → state không được set.

**Giải pháp:** LLM khai báo rõ qua structured tag trong system prompt:

```python
# Thêm vào system prompt:
"""
Khi cần user xác nhận trước khi thực hiện action không thể hoàn tác
(gửi email, nộp đơn), BẮT BUỘC kết thúc response bằng:
<needs_confirm action="SEND_INTERVIEW"></needs_confirm>

Khi không cần confirm, KHÔNG thêm tag này.
"""

# save_turn node — parse tag, không regex heuristic:
import re
tag = re.search(r'<needs_confirm action="(\w+)">', llm_response)
if tag:
    state["conv_state"]    = "AWAITING_CONFIRM"
    state["pending_action"] = tag.group(1)
    state["final_answer"]  = llm_response[:tag.start()].strip()
else:
    state["conv_state"] = "IDLE"
```

### Router với State Machine (≤ 50 lines tổng)

```python
_CONFIRM  = re.compile(r"^(ok|yes|đúng|đồng ý|xác nhận|gửi đi|được|ừ|sure|confirm)[\s!.]*$", re.I)
_REJECT   = re.compile(r"^(không|no|thôi|hủy|cancel|dừng)[\s!.]*$", re.I)
_CONTINUE = re.compile(r"\b(tiếp|thêm|more|tìm thêm|còn ai|next|3 người nữa|\d+ người nữa)\b", re.I)

_CONTINUATION_MAP = {
    "SEARCH_CANDIDATES": "FIND_MORE",
    "FIND_MORE":         "FIND_MORE",
}

def route_intent(state: dict) -> str:
    query = state["query"].strip()

    # L0: State machine — highest priority
    if state.get("conv_state") == "AWAITING_CONFIRM":
        if _CONFIRM.match(query):  return state["pending_action"]
        if _REJECT.match(query):   return "CANCELLED"
        return state["pending_action"]  # modification → LLM re-parse pending_params

    # L1: Hard-rule patterns (hiện tại — giữ nguyên)
    # ... existing regex matching ...

    # L2: Context-informed short query
    if len(query.split()) <= 4 and _CONTINUE.search(query):
        prev = state.get("prev_strategy", "")
        if prev in _CONTINUATION_MAP:
            return _CONTINUATION_MAP[prev]

    # L3: Default
    return "SEARCH_CANDIDATES"  # HR  /  "FIND_JOBS"  # Candidate
```

### Ordinal Entity Resolution

```python
_ORDINAL = re.compile(
    r"(?:người|ứng viên|vị trí|job|candidate)\s*(?:thứ\s*)?(\d+|đầu tiên|cuối|last|first)",
    re.I
)

def resolve_ordinal(query: str, ranked_list: list) -> Optional[int]:
    """Map 'người thứ 2' → cvId hoặc positionId từ ranked_list."""
    m = _ORDINAL.search(query)
    if not m: return None
    raw = m.group(1).lower()
    idx = {"đầu tiên": 0, "first": 0, "cuối": -1, "last": -1}.get(raw)
    if idx is None:
        try: idx = int(raw) - 1
        except: return None
    if 0 <= idx < len(ranked_list):
        return ranked_list[idx].get("cvId") or ranked_list[idx].get("positionId")
    return None
```

---

## 5. Performance Bottlenecks & Fixes

### 5.1 Xác định bottleneck thật của 2 phút (cần verify)

```
Verify ngay: scoring node gọi Gemini bao nhiêu lần cho 5 CVs?
  → 1 batch call (tất cả 5 CVs trong 1 prompt): ~5-8s — acceptable
  → 5 separate calls (1 call/CV):               ~25-40s — đây là bug

Cách kiểm tra: đếm số Gemini API calls trong logs khi chạy SEARCH_CANDIDATES
```

### 5.2 Priority Matrix

| ID | Fix | Impact | Effort | Sprint |
|----|-----|--------|--------|--------|
| **F0** | **Parallelize scoring calls (asyncio.gather)** | Critical | Low | S1 FIRST |
| **F1** | **Fix chunkText vs jdText reranker bug** | Critical | Low | S1 |
| **F2** | **Qdrant payload indexes** | High (ở scale) | Low | S1 |
| **F3** | **Eager-load reranker tại startup** | High (cold start) | Low | S1 |
| F4 | Async reranker (run_in_executor) | High (concurrency) | Low | S1 |
| F5 | Async embed_text (run_in_executor) | Medium | Low | S1 |
| F6 | LLM singleton cho expansion | Medium | Low | S1 |
| F7 | Parallelize CV + JD Qdrant searches | High | Low | S2 |
| F8 | Skip expansion: simple queries (≤5 words + skills) | Medium | Low | S2 |
| F9 | Skip expansion + retrieval: FIND_JOBS follow-up | High | Medium | S2 |
| F10 | JD vector cache per positionId (HR mode) | Medium | Low | S2 |
| F11 | LRU embedding cache | Medium | Low | S2 |
| F12 | Conversation state machine (AWAITING_CONFIRM) | Critical | Medium | S2 |
| F13 | Ordinal entity resolution | High | Medium | S2 |
| F14 | Tiered scoring: Flash default, Pro on-demand | High | Medium | S3 |
| F15 | AND/OR compound skill filter | High | Medium | S3 |
| F16 | Batch interview email (parallel tool calls) | Medium | Low | S3 |
| F17 | Token streaming qua SSE | High (UX) | High | S4 |
| F18 | Redis session cache | Medium | High | S4 |
| F19 | EXTERNAL mode: bypass scoring, dual-sort ranking | Critical (correctness) | Medium | S5 |

### 5.3 Target Latency sau optimization

| Intent | Hiện tại | Target | Fixes chính |
|--------|---------|--------|------------|
| SEARCH_CANDIDATES T1 | ~120s | 8–12s | F0,F1,F2,F3,F4,F7,F14 |
| FIND_MORE | ~60s | 6–10s | F0,F1,F2,F3,F4,F7 |
| SEND_INTERVIEW | ~5s | 1–3s | F12,F13 |
| GENERATE_QUESTIONS | ~8s | 4–7s | — |
| FIND_JOBS T1 | ~90s | 6–10s | F1,F2,F3,F4,F7,F14 |
| FIND_JOBS T2+ (cache) | ~30s | 1.5–3s | F9 |
| CV_GAP_ANALYSIS | ~15s | 3–5s | F2,F4 |
| APPLY | ~5s | 1–2s | F12 |
| AWAITING_CONFIRM flows | ~60s | <1s | F12 |

---

## 6. Sprint Plan

### Sprint 1 — Critical Fixes (1–2 ngày)

Không cần đụng đến logic — chỉ fix bugs và technical issues:

**F0: Parallelize scoring calls (LÀM NGAY — root cause của 2 phút)**
```python
# hr/nodes/scoring.py — BEFORE: sequential for loop
for candidate in candidates:
    response = await llm.ainvoke(...)   # chờ từng cái → 5 × 6s = 30s

# AFTER: parallel với asyncio.gather → tất cả chạy đồng thời → ~6s tổng
async def _score_one(candidate: dict) -> dict:
    response = await llm.ainvoke([HumanMessage(content=_build_prompt(candidate, jd_text))])
    return _parse_score(response, candidate)

results = await asyncio.gather(*[_score_one(c) for c in candidates])
```
Tương tự cần apply cho candidate scoring node nếu cùng pattern.

---

**F1: Fix reranker chunkText vs jdText (LÀM NGAY)**
```python
# reranker.py:69 — silent bug: JD chunks score trên empty string
# BEFORE:
pairs = [(query, chunk["payload"].get("chunkText", "")) for chunk in chunks]
# AFTER:
def _text(p): return (p.get("jdText") or p.get("chunkText") or "").strip()
pairs = [(query, _text(chunk["payload"])) for chunk in chunks]
```

**F3: Eager-load reranker tại startup**
```python
# main.py — lifespan():
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, reranker._get_model)
await loop.run_in_executor(None, lambda: reranker._get_model().predict([("warmup", "text")]))
```

**F4: Async reranker**
```python
# reranker.py — thêm async wrapper:
async def rerank_and_group_async(self, query, chunks, id_field, top_n):
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: self.rerank_and_group(query, chunks, id_field, top_n)
    )
```

**F5: Async embed_text trong retriever.py**
```python
# retriever.py — tất cả embed_text() calls:
loop = asyncio.get_event_loop()
query_vector = await loop.run_in_executor(
    None, lambda: self.embedding_service.embed_text(query, is_query=True)
)
```

**F6: LLM singleton cho expansion**
```python
# expansion.py — module-level singleton:
_LLM = None
def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, ...)
    return _LLM
```

**F2: Qdrant payload indexes (one-time migration)**
```python
# cv_embeddings:
for field, schema in [
    ("is_latest", "bool"), ("cvId", "integer"), ("candidateId", "keyword"),
    ("positionId", "integer"), ("sourceType", "keyword"),
    ("applied_position_ids", "integer"), ("seniorityLevel", "keyword"),
]:
    client.create_payload_index("cv_embeddings", field, field_schema=schema)
# jd_embeddings:
client.create_payload_index("jd_embeddings", "positionId", "integer")
```

---

### Sprint 2 — Routing & Graph Optimization (2–3 ngày)

**F7: Parallelize CV + JD Qdrant searches**
```python
# retriever.py — _retrieve_jd_search_with_cv():
loop = asyncio.get_event_loop()
cv_fut = loop.run_in_executor(None, lambda: qdrant.search_similar(cv_collection, ...))
jd_fut = loop.run_in_executor(None, lambda: qdrant.search_similar(jd_collection, ...))
cv_results, jd_chunks = await asyncio.gather(cv_fut, jd_fut)
```

**F8+F9: Skip expansion logic**
```python
# expansion.py:
def _should_skip(query: str, skills: list, scored_jobs, has_new_skills: bool) -> bool:
    simple_query = len(query.split()) <= 5 and len(skills) >= 1
    followup     = scored_jobs is not None and not has_new_skills
    return simple_query or followup
```

**F12: Conversation state machine**
- Thêm `conv_state` + `pending_action` + `pending_params` vào state schema
- Thêm `<needs_confirm>` tag vào system prompt
- Parse tag trong save_turn node
- Thêm Layer 0 vào router

**F13: Ordinal entity resolution**
- Thêm `ranked_cv_list` / `ranked_job_list` vào session state
- Implement `resolve_ordinal()` function
- Gọi trong route node trước khi dispatch SEND_INTERVIEW / CV_GAP_ANALYSIS

**F10+F11: Caching**
```python
# JD vector cache (retriever.py):
_JD_VEC: dict = {}  # positionId → vector
# LRU embedding cache (embedding.py):
from collections import OrderedDict
_EMBED_CACHE = OrderedDict()  # maxsize=512
```

---

### Sprint 3 — Quality & Scale (3–5 ngày)

**F14: Tiered scoring**
```python
# scoring.py:
def _model(query, strategy):
    if strategy == "APPLY": return GEMINI_PRO    # guardrail phải chính xác
    return GEMINI_FLASH                           # browsing/ranking dùng Flash
```

**F15: AND/OR compound skill**
```python
# router.py — detect AND intent:
_AND = re.compile(r"\bvà\b|\band\b|\bcả\b|\bboth\b", re.I)
skill_logic = "AND" if len(skills) >= 2 and _AND.search(query) else "OR"
# retriever.py — AND dùng multiple must conditions thay vì MatchAny
```

**F16: Batch email parallel**
```python
# HR reasoning node — tool loop không break sau call đầu:
results = await asyncio.gather(*[_send_email(args) for args in email_list])
```

---

### Sprint 4 — UX Improvement (5–7 ngày)

**F17: Token streaming** — User thấy tokens sau ~200ms thay vì đợi toàn bộ response.  
Lưu ý: Buffer tool-use turns (APPLY, SEND_INTERVIEW), stream non-tool turns.

**F18: Redis session cache** — Cache history với 10-min TTL, invalidate sau save_message.

---

### Sprint 5 — Architecture Correctness: EXTERNAL Mode (1–2 ngày)

**Vấn đề:** HR Chatbot EXTERNAL mode hiện re-score candidates đã có `cv_analysis` record (tạo bởi Candidate Chatbot khi apply). Dẫn đến: (1) ghi đè điểm → inconsistency, (2) lãng phí LLM call, (3) candidate thấy điểm khác với HR.

**F19: EXTERNAL mode bypass scoring + dual-sort ranking**

*Phần 1 — Bypass scoring trong reasoning node:*
```python
# hr/nodes/reasoning.py — auto-trigger scoring block
if strategy in ("RANK", "FILTER") and not state.get("pending_scoring_candidates"):
    if state["mode"] == "EXTERNAL":
        # Điểm đã có trong sql_metadata từ scope_node — không score lại
        print("[Reasoning] EXTERNAL mode → skip scoring, use existing cv_analysis scores")
    else:
        # INTERNAL: chưa có điểm → trigger scoring như cũ
        state["pending_scoring_candidates"] = [...]
```

*Phần 2 — Dual-sort ranking cho EXTERNAL (trong retrieval node):*
```python
# hr/nodes/retrieval.py — sau khi có cv_results từ reranker
# Primary sort: existing avg_score (fit với vị trí, static)
# Tiebreaker:   reranker_score   (fit với câu HR đang hỏi, query-aware)
if state["mode"] == "EXTERNAL":
    cv_id_to_meta = state.get("cv_id_to_meta", {})
    def _sort_key(chunk):
        cv_id     = chunk.get("payload", {}).get("cvId")
        meta      = cv_id_to_meta.get(cv_id, {})
        tech      = meta.get("technicalScore", 0)
        exp       = meta.get("experienceScore", 0)
        avg_score = (tech + exp) / 2
        reranker  = chunk.get("reranker_score", 0.0)
        return (avg_score, reranker)

    cv_results = sorted(cv_results, key=_sort_key, reverse=True)
```

*Phần 3 — Build prompts hiển thị điểm sẵn có (EXTERNAL):*
```python
# hr/nodes/prompts.py — thêm branch cho EXTERNAL mode
if state["mode"] == "EXTERNAL":
    # Inject bảng điểm từ cv_id_to_meta vào system_prompt
    # LLM đọc điểm → giải thích, so sánh, gợi ý — không cần tính lại
    scored_section = _build_external_scores_section(cv_results, cv_id_to_meta)
    system_prompt += f"\n\n**Điểm đã được pre-screened từ hệ thống:**\n{scored_section}"
```

**Tại sao dual-sort tốt hơn chỉ dùng avg_score:**

| Tình huống | avg_score only | dual-sort |
|---|---|---|
| HR hỏi "ai có K8s production exp?" | A=78, B=78 → thứ tự ngẫu nhiên | A=78/0.82, B=78/0.61 → A ưu tiên |
| HR hỏi "ai lead team tốt?" | B=78, A=78 → thứ tự ngẫu nhiên | B=78/0.79, A=78/0.55 → B ưu tiên |
| Điểm khác nhau rõ (80 vs 72) | Primary sort đã đủ | Primary sort đã đủ |

Reranker score thay đổi theo từng query → cùng pool ứng viên tự động re-order theo ngữ cảnh HR đang hỏi, không cần LLM call thêm.

---

## 7. System Bloat Control Principles

Áp dụng xuyên suốt quá trình implement:

1. **Router ≤ 50 lines** — nếu vượt, tách thành helper function, không thêm layer mới
2. **State fields tối giản** — mỗi field mới phải có clear owner (node nào set, node nào read, node nào clear)
3. **Prompt budget cứng** — tối đa 5 CVs/JDs trong scoring prompt, bất kể user yêu cầu bao nhiêu
4. **Một intent = một retrieval strategy** — không có conditional retrieval trong node, chỉ trong router
5. **Không implement category nào chỉ vì "có thể xảy ra"** — chỉ fix khi user báo bug thực tế
6. **CV_GAP_ANALYSIS fallback rõ ràng** — nếu `last_jd_id` null → redirect FIND_JOBS, không để LLM tự đoán
