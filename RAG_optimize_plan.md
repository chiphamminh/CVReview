# RAG Optimization Plan — CVReview Chatbot System

> **Author:** Senior Dev / AI Engineer analysis  
> **Date:** 2026-05-11  
> **Scope:** `chatbot-service` (Python/LangGraph) + `recruitment-service` (Java/MySQL) integration

---

## 1. Current Architecture Overview

### 1.1 Candidate Chatbot — Request Flow

```
User Query
    │
    ├─► [1] load_session_history + load_active_positions  (concurrent HTTP)
    │       → recruitment-service /internal/chatbot/
    │
    ├─► [2] route_candidate_intent  (hard-rule regex, zero LLM cost)
    │       → pipeline_strategy: JD_SEARCH / CV_ANALYSIS / JD_CONVERSE / ...
    │
    ├─► [3] expansion node  (LLM Flash, JD_SEARCH only)
    │       → expanded_query + skill_variants
    │
    ├─► [4] retrieval node
    │   ├─ JD_SEARCH: hybrid_retrieve_cv (Dense + Keyword → RRF → Cross-Encoder)
    │   │              + JD dense search + rerank_and_group
    │   ├─ CV_ANALYSIS: CV dense search only
    │   └─ JD_ANALYSIS: JD dense search only
    │
    ├─► [5] scoring node  (Gemini 2.5 Pro — Turn 1 only, cached on Turn 2+)
    │
    ├─► [6] build_prompts_node  (prompt assembly)
    │
    ├─► [7] llm_reasoning_node  (Gemini 2.5 Flash + tool use)
    │
    └─► [8] save_turn_node  (async HTTP persist to MySQL)
```

### 1.2 HR Chatbot — HR Mode Flow (retrieve_for_hr_mode_hr)

```
HR Query
    ├─► embed(query)  →  search JD chunks (positionId filter)  [Qdrant call #1]
    ├─► concat JD text  →  embed(jd_text)                      [Embedding call #2]
    └─► search CV chunks (JD vector)                           [Qdrant call #2]
         └─► rerank_and_group (Cross-Encoder)
```

### 1.3 Two-Stage RAG Pipeline (hybrid_retrieval.py)

```
expanded_query
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
Dense Search (asyncio.gather)        Keyword Search
embed(expanded_query) → Qdrant       MatchAny on `skills` metadata
limit = top_n × 4                    limit = top_n × 3
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
              RRF Merge (K=60, dense=0.6, keyword=0.4)
              Cap 6 chunks/cvId
                     ▼
          Cross-Encoder rerank → group by cvId (Max Score)
                     ▼
              Top-N unique CVs returned
```

---

## 2. Latency Breakdown & Bottleneck Analysis

### 2.1 Estimated Latency per Stage (JD_SEARCH, Turn 1)

| # | Stage | Module | Est. Time | Notes |
|---|-------|--------|-----------|-------|
| 1 | Session history load | recruitment_api.py | 100–200ms | HTTP call |
| 1 | Active positions load | recruitment_api.py | 100–150ms | Concurrent with above |
| 2 | Intent routing | router.py | ~1ms | Regex only, negligible |
| 3 | **Query expansion** | expansion.py | **500–2000ms** | LLM Flash call, 4s timeout |
| 4a | Query embedding | embedding.py | 30–80ms | Local SBERT, CPU-bound |
| 4b | Dense search (Qdrant Cloud) | hybrid_retrieval.py | 100–250ms | Network + index scan |
| 4c | Keyword search (Qdrant Cloud) | hybrid_retrieval.py | 80–200ms | Parallel with 4b |
| 4d | RRF merge | hybrid_retrieval.py | ~1ms | Pure Python dict ops |
| 4e | **Cross-Encoder rerank** | reranker.py | **200–800ms** | CPU-bound, ~45 pairs |
| 5 | **Gemini 2.5 Pro scoring** | scoring.py | **2000–5000ms** | Turn 1 only |
| 6 | Prompt assembly | prompts.py | ~2ms | String ops |
| 7 | **Gemini 2.5 Flash reasoning** | reasoning.py | **2000–5000ms** | Every turn |
| 8 | Persist messages (async) | persistence.py | ~100ms | Non-blocking |

**Total JD_SEARCH Turn 1: ~5–13 seconds**  
**Total JD_SEARCH Turn 2+ (cache hit): ~4–8 seconds**  
**Target after optimization: Turn 1 ≤ 6s, Turn 2+ ≤ 4s**

### 2.2 Identified Bottlenecks (Root Causes)

#### BN-01: Query Expansion LLM Call on Every JD_SEARCH Turn
- **Location:** `app/rag/shared/expansion.py` → `_build_flash_llm()`
- **Issue 1:** A **new `ChatGoogleGenerativeAI` instance** is created every call — connection overhead per request.
- **Issue 2:** Called on **every JD_SEARCH turn including Turn 2+**, even when conversation context already has skill context from Turn 1.
- **Issue 3:** For simple single-skill queries (`"tìm việc Python"`), LLM adds no value over whitelist extraction already done by router.
- **Impact:** +500–2000ms per JD_SEARCH request.

#### BN-02: Synchronous Embedding Blocks Event Loop in retriever.py
- **Location:** `app/services/retriever.py:103` → `retrieve_for_intent()`
  ```python
  query_vector = self.embedding_service.embed_text(query, is_query=True)
  ```
- **Issue:** `embed_text` is CPU-bound (SentenceTransformer inference). Called **synchronously** in an async method — blocks the FastAPI event loop, preventing other requests from being served concurrently.
- **Note:** `hybrid_retrieval.py` correctly uses `loop.run_in_executor()`. `retriever.py` does not.

#### BN-03: Sequential CV + JD Qdrant Searches in _retrieve_jd_search_with_cv
- **Location:** `app/services/retriever.py:198–228`
- **Issue:** CV search completes, then JD search starts. These are **completely independent** operations.
  ```python
  cv_results = self.qdrant_service.search_similar(...)  # blocks ~150ms
  jd_chunks  = self.qdrant_service.search_similar(...)  # starts AFTER cv
  ```
- **Impact:** +100–200ms wasted per JD_SEARCH call.

#### BN-04: Cross-Encoder Model Lazy Loading (Cold Start)
- **Location:** `app/services/reranker.py:35` → `_get_model()`
- **Issue:** BAAI/bge-reranker-v2-m3 (~550MB) is loaded on **first rerank call** only. The first real user request triggers model load, taking **5–15 seconds**.
- **Impact:** First user after restart gets extremely slow response with no warning.

#### BN-05: Cross-Encoder is CPU-Bound, Runs in Async Context
- **Location:** `app/services/reranker.py:72` → `model.predict(pairs)`
- **Issue:** `CrossEncoder.predict()` is synchronous and CPU-bound. Called directly in an `async` graph node without `run_in_executor()` — blocks the event loop during inference (~200–800ms).
- **Impact:** Blocks all concurrent requests while reranking.

#### BN-06: HR Mode Has 2 Sequential Embedding + Qdrant Calls
- **Location:** `app/services/retriever.py:382–412` → `retrieve_for_hr_mode_hr()`
- **Issue:** 
  1. `embed_text(query)` → `search_similar(jd_collection)` → get JD chunks
  2. Concatenate JD text → `embed_text(jd_text)` → `search_similar(cv_collection)`  
  The JD embedding vector (Step 2) is recomputed on **every HR query** even though it's deterministic for a given `position_id`.
- **Impact:** +30–200ms per HR query + repeated network round-trips.

#### BN-07: No Embedding Vector Cache
- **Location:** `app/services/embedding.py`
- **Issue:** No memoization. Identical queries (e.g., consecutive HR queries about the same position) re-run SentenceTransformer inference.
- **Impact:** +30–80ms per duplicate query.

#### BN-08: Skill Extraction Whitelist is O(N×M) Brute Force
- **Location:** `app/rag/candidate/router.py:131–140` → `_extract_skill_keywords()`
- **Issue:** For each query, iterates over all 90+ whitelist skills and does `re.search(rf'\b{skill}\b', query_lower)` — 90+ regex compilations per call (no `re.compile` cache used inside the loop).
- **Impact:** Minor (~2–5ms), but grows with whitelist size.

#### BN-09: No Conversation-Aware Routing
- **Location:** `app/rag/candidate/router.py`
- **Issue:** Router receives only the current `query` — no conversation history context. If Turn 1 was a JD_SEARCH and Turn 2 is `"tell me more about the salary"`, the router routes to `JD_CONVERSE` correctly. But `"what about experience requirements?"` maps to `JD_ANALYSIS` even though it likely refers to the same job from Turn 1. The `jd_id` is not threaded through from conversation context.
- **Impact:** Suboptimal retrieval routing for follow-up questions.

#### BN-10: Reranker Uses chunkText Field Only — Misses JD Context
- **Location:** `app/services/reranker.py:69`
  ```python
  pairs = [(query, chunk.get("payload", {}).get("chunkText", "")) for chunk in chunks]
  ```
- **Issue:** For JD chunks, the relevant text is in `jdText` field, not `chunkText`. The `get_chunk_text()` helper in `retriever.py` handles this correctly (`jdText` → fallback `chunkText`), but the reranker bypasses it.
- **Impact:** JD reranking scores are computed on empty strings when JD payload uses `jdText` key — degrades ranking quality.

---

## 3. Optimization Strategy

### Priority Matrix

| ID | Optimization | Impact | Effort | Priority |
|----|-------------|--------|--------|----------|
| O1 | Parallelize CV + JD Qdrant searches | High | Low | P1 |
| O2 | Async embed_text via run_in_executor | High | Low | P1 |
| O3 | Eager-load reranker at startup | High | Low | P1 |
| O4 | Async reranker via run_in_executor | High | Low | P1 |
| O5 | Fix reranker chunkText vs jdText bug | High | Low | P1 |
| O6 | LLM singleton for expansion | Medium | Low | P1 |
| O7 | Skip expansion on Turn 2+ (cache) | High | Medium | P2 |
| O8 | Skip expansion for simple queries | Medium | Low | P2 |
| O9 | LRU embedding cache | Medium | Low | P2 |
| O10 | Cache JD vector per positionId | Medium | Medium | P2 |
| O11 | Pre-compile skill regex patterns | Low | Low | P2 |
| O12 | Qdrant payload index verification | High | Low | P2 |
| O13 | Tiered scoring model (Flash→Pro) | Medium | Medium | P3 |
| O14 | Conversation-aware routing | Medium | High | P3 |
| O15 | Token streaming via SSE | High | High | P3 |
| O16 | Redis session cache | Medium | High | P3 |

---

## 4. Priority 1 — Quick Wins (Low Effort, High Impact)

### O1: Parallelize CV + JD Qdrant Searches

**File:** `app/services/retriever.py` → `_retrieve_jd_search_with_cv()`

**Problem:** CV and JD Qdrant searches run sequentially.

**Fix:**
```python
# BEFORE (sequential ~300ms total):
cv_results = self.qdrant_service.search_similar(cv_collection, ...)
jd_chunks  = self.qdrant_service.search_similar(jd_collection, ...)

# AFTER (parallel ~150ms total):
import asyncio

async def _search_cv():
    return self.qdrant_service.search_similar(cv_collection, ...)
async def _search_jd():
    return self.qdrant_service.search_similar(jd_collection, ...)

cv_results, jd_chunks = await asyncio.gather(_search_cv(), _search_jd())
```

Note: `qdrant_service.search_similar()` is synchronous (uses sync QdrantClient). Wrap each in `asyncio.get_event_loop().run_in_executor(None, ...)` before gathering, or migrate `QdrantService` to use `AsyncQdrantClient`.

**Estimated gain:** -100–200ms per JD_SEARCH request.

---

### O2: Fix Synchronous embed_text Blocking Event Loop

**File:** `app/services/retriever.py:103` → `retrieve_for_intent()`  
Also: `retriever.py:462` → `retrieve_for_hr_mode_candidate()`

**Problem:** `embed_text()` is called synchronously in async methods.

**Fix:**
```python
# BEFORE:
query_vector = self.embedding_service.embed_text(query, is_query=True)

# AFTER:
loop = asyncio.get_event_loop()
query_vector = await loop.run_in_executor(
    None, lambda: self.embedding_service.embed_text(query, is_query=True)
)
```

This matches the pattern already used in `hybrid_retrieval.py:_dense_search()`.

**Estimated gain:** Unblocks event loop during ~30–80ms inference. More important under concurrent load.

---

### O3: Eager-Load Reranker Model at Startup

**File:** `app/main.py` → `lifespan()` function  
**File:** `app/services/reranker.py`

**Problem:** BAAI/bge-reranker-v2-m3 loads on first user request (~5–15s cold start).

**Fix — Add warm-up in lifespan:**
```python
# app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # existing: embedding model + Qdrant preload
    logger.info("Pre-loading Cross-Encoder reranker model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, reranker._get_model)  # trigger lazy load
    # optional: warm up with a dummy pair to JIT-compile inference path
    await loop.run_in_executor(
        None, lambda: reranker._get_model().predict([("warm-up query", "warm-up text")])
    )
    logger.info("Reranker model ready.")
    yield
```

**Estimated gain:** Eliminates 5–15s first-request latency. Critical for production reliability.

---

### O4: Async Reranker Execution via Thread Pool

**File:** `app/services/reranker.py` → `rerank_and_group()`  
**File:** `app/rag/shared/hybrid_retrieval.py:235`

**Problem:** `CrossEncoder.predict(pairs)` is CPU-bound (~200–800ms), called in async context.

**Fix — Make `rerank_and_group` async:**
```python
# app/services/reranker.py
import asyncio

async def rerank_and_group_async(
    self,
    query: str,
    chunks: List[Dict[str, Any]],
    id_field: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: self.rerank_and_group(query, chunks, id_field, top_n)
    )
```

Keep synchronous `rerank_and_group()` for backward compat. Update all call sites in `hybrid_retrieval.py` and `retriever.py` to `await reranker.rerank_and_group_async(...)`.

**Estimated gain:** Unblocks event loop during reranking. Under load with 3+ concurrent users, this prevents request queuing.

---

### O5: Fix Reranker chunkText vs jdText Field Bug

**File:** `app/services/reranker.py:69`

**Problem:** Reranker always reads `payload["chunkText"]` but JD chunks store text in `payload["jdText"]`.

**Fix:**
```python
# BEFORE:
pairs = [
    (query, chunk.get("payload", {}).get("chunkText", ""))
    for chunk in chunks
]

# AFTER: use same helper as retriever.py
def _get_text(payload: dict) -> str:
    return (payload.get("jdText") or payload.get("chunkText") or "").strip()

pairs = [
    (query, _get_text(chunk.get("payload", {})))
    for chunk in chunks
]
```

**Impact:** JD reranking was silently scoring against empty strings. This fix alone may significantly improve JD ranking quality without any latency change.

---

### O6: LLM Singleton for Query Expansion

**File:** `app/rag/shared/expansion.py`

**Problem:** `_build_flash_llm()` creates a new `ChatGoogleGenerativeAI` instance per expansion call — repeated connection setup overhead.

**Fix:**
```python
# BEFORE:
def _build_flash_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(...)

async def _call_expansion_llm(query, skill_keywords):
    llm = _build_flash_llm()  # new instance every call
    ...

# AFTER: module-level singleton (initialized once at import time)
_EXPANSION_LLM: Optional[ChatGoogleGenerativeAI] = None

def _get_expansion_llm() -> ChatGoogleGenerativeAI:
    global _EXPANSION_LLM
    if _EXPANSION_LLM is None:
        _EXPANSION_LLM = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.1,
            max_output_tokens=512,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _EXPANSION_LLM
```

**Estimated gain:** Eliminates connection overhead (~20–50ms per expansion call).

---

## 5. Priority 2 — Medium-Effort Wins

### O7: Skip Expansion on Turn 2+ (Expansion Result Caching)

**File:** `app/rag/candidate/nodes/expansion.py`  
**File:** `app/rag/candidate/state.py`

**Problem:** Query expansion is called on every JD_SEARCH turn, including Turn 2+, even when the user's context hasn't changed.

**Strategy:** Store `(expanded_query, skill_variants)` in LangGraph state after Turn 1. On Turn 2+, if `pipeline_strategy == JD_SEARCH` and `state["expansion_cache"]` exists with the same topic scope, reuse it.

**State change:**
```python
# state.py — add field:
"expansion_cache": Optional[Dict]  # {"expanded_query": str, "skill_variants": List[str], "turn": int}
```

**Expansion node logic:**
```python
# nodes/expansion.py
cache = state.get("expansion_cache")
query = state["query"]

# Reuse if: cache exists AND query is a follow-up (similar length or refinement)
# Simple heuristic: if query has same extracted skills AND len(query) < 20 words
# A follow-up like "tell me more" or "what about salary?" shouldn't re-expand
if cache and _is_followup_query(query, state.get("query_entities", {})):
    expanded_query = cache["expanded_query"]
    skill_variants = cache["skill_variants"]
    print(f"[Expansion] Cache hit — reusing Turn {cache['turn']} expansion")
else:
    expanded_query, skill_variants = await expand_query(query, skill_keywords)
    state["expansion_cache"] = {
        "expanded_query": expanded_query,
        "skill_variants": skill_variants,
        "turn": state.get("turn_number", 1),
    }

def _is_followup_query(query: str, entities: dict) -> bool:
    words = query.split()
    has_skills = bool(entities.get("skill_keywords"))
    # Short queries without skill mentions are likely follow-ups
    return len(words) <= 12 and not has_skills
```

**Estimated gain:** Saves 500–2000ms on ~60% of Turn 2+ JD_SEARCH requests.

---

### O8: Skip Expansion for Simple Skill Queries

**File:** `app/rag/shared/expansion.py` → `expand_query()`

**Problem:** For queries like `"tìm việc python"` or `"java developer jobs"`, the router's whitelist already extracts `["python"]` or `["java"]`. An LLM call to produce `["Python", "Django", "Flask", "FastAPI"]` adds latency but marginal improvement.

**Heuristic Skip Conditions:**
1. Query ≤ 4 words AND skill_keywords already extracted ≥ 1 skill
2. Query is purely skill-based (matches `^(tìm việc|find.*job[s]?)?\s*<skill>$`)

```python
async def expand_query(query: str, skill_keywords: List[str]) -> Tuple[str, List[str]]:
    # Skip expansion for simple queries — keyword search with router-extracted
    # skills is already sufficient; LLM adds marginal recall at high latency cost.
    if _should_skip_expansion(query, skill_keywords):
        print(f"[Expansion] Skipped — simple query with {len(skill_keywords)} extracted skills")
        return query, skill_keywords

    # ... existing LLM call

def _should_skip_expansion(query: str, skill_keywords: List[str]) -> bool:
    words = query.split()
    return len(words) <= 5 and len(skill_keywords) >= 1
```

**Estimated gain:** Saves 500–2000ms for ~30% of JD_SEARCH queries.

---

### O9: LRU Embedding Cache for Frequent Queries

**File:** `app/services/embedding.py`

**Problem:** Embedding is deterministic. Identical queries (e.g., HR asks about the same position repeatedly) re-run inference.

**Fix:**
```python
from functools import lru_cache
import hashlib

class EmbeddingService:
    # ... existing code ...

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        cache_key = hashlib.md5(f"{text}:{is_query}".encode()).hexdigest()
        return self._embed_cached(cache_key, text, is_query)

    @lru_cache(maxsize=512)
    def _embed_cached(self, cache_key: str, text: str, is_query: bool) -> tuple:
        # lru_cache requires hashable return — use tuple
        vector = self._embed_raw(text, is_query)
        return tuple(vector)
```

Note: Since `lru_cache` requires hashable args and returns, the actual text must be passed. Use `functools.lru_cache` on a wrapper or a manual `dict` cache with `maxsize` eviction.

For production, keep it simple:
```python
from collections import OrderedDict

_EMBED_CACHE: OrderedDict = OrderedDict()
_EMBED_CACHE_MAX = 512

def embed_text(self, text: str, is_query: bool = False) -> List[float]:
    key = f"{is_query}:{text}"
    if key in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(key)  # LRU update
        return _EMBED_CACHE[key]
    vector = self._run_model(text, is_query)
    _EMBED_CACHE[key] = vector
    if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)
    return vector
```

**Estimated gain:** Near-zero latency for repeated queries. High hit rate for HR-mode (same JD query repeated across sessions).

---

### O10: Cache JD Embedding Vector per positionId

**File:** `app/services/retriever.py` → `retrieve_for_hr_mode_hr()`

**Problem:** For every HR query about position X, the system:
1. Fetches JD chunks from Qdrant
2. Concatenates JD text
3. Re-embeds the concatenated text

The JD content only changes when HR uploads a new JD. The embedding is deterministic. This is pure wasted computation.

**Fix — In-memory JD vector cache:**
```python
# app/services/retriever.py (top of file)
from typing import Dict, List
_JD_VECTOR_CACHE: Dict[int, List[float]] = {}  # positionId → embedding vector

# In retrieve_for_hr_mode_hr():
if position_id in _JD_VECTOR_CACHE:
    ranking_vector = _JD_VECTOR_CACHE[position_id]
    print(f"[HR Mode] Using cached JD vector for position {position_id}")
else:
    # ... existing fetch + embed logic ...
    _JD_VECTOR_CACHE[position_id] = ranking_vector

# Cache invalidation: expose a function called when HR uploads new JD
def invalidate_jd_vector_cache(position_id: int):
    _JD_VECTOR_CACHE.pop(position_id, None)
```

Wire invalidation into the `internal.py` route that handles JD updates, or add a dedicated `/internal/cache/invalidate` endpoint.

**Estimated gain:** Saves one full embed call (~50ms) + one Qdrant round trip (~150ms) per HR query = -200ms for every HR Mode request after the first.

---

### O11: Pre-Compile Skill Regex Patterns in Router

**File:** `app/rag/candidate/router.py` → `_extract_skill_keywords()`

**Problem:** For each query, `re.search(rf'\b{re.escape(skill)}\b', query_lower)` is called for all 90+ skills. Python re-compiles each pattern every call (no compiled pattern caching inside the loop).

**Fix — Pre-compile all skill patterns at module load time:**
```python
# Build compiled patterns once at module import
_SKILL_PATTERNS: Dict[str, re.Pattern] = {
    skill: re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
    for skill in _TECH_SKILL_WHITELIST
}

def _extract_skill_keywords(query: str) -> List[str]:
    found: set = set()
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(query):
            found.add(skill)
    return list(found)
```

**Estimated gain:** Minor (2–5ms → <0.5ms). Worth doing for correctness and scalability as whitelist grows.

---

### O12: Verify Qdrant Payload Index Configuration

**File:** `app/services/qdrant.py` or deployment config

**Problem (critical):** Without explicit payload indexes in Qdrant, every filtered vector search scans ALL payload fields linearly before applying HNSW. With thousands of CVs, unindexed filters can increase search time from ~10ms to 500ms+.

**Verify these fields are indexed in both collections:**

For `cv_embeddings`:
```python
# Must be created as payload indexes:
client.create_payload_index(
    collection_name="cv_embeddings",
    field_name="is_latest",
    field_schema=PayloadSchemaType.BOOL,
)
client.create_payload_index("cv_embeddings", "cvId", PayloadSchemaType.INTEGER)
client.create_payload_index("cv_embeddings", "candidateId", PayloadSchemaType.KEYWORD)
client.create_payload_index("cv_embeddings", "positionId", PayloadSchemaType.INTEGER)
client.create_payload_index("cv_embeddings", "sourceType", PayloadSchemaType.KEYWORD)
client.create_payload_index("cv_embeddings", "applied_position_ids", PayloadSchemaType.INTEGER)
```

For `jd_embeddings`:
```python
client.create_payload_index("jd_embeddings", "positionId", PayloadSchemaType.INTEGER)
client.create_payload_index("jd_embeddings", "section", PayloadSchemaType.KEYWORD)
```

Add these to the collection initialization code in `qdrant.py` or as a one-time migration script.

**Estimated gain:** Without indexes: 200–1000ms+ on filtered searches. With indexes: 10–50ms. This is potentially the **single highest-impact optimization** at scale.

---

## 6. Priority 3 — Architectural Improvements

### O13: Tiered Scoring Model Strategy

**File:** `app/rag/candidate/nodes/scoring.py`

**Current:** Gemini 2.5 Pro is used for ALL Turn 1 scoring (2000–5000ms, high cost).

**Proposed tiered approach:**
- **Tier 1 (default):** Gemini 2.5 Flash → fast initial ranking (~500–1500ms, ~10× cheaper)
- **Tier 2 (on-demand):** Gemini 2.5 Pro → triggered only when user asks for deep analysis (`"phân tích chi tiết"`, `"đánh giá kỹ"`) or after applying

**Logic:**
```python
# scoring.py
DEEP_ANALYSIS_PATTERNS = re.compile(
    r"\b(phân tích chi tiết|đánh giá kỹ|deep analysis|detailed review|thoroughly)\b",
    re.IGNORECASE
)

def _select_scoring_model(query: str) -> str:
    if DEEP_ANALYSIS_PATTERNS.search(query):
        return settings.SCORING_GEMINI_MODEL   # Pro
    return settings.GEMINI_MODEL               # Flash
```

**Estimated gain:** -1500–3500ms for ~80% of scoring calls. Also reduces Gemini Pro API costs significantly.

---

### O14: Conversation-Aware Intent Routing

**File:** `app/rag/candidate/nodes/session.py` + `app/rag/candidate/router.py`

**Problem:** Follow-up questions like `"salary?"` or `"what about remote work?"` don't carry the context that user is discussing a specific position found in Turn 1.

**Fix — Thread `last_mentioned_jd_id` through state:**

In `session.py` (load_session_history node), parse the last assistant turn's metadata to extract the most recently discussed `positionId`:
```python
# session.py: extract last mentioned position from metadata
for msg in reversed(history):
    if msg["role"] == "ASSISTANT" and msg.get("functionCall"):
        fc = json.loads(msg["functionCall"])
        if fc.get("scored_jobs"):
            top_job = sorted(fc["scored_jobs"], key=lambda j: j.get("overallScore", 0), reverse=True)
            if top_job:
                state["last_mentioned_jd_id"] = top_job[0]["positionId"]
                break
```

In `router.py`, use `last_mentioned_jd_id` to enrich routing for JD_CONVERSE/JD_ANALYSIS:
```python
# If routing to JD_CONVERSE or JD_ANALYSIS and no explicit jd_id in query,
# default to last_mentioned_jd_id from session
if strategy in ("JD_CONVERSE", "JD_ANALYSIS") and not state.get("jd_id"):
    state["jd_id"] = state.get("last_mentioned_jd_id")
```

**Impact:** Eliminates "I don't have context for that question" responses on natural follow-ups.

---

### O15: Token Streaming Through LangGraph

**File:** `app/rag/candidate/nodes/reasoning.py`  
**File:** `app/api/routes/candidate_chat.py`

**Problem:** The current implementation returns the full LLM response after completion. Users see nothing for 2–5 seconds then the entire response appears.

**Fix — Enable streaming in reasoning node:**
```python
# reasoning.py
async def llm_reasoning_node(state: CandidateChatState) -> CandidateChatState:
    # ... existing tool setup ...
    
    # Stream tokens via generator
    full_response = ""
    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            full_response += chunk.content
            # Emit SSE event if a queue/callback is available in state
            if state.get("stream_callback"):
                await state["stream_callback"](chunk.content)
    
    state["llm_response"] = full_response
    return state
```

**Route change:**
```python
# candidate_chat.py
@router.post("/chatbot/candidate/chat")
async def candidate_chat(request: ChatRequest):
    async def event_generator():
        # Pass SSE queue into graph state
        queue = asyncio.Queue()
        state["stream_callback"] = queue.put_nowait
        
        # Run graph in background
        task = asyncio.create_task(candidate_chatbot.chat(state))
        
        while not task.done():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"data: {json.dumps({'token': token})}\n\n"
            except asyncio.TimeoutError:
                pass
        
        result = await task
        yield f"data: {json.dumps({'done': True, 'metadata': result.metadata})}\n\n"
    
    return EventSourceResponse(event_generator())
```

**Impact:** Perceived latency drops dramatically. User sees tokens after ~200ms instead of waiting 4–5s. This is a UX transformation, not just a performance optimization.

---

### O16: Redis Session Cache for Hot Conversations

**File:** `app/services/recruitment_api.py` → `get_history()`

**Problem:** Every request makes an HTTP call to recruitment-service to load the last 20 messages. For active conversations, this is the same data each turn (plus 2 new rows).

**Fix — Redis TTL cache for session history:**
```python
# recruitment_api.py
import redis.asyncio as aioredis

_redis: Optional[aioredis.Redis] = None

async def get_history(session_id: str) -> List[Dict]:
    cache_key = f"session_history:{session_id}"
    
    # Try cache first
    if _redis:
        cached = await _redis.get(cache_key)
        if cached:
            return json.loads(cached)
    
    # Fetch from recruitment-service
    history = await _fetch_history_from_service(session_id)
    
    # Cache with 10-minute TTL
    if _redis:
        await _redis.setex(cache_key, 600, json.dumps(history))
    
    return history

async def save_message(session_id: str, role: str, content: str, function_call=None):
    # After saving, invalidate cache so next turn gets fresh history
    await _save_message_to_service(session_id, role, content, function_call)
    if _redis:
        await _redis.delete(f"session_history:{session_id}")
```

**Estimated gain:** -80–150ms per request for active sessions (cache hit). Also reduces load on recruitment-service MySQL under concurrent users.

---

## 7. Qdrant HNSW Tuning

The HNSW index parameters control the speed/recall tradeoff for Approximate Nearest Neighbor (ANN) search.

### Current Assumption
Qdrant defaults: `m=16, ef_construct=100, ef=128` (search beam width)

### Recommended Settings

**For `cv_embeddings` (high cardinality, precision matters):**
```python
HnswConfigDiff(
    m=32,              # More connections per node → better recall
    ef_construct=200,  # Better graph at index time (one-time cost)
)
# At search time:
SearchParams(hnsw_ef=128, exact=False)
```

**For `jd_embeddings` (small collection, can afford higher ef):**
```python
HnswConfigDiff(m=16, ef_construct=100)
SearchParams(hnsw_ef=64, exact=False)  # JD collection is small, low ef is fine
```

**Trade-off:** Higher `m` and `ef_construct` increase index build time and memory, but are a one-time cost. Search `hnsw_ef` directly controls search speed vs recall — tune empirically.

---

## 8. Intent Quality Improvements

### IQ-01: Fuzzy Vietnamese Pattern Coverage

**Problem:** Current patterns use exact keywords. Vietnamese has many variations:
- "tìm việc" ≠ "kiếm việc" ≠ "xin việc" (all mean "find a job")
- Typos: "tim viec", "tìm viêc"

**Fix — Extend patterns (low effort, high recall improvement):**
```python
_JD_SEARCH_PATTERN = re.compile(
    r"\b(tìm việc|tim viec|kiếm việc|kiem viec|xin việc|xin viec|"
    r"tìm công việc|tim cong viec|tìm vị trí|tim vi tri|"
    r"việc làm phù hợp|viec lam phu hop|"
    r"công việc phù hợp|cong viec phu hop|"
    r"job.*search|find.*job[s]?|looking.*for.*job|"
    r"có việc nào|co viec nao|gợi ý việc|goi y viec|"
    r"recommend.*job[s]?|job.*recommend)\b",
    re.IGNORECASE,
)
```

### IQ-02: Seniority Level Extraction in Router

**Problem:** Router extracts skill keywords but not seniority level. "Senior Java Developer" and "Junior Java Developer" produce the same `skill_keywords = ["java"]` but should produce different retrieval signals.

**Fix — Add seniority extraction:**
```python
_SENIORITY_PATTERNS = {
    "junior": re.compile(r"\b(junior|intern|entry.?level|fresher|mới ra trường)\b", re.IGNORECASE),
    "mid":    re.compile(r"\b(mid.?level|2.?3 năm|2.?3 years?)\b", re.IGNORECASE),
    "senior": re.compile(r"\b(senior|lead|principal|architect|5\+|nhiều năm)\b", re.IGNORECASE),
}

def _extract_seniority(query: str) -> Optional[str]:
    for level, pattern in _SENIORITY_PATTERNS.items():
        if pattern.search(query):
            return level
    return None
```

Pass `seniority_level` to retriever and use it in Qdrant metadata filter:
```python
FieldCondition(key="seniorityLevel", match=MatchValue(value=seniority_level))
```

### IQ-03: Numeric top_k Extraction from Query

**Problem:** HR asks "tìm top 5 ứng viên" or "show me 10 candidates". The `top_n` parameter must be parsed from the query for HR mode. Verify this is consistently handled in the HR router.

If `top_n` parsing is done with regex in the HR router, ensure it also handles Vietnamese patterns:
```python
_TOP_N_PATTERN = re.compile(
    r"top\s*(\d+)|(\d+)\s*(candidates?|ứng viên|người|profiles?)",
    re.IGNORECASE
)
```

---

## 9. Implementation Roadmap

### Sprint 1 (1–2 days) — P1 Quick Wins
- [ ] O5: Fix reranker `chunkText` vs `jdText` bug ← do this FIRST (silent bug)
- [ ] O3: Eager-load reranker model at startup
- [ ] O6: LLM singleton for expansion
- [ ] O4: Async reranker via run_in_executor
- [ ] O2: Fix synchronous embed_text in retriever.py
- [ ] O11: Pre-compile skill regex patterns

### Sprint 2 (2–3 days) — P2 Medium Wins  
- [ ] O1: Parallelize CV + JD Qdrant searches (requires async Qdrant client or executor)
- [ ] O12: Verify + create Qdrant payload indexes ← run as one-time migration
- [ ] O9: LRU embedding cache
- [ ] O10: JD vector cache per positionId
- [ ] O8: Skip expansion for simple queries

### Sprint 3 (3–5 days) — P2/P3 Architecture
- [ ] O7: Expansion cache across turns (state schema change)
- [ ] O13: Tiered scoring model (Flash → Pro)
- [ ] O14: Conversation-aware routing (last_mentioned_jd_id)
- [ ] IQ-01: Extend Vietnamese intent patterns
- [ ] IQ-02: Seniority extraction in router

### Sprint 4 (5–7 days) — P3 High Value
- [ ] O15: Token streaming through LangGraph + SSE route
- [ ] O16: Redis session cache
- [ ] HNSW parameter tuning (requires Qdrant collection recreation or index update)

---

## 10. Performance Monitoring

After each sprint, measure:

| Metric | Baseline Target | Optimized Target |
|--------|----------------|-----------------|
| JD_SEARCH Turn 1 P50 latency | — | ≤ 6s |
| JD_SEARCH Turn 2+ P50 latency | — | ≤ 4s |
| CV_ANALYSIS P50 latency | — | ≤ 3s |
| Qdrant filtered search time | — | ≤ 50ms (with indexes) |
| Cross-Encoder rerank time | — | ≤ 300ms (CPU) |
| Expansion skip rate | 0% | ≥ 40% |
| Scoring cache hit rate (Turn 2+) | — | ≥ 90% |
| First-request cold start | 5–15s | ≤ 2s |

Add timing instrumentation using Python `time.perf_counter()` per graph node and log to structured JSON logs. This enables bottleneck tracking over time.

```python
# Pattern for all graph nodes:
import time

async def llm_reasoning_node(state):
    t0 = time.perf_counter()
    # ... node logic ...
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[Timing] llm_reasoning: {elapsed_ms:.0f}ms | session={state['session_id']}")
```

---

## 11. Risk Assessment

| Optimization | Risk | Mitigation |
|-------------|------|-----------|
| O1 (parallel searches) | Qdrant connection pool exhaustion under load | Use `AsyncQdrantClient` with connection pooling |
| O7 (expansion cache) | Stale cache if user pivots to completely new job domain | Clear cache on strategy change or high semantic distance |
| O10 (JD vector cache) | Stale vector after HR re-uploads JD | Invalidate via JD update webhook/endpoint |
| O13 (tiered scoring) | Flash model gives lower quality scores → worse guardrail decisions | Keep Pro for APPLY flow; Flash only for browsing/ranking |
| O15 (streaming) | Tool-use turns (finalize_application) can't stream mid-execution | Buffer tool call turns, stream non-tool turns only |
| O16 (Redis cache) | Cache stampede on cold session | Add jitter to TTL + use `SET NX` pattern |
