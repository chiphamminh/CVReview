# Chatbot Service — Bug Analysis & Resolution Plan

> **Scope:** Post-Sprint-2 test findings across HR Chatbot (HR_MODE) and Candidate Chatbot  
> **Status:** Awaiting implementation

---

## Table of Contents
1. [HR Chatbot Issues](#hr-chatbot-issues)
   - [HR-1] Auto-scoring after RANK/FILTER — Design Question
   - [HR-2] Learning Path shown in HR Chatbot — Wrong Domain
   - [HR-3] Skill Extraction False Positives (Vietnamese words → skills)
   - [HR-4] FIND_MORE misses un-scored CVs
   - [HR-5] ACTION pipeline — "Unknown Position" in prompt
2. [Candidate Chatbot Issues](#candidate-chatbot-issues)
   - [C-1] Expansion timeout → 500 Internal Server Error
3. [Cross-Cutting Concerns](#cross-cutting-concerns)
   - [X-1] active_cv_ids grows unboundedly across turns
   - [X-2] Slow response for SCORE / RANK / FILTER intents

---

## HR Chatbot Issues

---

### [HR-1] Auto-scoring after RANK/FILTER — Design Question

**Observation:**  
Query: *"Lọc top 3 ứng viên phù hợp với vị trí này"*  
→ System retrieves 3 candidates, then immediately invokes the scoring node.

**Is this flow correct?**

**Analysis — Two Valid Schools of Thought:**

| Approach | Pros | Cons |
|---|---|---|
| **Auto-score on RANK/FILTER** (current) | Richer output; HR gets scored results in one turn | Wastes LLM credits when HR is just browsing; latency spike per query |
| **Lazy scoring — only on explicit request** | Cheaper; faster; more predictable | HR has to ask twice ("liệt kê → chấm điểm") |

**Root Cause of the Lấn Cấn:**  
The `evaluate_candidates` tool is documented as *"KHÔNG được gọi trực tiếp bởi LLM"* — intercepted by the graph. But the system prompt says *"Khi HR requests to score/evaluate/chấm điểm candidates, invoke the `evaluate_candidates` tool"*. For a generic query like "lọc top 3", the LLM infers implicit scoring intent and calls the tool — this is a **prompt-driven over-inference**, not intended architecture.

**Fix — Two-Part:**

**Part 1: Tighten system prompt** in `nodes/prompts.py`:

```python
# BEFORE (too broad):
"- When HR requests to score/evaluate/chấm điểm candidates, invoke the `evaluate_candidates` tool.\n"

# AFTER (explicit signal required):
"- ONLY invoke `evaluate_candidates` when HR explicitly uses keywords like: "
"'chấm điểm', 'đánh giá', 'score', 'evaluate', 'rate', 'xếp hạng theo điểm'. "
"For listing, filtering, or ranking queries WITHOUT these keywords, do NOT invoke this tool.\n"
```

**Part 2: Add a pre-check guard in `nodes/reasoning.py`** inside the tool execution loop:

```python
import re as _re
_EXPLICIT_SCORE_SIGNALS = _re.compile(
    r"\b(chấm điểm|cham diem|đánh giá|danh gia|score|evaluate|rate|"
    r"cho điểm|cho diem|xếp hạng theo điểm|ranking by score)\b",
    _re.IGNORECASE
)

# Inside the tool loop, when tool_name == "evaluate_candidates":
if tool_name == "evaluate_candidates":
    if not _EXPLICIT_SCORE_SIGNALS.search(state["query"]):
        print("[Scoring Guard] evaluate_candidates blocked — no explicit scoring signal.")
        continue  # Skip silently; LLM over-inferred intent
    # ... rest of existing intercept logic ...
```

---

### [HR-2] Learning Path shown in HR Chatbot — Wrong Domain

**Observation:**  
HR chatbot scoring responses include `learningPath`, coaching candidates on how to improve. This is a Candidate Chatbot concern and is semantically wrong for an HR recruiter tool.

**Root Cause:**  
`hr_scoring_node` uses a generic scoring prompt that always requests `learningPath`. The prompt does not distinguish between HR and Candidate contexts.

**Fix — Two Layers:**

**Layer 1: Remove `learningPath` from HR scoring prompt** in `nodes/scoring.py`:

```python
scoring_prompt = f"""...
Evaluate this candidate and respond with ONLY a JSON object (no markdown):
{{
  "technicalScore": <integer 0-100>,
  "experienceScore": <integer 0-100>,
  "overallStatus": "<EXCELLENT_MATCH|GOOD_MATCH|POTENTIAL|POOR_FIT>",
  "feedback": "<2-3 sentence HR-facing summary for recruiter use>",
  "skillMatch": "<comma-separated matched skills>",
  "skillMiss": "<comma-separated missing critical skills>"
}}
NOTE: Do NOT include learningPath. This is an HR tool for recruiter decisions, not candidate coaching."""
```

**Layer 2: Remove `learningPath` from formatted output** in `nodes/scoring.py`:

```python
scoring_results.append(
    f"**{name}** {status_icon}\n"
    f"• Kỹ thuật: {score_data.get('technicalScore')}/100 | "
    f"Kinh nghiệm: {score_data.get('experienceScore')}/100\n"
    f"• Nhận xét: {score_data.get('feedback')}\n"
    f"• Phù hợp: {score_data.get('skillMatch')}\n"
    f"• Còn thiếu: {score_data.get('skillMiss')}"
    # learningPath intentionally omitted — HR context only
)
```

---

### [HR-3] Skill Extraction False Positives (Vietnamese words → skills)

**Observation:**  
Query: *"Tôi thấy ứng viên Chí và A rất nổi trội, có ứng viên nào có kinh nghiệm làm việc với Docker không?"*  
Result: `skills = ['docker', 'a', 'kinh']` — two false positives.

**Root Cause Analysis:**

The culprit is the first branch of `_SKILL_KEYWORD_PATTERN`:

```python
r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*|"  # TitleCase phrases
```

Combined with `re.IGNORECASE`, this becomes functionally equivalent to `\b\w+\b` — it matches nearly any word:
- `"A"` → matches `[A-Z][a-z]*` with zero lowercase chars (single letter)
- `"kinh"` → matches after IGNORECASE normalizes "Kinh" to match the pattern

The pattern was designed to catch `"Spring Boot"` but the lack of a minimum character length floor makes it catastrophically broad.

**Fix — Replace with a whitelist-based approach (recommended):**

```python
# In router.py (both HR and Candidate routers) — replace _SKILL_KEYWORD_PATTERN

_TECH_SKILL_WHITELIST = frozenset({
    # Languages
    "java", "python", "javascript", "typescript", "kotlin", "swift",
    "golang", "go", "rust", "c++", "c#", "php", "ruby", "scala",
    # Frameworks
    "spring", "spring boot", "spring mvc", "spring framework",
    "django", "fastapi", "flask", "express", "nestjs",
    "react", "angular", "vue", "next.js", "nuxt",
    "node.js", "nodejs",
    # Data / DB
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "oracle", "sqlite", "dynamodb",
    # DevOps / Infra
    "docker", "kubernetes", "k8s", "kafka", "rabbitmq",
    "aws", "gcp", "azure", "terraform", "ansible",
    "ci/cd", "cicd", "jenkins", "github actions", "gitlab ci",
    # Tools & Practices
    "git", "devops", "agile", "scrum", "microservices", "restful", "graphql",
    # Add more as needed per quarter
})

def _extract_skill_keywords(query: str) -> List[str]:
    """
    Whitelist-based skill extraction — zero false positives.
    Checks individual words AND bigrams (for "spring boot", "node.js" etc).
    """
    tokens = re.findall(r'[\w\./\+#]+', query.lower())
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    
    found: set = set()
    for token in tokens + bigrams:
        if token in _TECH_SKILL_WHITELIST:
            found.add(token)
    return list(found)
```

**Trade-offs:**

| Approach | Precision | Recall | Maintenance |
|---|---|---|---|
| Regex TitleCase (current) | ❌ Very Low | Medium | Low |
| Stricter Regex | Medium | Medium | Medium |
| **Whitelist (recommended)** | ✅ High | Medium | Requires quarterly updates |

IT skill sets are stable enough that a quarterly-maintained whitelist is vastly preferable to regex false positives that corrupt the Qdrant keyword search.

---

### [HR-4] FIND_MORE misses un-scored CVs

**Observation:**  
Query: *"Vậy còn ứng viên nào khác có kinh nghiệm với Python không?"*  
Response references only scored candidates and claims no information on others.

**Root Cause Analysis — Two Bugs, One Symptom:**

**Bug A (LLM reasoning error):** The `sql_metadata` block injected into the prompt lists all candidates. When a candidate is un-scored, their `score` is NULL. The LLM correctly sees NULL score and incorrectly concludes it has "no information about this candidate" — even though Qdrant retrieved their full CV chunk text. The LLM is conflating "no score" with "no data."

**Bug B (active_cv_ids bloat, see [X-1]):** The FIND_MORE exclusion list contains more IDs than the user intended, blocking more candidates from being retrieved.

**The user's diagnosis is correct:**
- ✅ Qdrant retrieval does NOT require a candidate to be scored — it searches CV chunk text directly
- ✅ Scoring is an optional enrichment layer, not a prerequisite for RAG retrieval
- ✅ FIND_MORE should only exclude the IDs from the *last shown result*, not all historically seen IDs

**Fix A — Clarify the prompt for FIND_MORE in `nodes/prompts.py`:**

```python
_STRATEGY_HINTS: dict[str, str] = {
    # ... existing entries ...
    "FIND_MORE": (
        "\n\nINSTRUCTION: These are NEW candidates not previously shown. "
        "Base your assessment STRICTLY on the CV chunks provided in 'CV Data Retrieved' above. "
        "A candidate does NOT need an AI score to be evaluated — read their CV text directly. "
        "Do NOT say you lack information if CV chunks are present for a candidate. "
        "Treat NULL/missing score as 'not yet scored' and assess from CV content alone."
    ),
}
```

**Fix B:** Resolved by [X-1] fix — limiting active_cv_ids to the actual requested top_n prevents over-exclusion.

---

### [HR-5] ACTION pipeline — "Unknown Position" in prompt

**Observation:**  
When `pipeline_strategy == "ACTION"`, the system prompt shows `Position: Unknown Position`.

**Root Cause:**

In `nodes/prompts.py`:
```python
position_name = "Unknown Position"
if state.get("jd_context"):  # ← jd_context = [] for ACTION (Qdrant bypassed)
    for chunk in state["jd_context"]:
        ...
```

ACTION bypasses Qdrant entirely, leaving `jd_context = []`. The position name resolution loop never runs.

**Fix — Add `sql_metadata` as fallback + store `position_name` in scope node:**

**Step 1: Add `position_name` to `HRChatState`** in `state.py`:
```python
position_name: str  # resolved by scope node
```

**Step 2: Resolve in `nodes/scope.py`** during `load_candidate_scope_node`:
```python
# After loading filtered_apps:
if filtered_apps:
    state["position_name"] = filtered_apps[0].get("positionName", f"Position #{state['position_id']}")
else:
    state["position_name"] = f"Position #{state['position_id']}"
```

**Step 3: Use in `nodes/prompts.py`** — replace the JD-context loop:
```python
# Primary: from state (resolved by scope node, always available)
position_name = state.get("position_name", "Unknown Position")

# Secondary: JD context chunks (for RANK/FILTER/FIND_MORE — more specific)
if state.get("jd_context"):
    for chunk in state["jd_context"]:
        name = chunk.get("payload", {}).get("positionName")
        if name:
            position_name = name
            break
```

---

## Candidate Chatbot Issues

---

### [C-1] Expansion timeout → 500 Internal Server Error

**Observation:**
```
[Candidate Expansion] strategy=JD_SEARCH | keywords=['job', 'cv'] → calling LLM
[Expansion] TIMEOUT after 2.0s — using fallback
POST /chatbot/candidate/chat HTTP/1.1 → 500 Internal Server Error
```

**Root Cause Analysis:**

The expansion module has correct fallback logic and does NOT crash — it returns `(original_query, skill_keywords)` on timeout. The 500 originates **downstream of expansion**, most likely in `retrieve_context_node` or `scoring_node`, where an unhandled exception propagates to FastAPI.

**Why only Candidate chatbot?**  
In the HR chatbot, COMPARE / DETAIL / ACTION / AGGREGATE (which don't invoke expansion at all) are common intents. For the Candidate chatbot, **JD_SEARCH is the default catch-all** — every unrecognized query hits the maximum-latency path: expansion → dense embed → Qdrant dense + keyword (parallel) → cross-encoder rerank → Mode A full JD fetch → scoring (Pro model). Under load or on a cold Gemini instance, this chain exceeds 2s just in expansion, then additional downstream calls tip the request over the infrastructure timeout.

**Three-part fix:**

**Fix 1 — Wrap retrieval node in try/except** (`candidate/nodes/retrieval.py`):

```python
async def retrieve_context_node(state: CandidateChatState) -> CandidateChatState:
    try:
        strategy = state.get("pipeline_strategy", "JD_SEARCH")
        # ... existing dispatch ...
    except asyncio.TimeoutError:
        print("[Candidate Retrieve] Qdrant timeout — returning empty context")
        state["cv_context"] = []
        state["jd_context"] = []
        state["retrieval_stats"] = {"error": "retrieval_timeout"}
        return state
    except Exception as e:
        print(f"[Candidate Retrieve] Error: {e}")
        import traceback; traceback.print_exc()
        state["cv_context"] = []
        state["jd_context"] = []
        state["retrieval_stats"] = {"error": str(e)}
        return state
```

**Fix 2 — Increase expansion timeout** (`shared/expansion.py`):

```python
# 2s is too aggressive for a cold Gemini Flash call
_EXPANSION_TIMEOUT_SECONDS = float(getattr(settings, "EXPANSION_TIMEOUT_SECONDS", 4.0))
```

**Fix 3 — Add GENERAL strategy to Candidate Router** to prevent casual queries from hitting the full JD_SEARCH pipeline (`candidate/router.py`):

```python
_GENERAL_PATTERN = re.compile(
    r"^(xin chào|chào|hello|hi|hey|cảm ơn|camon|ok|được rồi|.{1,15})$",
    re.IGNORECASE
)

# Add before the final `else` in route_candidate_intent_node:
elif _GENERAL_PATTERN.match(query.strip()):
    strategy = "GENERAL"
    print("[Candidate Router] P2e → GENERAL (short/greeting query)")

else:
    strategy = "JD_SEARCH"
    ...
```

Map GENERAL to `"general"` legacy intent so the LLM responds with a helpful message without Qdrant retrieval.

---

## Cross-Cutting Concerns

---

### [X-1] active_cv_ids grows unboundedly across turns

**Observation:**
- Turn 1: "top 3 ứng viên" → `active_cv_ids = [12, 14, 16]` ✅
- Turn N: subsequent queries → `active_cv_ids = [12, 13, 14, 15, 16]` ❌

**Root Cause:**

The replacement logic in `retrieve_hr_context_node` looks correct:
```python
state["active_cv_ids"] = new_active_ids  # replaces, not appends
```

But the actual bug: **`top_n` extracted by `_extract_top_n` defaults to `_HR_DEFAULT_TOP_N = 10`** when no explicit number is in the query. A query like "tìm ứng viên có Python" (no explicit count) causes the system to retrieve up to 10 CVs and store all 10 as `active_cv_ids` — even if the user only wanted a summary.

On subsequent turns with different queries, the pool keeps getting repopulated with all available CVs (since there are only 5 in the test set), eventually saturating `active_cv_ids` with every ID.

**Fix — Respect requested_n when updating active_cv_ids** in `hr/nodes/retrieval.py`:

```python
# After hybrid retrieval, in the RANK/FILTER branch:
if strategy != "FIND_MORE":
    requested_n = entities.get("top_n") or _extract_top_n(query)
    # cv_results are already ranked by reranker; slice to requested_n
    top_results = cv_results[:requested_n]
    
    # Deduplicate while preserving rank order
    seen: set = set()
    deduped_ids: List[int] = []
    for chunk in top_results:
        cv_id = chunk.get("payload", {}).get("cvId")
        if cv_id is not None and cv_id not in seen:
            seen.add(cv_id)
            deduped_ids.append(cv_id)
    
    state["active_cv_ids"] = deduped_ids
    print(f"[HR Retrieve] active_cv_ids set to top-{requested_n}: {deduped_ids}")
```

**Additionally, fix FIND_MORE to merge instead of keeping stale IDs:**

```python
# FIND_MORE: append new IDs to existing active_cv_ids
if strategy == "FIND_MORE":
    new_ids = [
        chunk.get("payload", {}).get("cvId")
        for chunk in cv_results
        if chunk.get("payload", {}).get("cvId") is not None
    ]
    # Merge: old (shown before) + new (just found), deduped, order preserved
    merged = list(dict.fromkeys(active_cv_ids + new_ids))
    state["active_cv_ids"] = merged
    print(f"[HR Retrieve] FIND_MORE merged active_cv_ids={merged}")
```

---

### [X-2] Slow response for SCORE / RANK / FILTER intents

**Root Cause Breakdown:**

| Stage | Estimated Latency | Notes |
|---|---|---|
| Query Expansion (LLM Flash) | 500ms – 2s | Cold Gemini start |
| Dense Embedding (BGE) | 50 – 100ms | CPU-bound, run_in_executor |
| Qdrant Dense + Keyword (parallel) | 40 – 100ms | Network RTT |
| Cross-encoder Rerank | 200 – 800ms | CPU-bound, scales with chunk count |
| JD Fetch (Mode A, recruitment-service) | 100 – 300ms | Internal API call |
| LLM Reasoning (Gemini Flash) | 1 – 2s | Standard response |
| **Total — RANK (no scoring)** | ~2 – 5s | Acceptable |
| **HR Scoring (per candidate)** | ~1 – 2s × N | Bottleneck for large pools |
| **Candidate Scoring (Pro model)** | ~3 – 5s × N | Severe bottleneck |

**Optimization Strategies (prioritized by ROI):**

**Quick Wins:**

1. **Reduce default top_n** — `_HR_DEFAULT_TOP_N = 10 → 5`:
   ```python
   # router.py:
   _HR_DEFAULT_TOP_N = 5  # was 10; halves retrieval + reranking cost
   ```

2. **Candidate scoring: use Flash instead of Pro** for initial scoring pass:
   ```python
   # candidate/nodes/scoring.py:
   # Change settings.SCORING_GEMINI_MODEL to the Flash model for Turn 1 scoring
   # Reserve Pro model only when HR/candidate explicitly requests detailed analysis
   ```

3. **Verify cross-encoder cap is applied pre-reranker:**  
   Confirm `_MAX_CHUNKS_PER_CV = 6` in `hybrid_retrieval.py` is correctly applied before the reranker receives input. If the cap silently fails, reranker processes all merged chunks.

**Medium-effort wins:**

4. **Parallelize session + scope loading** in the HR graph entry:
   ```python
   # Current: load_session → load_scope (sequential, ~300–600ms combined)
   # Proposed: combine into a single node using asyncio.gather()
   async def load_session_and_scope_node(state):
       history_task = _load_history(state)
       scope_task   = _load_scope(state)
       await asyncio.gather(history_task, scope_task)
   ```

5. **Cache JD text in state across turns** — for Candidate JD_SEARCH, Mode A fetches the full JD text on Turn 1 and caches it in `scored_jobs`. Verify this cache is correctly restored from `functionCall` in `session.py` so Turn 2+ never re-fetches the JD.

**Sprint 3 scope:**

6. **Stream scoring results** via SSE so HR sees candidates appear progressively instead of waiting for all N to complete.

---

## Summary & Implementation Priority

| ID | Issue | Severity | Effort | Files to Change |
|---|---|---|---|---|
| **HR-3** | Skill extraction false positives | 🔴 Critical | Small | `hr/router.py`, `candidate/router.py` |
| **C-1** | Candidate expansion timeout → 500 | 🔴 Critical | Small | `candidate/nodes/retrieval.py`, `shared/expansion.py`, `candidate/router.py` |
| **X-1** | active_cv_ids unbounded growth | 🔴 High | Small | `hr/nodes/retrieval.py` |
| **HR-5** | Unknown Position in ACTION prompt | 🟠 Medium | Small | `hr/state.py`, `hr/nodes/scope.py`, `hr/nodes/prompts.py` |
| **HR-4** | FIND_MORE misses un-scored CVs | 🟠 Medium | Small | `hr/nodes/prompts.py` (+ X-1 fix) |
| **HR-1** | Auto-scoring on implicit RANK/FILTER | 🟠 Medium | Small | `hr/nodes/reasoning.py`, `hr/nodes/prompts.py` |
| **HR-2** | Learning path shown in HR context | 🟡 Low | Small | `hr/nodes/scoring.py` |
| **X-2** | Slow SCORE / RANK / FILTER | 🟡 Low | Large | Multiple files (Sprint 3) |

**Recommended deployment order:**
```
Phase 1 (Hotfixes — deploy now):
  HR-3 → C-1 → X-1 → HR-5

Phase 2 (Behavioral — next sprint):
  HR-1 → HR-4 → HR-2

Phase 3 (Performance — Sprint 3):
  X-2 (parallel loading, model tier, streaming)
```
