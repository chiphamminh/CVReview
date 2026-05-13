"""
Node: Query Expansion for HR Chatbot.

Wraps the shared expand_query() core logic (app/rag/shared/expansion.py).
Only triggers for strategies that involve Qdrant retrieval with vocabulary
mismatch risk: {RANK, FILTER, FIND_MORE}.

All other strategies (COMPARE, DETAIL, ACTION, AGGREGATE) bypass this node
because they either:
  - Don't touch Qdrant at all (ACTION, AGGREGATE)
  - Use pinned cvId fetch where expansion adds zero value (COMPARE, DETAIL)
"""

from typing import List

from app.rag.hr.state import HRChatState
from app.rag.shared.expansion import expand_query

# Strategies that benefit from LLM query expansion before Qdrant retrieval.
# Matches plan §4: "Trigger chỉ khi intent thuộc {RANK, FILTER, FIND_MORE, JD_SEARCH}"
_EXPANSION_STRATEGIES = {"RANK", "FILTER", "FIND_MORE"}


async def query_expansion_node(state: HRChatState) -> HRChatState:
    """
    LangGraph node that runs query expansion for RANK / FILTER / FIND_MORE strategies.

    Skips expansion for all other strategies and writes passthrough values so
    retrieve_hr_context_node always has `expanded_query` and `skill_variants`
    available regardless of which path was taken.

    Reads from state:
      - pipeline_strategy: routing decision from route_hr_intent_node
      - query:             original HR query
      - query_entities:    dict containing skill_keywords extracted by router

    Writes to state:
      - expanded_query:  enriched query string for dense vector embedding
      - skill_variants:  merged + deduped skill synonyms for Qdrant MatchAny filter
    """
    strategy: str       = state.get("pipeline_strategy", "RANK")
    query: str          = state["query"]
    skill_keywords: List[str] = state.get("query_entities", {}).get("skill_keywords", [])

    if strategy not in _EXPANSION_STRATEGIES:
        state["expanded_query"] = query
        state["skill_variants"] = skill_keywords
        print(f"[HR Expansion] strategy={strategy} → skipped (not in expansion strategies)")
        return state

    # Skip LLM when no skill keywords: keyword search leg returns 0 results regardless,
    # so expansion produces no retrieval benefit and only risks the timeout penalty.
    if not skill_keywords:
        state["expanded_query"] = query
        state["skill_variants"] = []
        print(f"[HR Expansion] strategy={strategy} | keywords=[] → skipped (no keyword leg benefit)")
        return state

    # F8: Skip expansion for simple queries — short query with explicit skill keywords
    # already provides a precise signal; LLM synonyms add noise more than recall.
    if len(query.split()) <= 5 and len(skill_keywords) >= 1:
        state["expanded_query"] = query
        state["skill_variants"] = skill_keywords
        print(f"[HR Expansion] Simple query skip (≤5 words + skills) → passthrough")
        return state

    print(f"[HR Expansion] strategy={strategy} | keywords={skill_keywords} → calling LLM")
    expanded_query, skill_variants = await expand_query(query, skill_keywords)

    state["expanded_query"] = expanded_query
    state["skill_variants"] = skill_variants
    return state