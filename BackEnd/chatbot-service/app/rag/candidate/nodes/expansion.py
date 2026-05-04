"""
Node: Query Expansion for Candidate Chatbot.

Wraps the shared expand_query() core logic (app/rag/shared/expansion.py).
Only triggers for JD_SEARCH strategy where vocabulary expansion improves
JD recall. All other strategies skip and receive passthrough values.

Expansion strategies for Candidate:
  JD_SEARCH — expand candidate's query into job-role synonyms + skill variants
               so the hybrid JD retrieval catches more relevant positions.
"""

from typing import List

from app.rag.candidate.state import CandidateChatState
from app.rag.shared.expansion import expand_query

# Only JD_SEARCH benefits from expansion on the candidate side.
# JD_ANALYSIS / JD_CONVERSE use specific JD chunks (no vocabulary mismatch issue).
# CV_ANALYSIS searches only within the candidate's own CV (no expansion needed).
_EXPANSION_STRATEGIES = {"JD_SEARCH"}


async def query_expansion_node(state: CandidateChatState) -> CandidateChatState:
    """
    LangGraph node that runs query expansion for JD_SEARCH strategy.

    Skips for all other strategies and writes passthrough values so downstream
    retrieval_node always has expanded_query and skill_variants available.

    Writes to state:
      - expanded_query:  enriched query string for dense JD embedding
      - skill_variants:  merged skill synonyms for keyword MatchAny filter
    """
    strategy: str       = state.get("pipeline_strategy", "JD_SEARCH")
    query: str          = state["query"]
    skill_keywords: List[str] = state.get("query_entities", {}).get("skill_keywords", [])

    if strategy not in _EXPANSION_STRATEGIES:
        state["expanded_query"] = query
        state["skill_variants"] = skill_keywords
        print(f"[Candidate Expansion] strategy={strategy} → skipped")
        return state

    print(f"[Candidate Expansion] strategy={strategy} | keywords={skill_keywords} → calling LLM")
    expanded_query, skill_variants = await expand_query(query, skill_keywords)

    state["expanded_query"] = expanded_query
    state["skill_variants"] = skill_variants
    return state
