"""
Query Expansion Module — LLM Flash call for synonym/variant generation.

Purpose: Improve hybrid retrieval recall by expanding the user's raw query into:
  - `expanded_query`: a richer string for dense vector embedding
  - `skill_variants`: list of skill synonyms for keyword MatchAny filter on `skills` field

Both outputs serve different roles:
  - expanded_query  → embedded into vector for DENSE search
  - skill_variants  → used as Qdrant MatchAny filter for KEYWORD search

Fallback behaviour (timeout > 4s or any error):
  - expanded_query = original query
  - skill_variants = skill_keywords already extracted by the router
  - Warning is logged; pipeline continues normally

Note: Dedup skill_variants preserves original casing (first-seen wins).
  Previously: {s.lower(): s for s in ...}.values() kept the LAST seen casing per key,
  which could downcase "Java" → "java" and break case-sensitive Qdrant MatchAny.
  Now: first-seen casing is preserved and all variants are passed through unchanged.
"""

import asyncio
import json
from typing import List, Tuple

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# LLM setup — use the cheapest/fastest model for expansion
# ---------------------------------------------------------------------------

_EXPANSION_TIMEOUT_SECONDS = float(getattr(settings, "EXPANSION_TIMEOUT_SECONDS", 4.0))

# Module-level singleton — avoids re-initialising the LLM client on every expansion call
_LLM: Optional[ChatGoogleGenerativeAI] = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.1,
            max_output_tokens=512,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _LLM


_EXPANSION_PROMPT_TEMPLATE = """\
You are a technical recruitment assistant.

Given the HR's query and extracted skill keywords, expand them to improve search recall.

## HR Query:
{query}

## Extracted Skill Keywords:
{skill_keywords}

## Task:
Return a JSON object with exactly two keys:
1. "expanded_query": A single enriched string that adds job role synonyms, \
related technologies, and seniority terms. Keep it under 100 words. \
Write in the SAME language as the original query.
2. "skill_variants": A JSON array of skill name variants and synonyms (English only). \
Include abbreviations, framework names, and common alternate spellings. \
Maximum 20 items.

## Rules:
- Output ONLY the JSON object. No markdown fences, no explanations.
- expanded_query must NOT be empty.
- If no skills are present, skill_variants should be an empty array [].

## Example output:
{{"expanded_query": "senior backend developer Java Spring Boot microservices REST API",\
 "skill_variants": ["Java", "Spring Boot", "Spring Framework", "Spring MVC", "JPA", "Hibernate", "REST", "Microservices"]}}"""


async def _call_expansion_llm(query: str, skill_keywords: List[str]) -> Tuple[str, List[str]]:
    """
    Call LLM for query expansion with a hard timeout.
    Returns (expanded_query, skill_variants).
    Raises asyncio.TimeoutError on timeout, Exception on parse failure.
    """
    llm = _get_llm()
    prompt = _EXPANSION_PROMPT_TEMPLATE.format(
        query=query,
        skill_keywords=", ".join(skill_keywords) if skill_keywords else "(none)",
    )

    response = await asyncio.wait_for(
        llm.ainvoke([HumanMessage(content=prompt)]),
        timeout=_EXPANSION_TIMEOUT_SECONDS,
    )

    raw = response.content if isinstance(response.content, str) else str(response.content)
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)

    expanded_query = str(parsed.get("expanded_query") or query).strip()
    skill_variants = [str(s) for s in parsed.get("skill_variants") or []]

    if not expanded_query:
        expanded_query = query

    return expanded_query, skill_variants


def _dedup_preserve_casing(items: List[str]) -> List[str]:
    """
    Deduplicate a list of strings case-insensitively, preserving the FIRST seen casing.

    The old approach {s.lower(): s for s in items}.values() kept the
    LAST seen value per lowercase key. This could silently downcase "Java" → "java"
    when the LLM returned "java" after the router had already extracted "Java".
    Qdrant MatchAny is case-sensitive, so losing correct casing causes keyword
    search misses against payloads that store skills as "Java", "Spring Boot", etc.

    This implementation keeps first-seen casing and is O(n).
    """
    seen_lower: set = set()
    result: List[str] = []
    for s in items:
        key = s.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            result.append(s)
    return result


async def expand_query(
    query: str,
    skill_keywords: List[str],
) -> Tuple[str, List[str]]:
    """
    Expand HR/Candidate query into a richer search signal.

    Args:
        query:          The original query string.
        skill_keywords: Skills already extracted by the router's entity extraction.

    Returns:
        Tuple of (expanded_query, skill_variants).
        On any failure, falls back gracefully to (original query, skill_keywords).
    """
    try:
        expanded_query, llm_variants = await _call_expansion_llm(query, skill_keywords)

        # Merge router-extracted keywords (first) + LLM variants (second).
        # skill_keywords go first so their casing (from the raw query) wins on dedup.
        all_variants = _dedup_preserve_casing(skill_keywords + llm_variants)

        print(
            f"[Expansion] OK | expanded_query='{expanded_query[:80]}...'"
            f" | variants={len(all_variants)}"
        )
        return expanded_query, all_variants

    except asyncio.TimeoutError:
        print(f"[Expansion] TIMEOUT after {_EXPANSION_TIMEOUT_SECONDS}s — using fallback")
    except json.JSONDecodeError as e:
        print(f"[Expansion] JSON parse error: {e} — using fallback")
    except Exception as e:
        print(f"[Expansion] Unexpected error: {e} — using fallback")

    return query, skill_keywords