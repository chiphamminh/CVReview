"""
Query Expansion Module — LLM Flash call for synonym/variant generation.

Purpose: Improve hybrid retrieval recall by expanding the user's raw query into:
  - `expanded_query`: a richer string for dense vector embedding
  - `skill_variants`: list of skill synonyms for keyword MatchAny filter on `skills` field

Both outputs serve different roles:
  - expanded_query  → embedded into vector for DENSE search
  - skill_variants  → used as Qdrant MatchAny filter for KEYWORD search

Fallback behaviour (timeout > 2s or any error):
  - expanded_query = original query
  - skill_variants = skill_keywords already extracted by the router
  - Warning is logged; pipeline continues normally
"""

import asyncio
import json
from typing import List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# LLM setup — use the cheapest/fastest model for expansion
# ---------------------------------------------------------------------------

_EXPANSION_TIMEOUT_SECONDS = 2.0

# Structured output prompt — LLM must return pure JSON, no markdown
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


def _build_flash_llm() -> ChatGoogleGenerativeAI:
    """Build a low-temperature LLM instance optimised for structured output generation."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,  # Flash-tier model
        temperature=0.1,
        max_output_tokens=512,
        google_api_key=settings.GEMINI_API_KEY,
    )


async def _call_expansion_llm(query: str, skill_keywords: List[str]) -> Tuple[str, List[str]]:
    """
    Call LLM for query expansion with a hard timeout.
    Returns (expanded_query, skill_variants).
    Raises asyncio.TimeoutError on timeout, Exception on parse failure.
    """
    llm = _build_flash_llm()
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

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)

    expanded_query  = str(parsed.get("expanded_query") or query).strip()
    skill_variants  = [str(s) for s in parsed.get("skill_variants") or []]

    if not expanded_query:
        expanded_query = query

    return expanded_query, skill_variants


async def expand_query(
    query: str,
    skill_keywords: List[str],
) -> Tuple[str, List[str]]:
    """
    Expand HR query into a richer search signal.

    Args:
        query:          The original HR query string.
        skill_keywords: Skills already extracted by the router's entity extraction.

    Returns:
        Tuple of (expanded_query, skill_variants).
        On any failure, falls back gracefully to (original query, skill_keywords).
    """
    try:
        expanded_query, skill_variants = await _call_expansion_llm(query, skill_keywords)

        # Merge router-extracted keywords into skill_variants (dedup, lowercase-safe)
        all_variants = list({s.lower(): s for s in skill_keywords + skill_variants}.values())

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

    # Fallback: original query + router-extracted keywords
    return query, skill_keywords
