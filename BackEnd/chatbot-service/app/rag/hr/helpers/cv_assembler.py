"""
Helper to assemble raw CV chunks into a "Virtual Full CV" for the COMPARE and DETAIL pipelines.
Sections are ordered logically so the LLM gets a coherent document structure.
"""

from typing import List, Dict, Any

# Canonical section display order for "virtual full CV" assembly (COMPARE/DETAIL).
_SECTION_ORDER = ["SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION", "PROJECTS"]


def _normalize_section(section: str) -> str:
    """Normalize section names to match canonical ordering."""
    s = section.upper()
    if s == "PROJECTS" or s.startswith("PROJECT_"):
        return "PROJECTS"
    return s


def assemble_virtual_full_cv(
    raw_chunks: List[Dict[str, Any]], 
    cv_ids: List[int],
    max_chunks_per_cv: int = 12
) -> List[Dict[str, Any]]:
    """
    Groups raw chunks by cvId, sorts each group by the canonical section order, 
    and returns a flattened list of chunks representing virtual full CVs,
    preserving the original cvId order requested.
    """
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for chunk in raw_chunks:
        cv_id = chunk.get("payload", {}).get("cvId")
        if cv_id is not None:
            grouped.setdefault(cv_id, []).append(chunk)

    pinned: List[Dict[str, Any]] = []
    for cv_id in cv_ids:
        raw = grouped.get(cv_id, [])
        if not raw:
            print(f"[CVAssembler] WARNING: cvId={cv_id} not found in scroll results — may be unindexed or deleted")
        ordered = sorted(
            raw,
            key=lambda c: (
                _SECTION_ORDER.index(_normalize_section(c.get("payload", {}).get("section", "")))
                if _normalize_section(c.get("payload", {}).get("section", "")) in _SECTION_ORDER
                else 99
            ),
        )
        pinned.extend(ordered[:max_chunks_per_cv])

    return pinned
