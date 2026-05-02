from typing import List, Dict, Any

def _resolve_candidates_by_name(
    name_query: str,
    sql_metadata: List[Dict[str, Any]],
    mode: str
) -> List[Dict[str, Any]]:
    """
    Return ALL records matching a partial name query — used for email disambiguation.
    Filters by mode so HR_MODE only searches HR-uploaded CVs.
    """
    if not name_query or not sql_metadata:
        return []

    query_lower = name_query.lower().strip()

    return [
        app for app in sql_metadata
        if query_lower in (app.get("candidateName") or "").lower()
    ]
