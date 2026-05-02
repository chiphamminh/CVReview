from typing import List, Dict, Any

def _format_cv_context(
    cv_context: List[Dict[str, Any]],
    cv_id_to_meta: Dict[int, Dict[str, Any]]
) -> str:
    """
    Format retrieved CV chunks for the LLM prompt.

    Groups chunks by cvId (not candidateId — HR CVs have no candidateId).
    Uses cv_id_to_meta to inject the candidate's real name and email into
    each block, so the LLM can answer "Ai tên Minh?" correctly.

    Log the section name of each chunk for debugging.
    """
    if not cv_context:
        return "No CV data found for the current filter criteria."

    # Group chunks by cvId
    cv_chunks: Dict[int, List[str]] = {}
    cv_best_score: Dict[int, float] = {}

    for chunk in cv_context:
        payload = chunk.get("payload", {})
        cv_id   = payload.get("cvId")
        section = payload.get("section", "Unknown")
        text    = payload.get("chunkText", "").strip()
        score   = chunk.get("score", 0.0)

        if cv_id is None:
            continue

        print(f"[Retrieval] CV chunk: cvId={cv_id}, section={section}, score={score:.2f}")

        if cv_id not in cv_chunks:
            cv_chunks[cv_id] = []
            cv_best_score[cv_id] = score

        if text:
            cv_chunks[cv_id].append(f"[{section}]\n{text}")
        if score > cv_best_score[cv_id]:
            cv_best_score[cv_id] = score

    parts = []
    for i, (cv_id, texts) in enumerate(cv_chunks.items(), 1):
        meta  = cv_id_to_meta.get(cv_id, {})
        name  = meta.get("candidateName") or "Unknown"
        email = meta.get("candidateEmail") or "N/A"
        score_val = meta.get("score")
        score_str = f"{score_val}/100" if score_val is not None else "Not scored"

        header = (
            f"--- Candidate #{i}: {name} | Email: {email} | "
            f"AI Score: {score_str} | cvId: {cv_id} | "
            f"Similarity: {cv_best_score[cv_id]:.2f} ---"
        )
        body = "\n".join(texts)
        parts.append(f"{header}\n{body}")

    total = len(cv_chunks)
    return f"[{total} unique candidate(s)]\n\n" + "\n\n".join(parts)


def _format_jd_context(jd_context: List[Dict[str, Any]]) -> str:
    """Format retrieved JD chunks for the LLM prompt (Small-to-Big)."""
    if not jd_context:
        return "No Job Description data found for this position."

    # Extract unique position info (all chunks in a search usually belong to 1 position)
    pos_info = {}
    for chunk in jd_context:
        payload = chunk.get("payload", {})
        pid = payload.get("positionId")
        if pid and pid not in pos_info:
            pos_info[pid] = {
                "name": payload.get("positionName", "Unknown"),
                "chunks": []
            }
        if pid:
            section = payload.get("sectionName", "General")
            text = payload.get("chunkText", "").strip()
            if text:
                pos_info[pid]["chunks"].append(f"[{section}]\n{text}")

    parts = []
    for pid, info in pos_info.items():
        header = f"--- Job Description: {info['name']} (ID: {pid}) ---"
        body = "\n\n".join(info["chunks"])
        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)


def _format_sql_metadata(sql_metadata: List[Dict[str, Any]], mode: str) -> str:
    """
    Format application score rows from SQL as a compact reference table.
    Filters by mode so the LLM only sees relevant records.
    """
    if not sql_metadata:
        return ""

    rows = []
    for app in sql_metadata:
        name  = app.get("candidateName", "N/A")
        email = app.get("candidateEmail", "N/A")
        score = app.get("score", "Not scored")
        cv_id = app.get("appCvId", "N/A")
        rows.append(f"• {name} | Email: {email} | Score: {score} | cvId: {cv_id}")

    return "\n".join(rows)
