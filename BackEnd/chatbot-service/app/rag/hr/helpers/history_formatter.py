from typing import List, Dict, Any

def _format_history(history: List[Dict[str, Any]]) -> str:
    """Convert DB history rows into a compact conversation transcript."""
    if not history:
        return ""
    lines = []
    for msg in history:
        role_label = "HR" if msg.get("role") == "USER" else "Assistant"
        lines.append(f"{role_label}: {msg.get('content', '')}")
    return "\n".join(lines)
