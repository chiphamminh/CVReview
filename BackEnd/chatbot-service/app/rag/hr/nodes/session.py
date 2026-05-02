import json
import re
from app.rag.hr.state import HRChatState
from app.services.recruitment_api import recruitment_api
from app.config import get_settings

settings = get_settings()

_CV_COUNT_PATTERN = re.compile(
    r"\b(bao nhiêu cv|how many cv|bao nhieu cv|số lượng cv|so luong cv|"
    r"upload bao nhiêu|da upload|tổng số cv|tong so cv|"
    r"how many resume|total cv|cv count|cv statistics|cv stat)\b",
    re.IGNORECASE,
)

async def load_hr_session_history_node(state: HRChatState) -> HRChatState:
    """Fetch the sliding window of the last N turns."""
    try:
        # Cập nhật query không limit (hoặc limit lớn) để lấy functionCall gần nhất
        history = await recruitment_api.get_history(
            session_id=state["session_id"],
            limit=100
        )
        
        # Limit history given to LLM to avoid prompt explosion
        state["conversation_history"] = history[-settings.MAX_HISTORY_TURNS:] if history else []

        # Restore session cache from the most recent ASSISTANT turn
        for turn in reversed(history):
            if turn.get("role") == "ASSISTANT":
                func_data_str = turn.get("functionCall")
                if func_data_str:
                    try:
                        func_data = json.loads(func_data_str)
                        if not isinstance(func_data, dict):
                            break
                        if "pending_emails" in func_data:
                            state["pending_emails"] = func_data["pending_emails"]
                            print(f"[Cache Hit] Restored {len(state['pending_emails'])} pending_email(s)")
                        # ISSUE-01: restore active_cv_ids so follow-up COMPARE/DETAIL can pin-fetch
                        if "active_cv_ids" in func_data:
                            state["active_cv_ids"] = func_data["active_cv_ids"]
                            print(f"[Cache Hit] Restored active_cv_ids: {state['active_cv_ids']}")
                    except json.JSONDecodeError:
                        pass
                break
    except Exception as e:
        print(f"[HR Graph] Could not load history: {e}")
        state["conversation_history"] = []

    return state
