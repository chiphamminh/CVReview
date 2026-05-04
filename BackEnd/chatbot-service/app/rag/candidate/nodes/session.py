"""
Node 0 — Load session history and active positions concurrently.

Restores the scored_jobs cache from the most recent ASSISTANT turn and
performs Tầng 1 hard-rule apply-intent detection so downstream nodes can
skip scoring when a cache hit is available.
"""

import json
import re
import asyncio
from typing import Dict

from app.rag.candidate.state import CandidateChatState
from app.services.recruitment_api import recruitment_api
from app.config import get_settings

settings = get_settings()

_APPLY_PATTERNS = re.compile(
    r"\b(apply|nộp đơn|nop don|ứng tuyển|ung tuyen|finalize|submit.*application"
    r"|i want to apply|giúp tôi apply|help me apply)\b",
    re.IGNORECASE,
)


def _is_apply_intent(query: str) -> bool:
    """Tầng 1: Hard-rule detect apply intent before scoring node."""
    return bool(_APPLY_PATTERNS.search(query))


async def load_session_history_node(state: CandidateChatState) -> CandidateChatState:
    """Fetch conversation history and active positions concurrently.

    Builds `position_ref_map` for downstream O(1) ID-to-name lookups.
    Also restores scoring cache and detects apply intent (Tầng 1) early.
    """

    async def _get_history():
        try:
            history = await recruitment_api.get_history(
                state["session_id"], limit=settings.MAX_HISTORY_TURNS
            )
            # Restore multi-dimensional scored_jobs from the most recent ASSISTANT turn.
            for turn in reversed(history):
                if turn.get("role") == "ASSISTANT":
                    func_data_str = turn.get("functionCall")
                    if func_data_str:
                        try:
                            func_data = json.loads(func_data_str)
                            if isinstance(func_data, dict) and "scored_jobs" in func_data:
                                state["scored_jobs"] = func_data["scored_jobs"]
                                print(
                                    f"[Cache Hit] Restored {len(state['scored_jobs'])} "
                                    f"scored jobs from history."
                                )
                                break
                        except json.JSONDecodeError:
                            continue
            return history
        except Exception as e:
            print(f"[API Error] Could not load history: {e}")
            return []

    async def _get_active_positions():
        try:
            positions = await recruitment_api.get_active_positions()
            return positions, True
        except Exception as e:
            print(f"[API Error] Could not load active positions: {e}")
            return [], False

    history_result, positions_result = await asyncio.gather(
        _get_history(),
        _get_active_positions(),
    )

    positions, positions_ok = positions_result
    state["conversation_history"] = history_result

    if positions_ok:
        state["active_position_ids"] = [p["id"] for p in positions]
    else:
        state["active_position_ids"] = None
        print(
            "[Session] WARNING: active_position_ids set to None "
            "(get_active_positions failed) — active-position guard will be skipped"
        )

    ref_map: Dict[int, str] = {}
    for p in positions:
        parts = [p.get("name", ""), p.get("language", ""), p.get("level", "")]
        label = " ".join(part for part in parts if part)
        ref_map[p["id"]] = label
    state["position_ref_map"] = ref_map

    state["is_apply_intent"] = _is_apply_intent(state["query"])
    if state["is_apply_intent"]:
        print("[Tầng 1] Apply intent detected via hard-rule — will skip scoring if cache hit.")

    return state