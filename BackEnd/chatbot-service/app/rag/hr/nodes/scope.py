from app.rag.hr.state import HRChatState
from app.services.recruitment_api import recruitment_api

async def load_candidate_scope_node(state: HRChatState) -> HRChatState:
    """
    Fetch SQL application metadata for name/email/score display (both modes).
    """
    try:
        applications = await recruitment_api.get_applications(
            position_id=state["position_id"]
        )
        target_source = "HR" if state["mode"] == "HR_MODE" else "CANDIDATE"
        filtered_apps = [app for app in applications if app.get("sourceType") == target_source]

        # Fast cvId → metadata lookup
        state["cv_id_to_meta"] = {}
        for app in filtered_apps:
            if target_source == "CANDIDATE" and app.get("masterCvId") is not None:
                state["cv_id_to_meta"][app["masterCvId"]] = app
            elif app.get("appCvId") is not None:
                state["cv_id_to_meta"][app["appCvId"]] = app

        state["sql_metadata"] = filtered_apps

        print(
            f"[HR Graph] {state['mode']}: {len(filtered_apps)} SQL record(s) loaded "
            f"(for metadata display — Qdrant filter is position-based)."
        )

    except Exception as e:
        print(f"[HR Graph] Could not load candidate scope: {e}")
        state["sql_metadata"] = []
        state["cv_id_to_meta"] = {}

    return state
