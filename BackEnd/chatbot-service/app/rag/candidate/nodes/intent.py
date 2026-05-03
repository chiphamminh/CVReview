"""
Node 1 — Intent classification.

Apply intent detected by Tầng 1 (session node) overrides the classified intent
to `jd_search` so the scoring context pipeline is triggered on the same turn.
"""

from app.rag.candidate.state import CandidateChatState
from app.rag.intent import intent_classifier


def classify_intent_node(state: CandidateChatState) -> CandidateChatState:
    """Classify the candidate's query intent using the shared intent classifier."""
    result = intent_classifier.classify(state["query"])
    print(f"[Intent] {result['intent']} (conf: {result['confidence']:.2f}, domain: {result['domain']})")
    state["intent"] = result["intent"]
    state["intent_confidence"] = result["confidence"]
    state["domain"] = result["domain"]

    # Tầng 1 override: ensure the scoring pipeline runs on apply turns.
    if state.get("is_apply_intent") and state["intent"] not in ("jd_search",):
        state["intent"] = "jd_search"
        print("[Tầng 1] Intent overridden to jd_search for apply flow.")

    return state


def should_retrieve(state: CandidateChatState) -> str:
    """Routing edge: decide if Qdrant retrieval is needed."""
    intent = state["intent"]
    if intent in ("jd_search", "jd_analysis", "cv_analysis"):
        return "retrieve"
    if intent == "general" and state.get("domain") == "general" and state["intent_confidence"] > 0.7:
        return "skip_retrieve"
    return "retrieve"
