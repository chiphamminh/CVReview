# """
# LangGraph workflow for the Candidate chatbot.
# Pipeline: Intent Classification → Parallel Retrieval → Multi-Dimensional Scoring → Prompt Build → LLM → Persist → Response

# Phase 4 changes:
# - scoring_node: Updated to produce multi-dimensional JSON (technicalScore, experienceScore,
#   overallStatus, skillMatch, skillMiss, feedback, learningPath).
# - llm_reasoning_node: finalize_application guardrail updated — blocks on POOR_FIT status,
#   allows EXCELLENT_MATCH / GOOD_MATCH / POTENTIAL.
# - candidate_tools.finalize_application updated to pass multi-score fields to Java.
# """

# import json
# import re
# import asyncio
# from typing import TypedDict, Literal, Optional, List, Dict, Any
# from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# from app.rag.intent import intent_classifier
# from app.rag.prompts import get_prompt_for_intent, build_cv_context, build_jd_context
# from app.services.retriever import retriever
# from app.services.recruitment_api import recruitment_api
# from app.rag.candidate_tools import CANDIDATE_TOOLS
# from app.config import get_settings

# settings = get_settings()

# # ---------------------------------------------------------------------------
# # Apply intent — hard-rule keywords (Tầng 1 routing, zero LLM cost)
# # ---------------------------------------------------------------------------

# _APPLY_PATTERNS = re.compile(
#     r"\b(apply|nộp đơn|nop don|ứng tuyển|ung tuyen|finalize|submit.*application|i want to apply|giúp tôi apply|help me apply)\b",
#     re.IGNORECASE,
# )

# # MatchStatus values that allow application submission
# _APPLY_ALLOWED_STATUSES = {"EXCELLENT_MATCH", "GOOD_MATCH", "POTENTIAL"}


# def _is_apply_intent(query: str) -> bool:
#     """Tầng 1: Hard-rule detect apply intent before scoring node."""
#     return bool(_APPLY_PATTERNS.search(query))


# class ChatState(TypedDict):
#     """State passed between graph nodes."""

#     # Input
#     session_id: str
#     query: str
#     candidate_id: str
#     cv_id: Optional[int]
#     jd_id: Optional[int]

#     # Session
#     conversation_history: List[Dict[str, Any]]
#     active_position_ids: List[int]
#     position_ref_map: Dict[int, str]

#     # Processing
#     intent: Literal["jd_search", "jd_analysis", "cv_analysis", "general"]
#     intent_confidence: float
#     domain: str
#     is_apply_intent: bool

#     # Retrieved context
#     cv_context: List[Dict[str, Any]]
#     jd_context: List[Dict[str, Any]]
#     retrieval_stats: Dict[str, Any]

#     # Multi-dimensional pre-screening scores
#     scored_jobs: Optional[List[Dict[str, Any]]]

#     # LLM pipeline
#     system_prompt: str
#     user_prompt: str
#     llm_response: str
#     function_calls: Optional[List[Dict[str, Any]]]

#     # Output
#     final_answer: str
#     metadata: Dict[str, Any]


# def get_temperature_for_intent(intent: str) -> float:
#     temperatures = {
#         "cv_analysis": 0.2,
#         "jd_analysis": 0.25,
#         "jd_search": 0.3,
#         "general": 0.4,
#     }
#     return temperatures.get(intent, 0.3)


# def _extract_llm_text(content: Any) -> str:
#     """Normalise LLM response content regardless of whether it is a str or a list of blocks."""
#     if isinstance(content, list):
#         return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
#     return str(content)


# # ---------------------------------------------------------------------------
# # Node 0 — Load session history + active positions (concurrent)
# # ---------------------------------------------------------------------------

# async def load_session_history_node(state: ChatState) -> ChatState:
#     """Fetch conversation history and active positions concurrently.
#     Builds `position_ref_map` for downstream O(1) ID-to-name lookups.
#     Also restores scoring cache and detects apply intent (Tầng 1) early.
#     """

#     async def _get_history():
#         try:
#             history = await recruitment_api.get_history(state["session_id"], limit=settings.MAX_HISTORY_TURNS)

#             # Restore multi-dimensional scored_jobs from the most recent ASSISTANT turn
#             for turn in reversed(history):
#                 if turn.get("role") == "ASSISTANT":
#                     func_data_str = turn.get("functionCall")
#                     if func_data_str:
#                         try:
#                             func_data = json.loads(func_data_str)
#                             if isinstance(func_data, dict) and "scored_jobs" in func_data:
#                                 state["scored_jobs"] = func_data["scored_jobs"]
#                                 print(f"[Cache Hit] Restored {len(state['scored_jobs'])} scored jobs from history.")
#                                 break
#                         except json.JSONDecodeError:
#                             continue

#             return history
#         except Exception as e:
#             print(f"[API Error] Could not load history: {e}")
#             return []

#     async def _get_active_positions():
#         try:
#             return await recruitment_api.get_active_positions()
#         except Exception as e:
#             print(f"[API Error] Could not load active positions: {e}")
#             return []

#     history, positions = await asyncio.gather(_get_history(), _get_active_positions())
#     state["conversation_history"] = history
#     state["active_position_ids"] = [p["id"] for p in positions]

#     ref_map: Dict[int, str] = {}
#     for p in positions:
#         parts = [p.get("name", ""), p.get("language", ""), p.get("level", "")]
#         label = " ".join(part for part in parts if part)
#         ref_map[p["id"]] = label
#     state["position_ref_map"] = ref_map

#     state["is_apply_intent"] = _is_apply_intent(state["query"])
#     if state["is_apply_intent"]:
#         print("[Tầng 1] Apply intent detected via hard-rule — will skip scoring if cache hit.")

#     return state


# # ---------------------------------------------------------------------------
# # Node 1 — Intent classification
# # ---------------------------------------------------------------------------

# def classify_intent_node(state: ChatState) -> ChatState:
#     """Classify the candidate's query intent.
#     Apply intent detected in Tầng 1 overrides to jd_search so scoring context is available.
#     """
#     result = intent_classifier.classify(state["query"])
#     print(f"[Intent] {result['intent']} (conf: {result['confidence']:.2f}, domain: {result['domain']})")
#     state["intent"] = result["intent"]
#     state["intent_confidence"] = result["confidence"]
#     state["domain"] = result["domain"]

#     if state.get("is_apply_intent") and state["intent"] not in ("jd_search",):
#         state["intent"] = "jd_search"
#         print("[Tầng 1] Intent overridden to jd_search for apply flow.")

#     return state


# # ---------------------------------------------------------------------------
# # Node 2 — Retrieve context (Dual-mode JD retrieval with Reranking)
# # ---------------------------------------------------------------------------

# async def retrieve_context_node(state: ChatState) -> ChatState:
#     """
#     Node 2 — Dual-mode JD retrieval strategy based on scoring cache.

#     Reranker is integrated inside retriever — JD chunks are fetched in bulk
#     from Qdrant, reranked by Cross-Encoder, then grouped by positionId (Max Score).

#     Mode A — No cache (Turn 1): Fetch full JD text after reranking for scoring precision.
#     Mode B — Cache exists (Turn 2+): Use reranked section chunks directly.
#     """
#     intent = state["intent"]
#     has_scoring_cache = bool(state.get("scored_jobs"))

#     result = await retriever.retrieve_for_intent(
#         query=state["query"],
#         intent=intent,
#         candidate_id=state.get("candidate_id"),
#         cv_id=state.get("cv_id"),
#         jd_id=state.get("jd_id"),
#         active_jd_ids=state.get("active_position_ids") if intent == "jd_search" else None,
#     )

#     cv_context = result.get("cv_context", [])
#     chunk_hits  = result.get("jd_context", [])
#     jd_context  = chunk_hits

#     if intent in ("jd_search", "jd_analysis") and chunk_hits:
#         seen: set = set()
#         position_ids: List[int] = []
#         for hit in chunk_hits:
#             pid = hit.get("payload", {}).get("positionId")
#             if pid is not None and pid not in seen:
#                 seen.add(pid)
#                 position_ids.append(pid)

#         if not has_scoring_cache and position_ids:
#             print(f"[Retriever] Mode A (no cache): fetching full JD for {len(position_ids)} positions")
#             try:
#                 full_jd_list = await recruitment_api.get_position_details(position_ids)
#                 jd_context = [
#                     {
#                         "score": 1.0,
#                         "payload": {
#                             "positionId":   jd["id"],
#                             "positionName": jd.get("name", ""),
#                             "language":     jd.get("language", ""),
#                             "level":        jd.get("level", ""),
#                             "jdText":       jd.get("jdText", ""),
#                         },
#                     }
#                     for jd in full_jd_list
#                     if jd.get("jdText")
#                 ]
#                 print(f"[Retriever] Mode A: Expanded to {len(jd_context)} full-JD objects")
#             except Exception as e:
#                 print(f"[Retriever] Mode A: Full JD fetch failed, falling back to chunks: {e}")
#         else:
#             jd_context = chunk_hits
#             reason = "scoring cache hit" if has_scoring_cache else "no position IDs"
#             print(f"[Retriever] Mode B ({reason}): using {len(jd_context)} section chunks")

#         # Python-side active-position guard (belt-and-suspenders)
#         if intent == "jd_search" and state.get("active_position_ids"):
#             active_ids = set(state["active_position_ids"])
#             jd_context = [
#                 jd for jd in jd_context
#                 if jd.get("payload", {}).get("positionId") in active_ids
#             ]

#     state["cv_context"]      = cv_context
#     state["jd_context"]      = jd_context
#     state["retrieval_stats"] = result.get("retrieval_stats", {})
#     return state


# # ---------------------------------------------------------------------------
# # Node 2.5 — Deep CV-JD scoring with multi-dimensional schema
# # ---------------------------------------------------------------------------

# async def scoring_node(state: ChatState) -> ChatState:
#     """
#     Node 2.5 — Multi-dimensional CV-JD scoring using the dedicated Pro model.

#     Runs only for jd_search intent on Turn 1. Produces:
#       technicalScore, experienceScore, overallStatus, skillMatch, skillMiss, feedback, learningPath.

#     overallStatus follows the MatchStatus enum:
#       EXCELLENT_MATCH (≥85 tech + ≥80 exp) | GOOD_MATCH (≥70 + ≥65) | POTENTIAL | POOR_FIT
#     """
#     if state["intent"] != "jd_search" or not state["jd_context"] or not state["cv_context"]:
#         state["scored_jobs"] = state.get("scored_jobs")
#         return state

#     if state.get("scored_jobs"):
#         print("[Scoring] Cache hit — bypassing LLM scoring call.")
#         return state

#     print(f"[Scoring] Running multi-dimensional scoring with model: {settings.SCORING_GEMINI_MODEL}...")

#     cv_profile = build_cv_context(state["cv_context"])

#     jds_block = ""
#     for jd in state["jd_context"]:
#         payload  = jd.get("payload", {})
#         jd_id    = payload.get("positionId", "unknown")
#         jd_title = " ".join(filter(None, [
#             payload.get("positionName"),
#             payload.get("language"),
#             payload.get("level"),
#         ])) or payload.get("positionName", "Unknown Position")
#         jd_text  = payload.get("jdText", "")
#         jds_block += f"\n[JD ID: {jd_id} | Title: {jd_title}]\n{jd_text}\n"

#     scoring_prompt = f"""You are a strict HR scoring system. Score each CV against the provided JDs.

# SCORING SYSTEM:

# 1. Technical Score (0–100):
#    - Start: 100 pts.
#    - Deduct 8 pts for each missing REQUIRED skill explicitly stated in JD.
#    - Deduct 12 pts if candidate has < 70% of required tech stack overall.

# 2. Experience Score (0–100):
#    - 85-100: Led architectural decisions, measurable business impact, system design ownership.
#    - 65-84:  Mid-level, implemented features with rationale, understands trade-offs.
#    - 40-64:  Junior/basic CRUD projects, limited scale or architectural thinking.
#    - 0-39:   No relevant professional experience.

# 3. Overall Status (derive from scores above):
#    - EXCELLENT_MATCH: technicalScore >= 85 AND experienceScore >= 80
#    - GOOD_MATCH:      technicalScore >= 70 AND experienceScore >= 65
#    - POTENTIAL:       technicalScore >= 55 OR experienceScore >= 55 (but not GOOD_MATCH)
#    - POOR_FIT:        All other cases

# RULES:
# - STRICT: If a skill is not explicitly stated in CV, assume candidate does NOT have it.
# - NO HALLUCINATION: Do not infer skills from project context alone.
# - HIERARCHICAL SKILL INFERENCE: If candidate is overqualified for a lower-level role, do NOT penalize. Score highly. Set feedback to "Overqualified".
# - learningPath: Provide ONLY for POTENTIAL or POOR_FIT. For POOR_FIT, this is MANDATORY. For EXCELLENT_MATCH or GOOD_MATCH, set to null.

# CV:
# {cv_profile}

# JDs to score:
# {jds_block}

# Return EXACTLY this JSON array (no markdown, no preamble):
# [
#   {{
#     "positionId": <integer>,
#     "technicalScore": <0-100>,
#     "experienceScore": <0-100>,
#     "overallStatus": "<EXCELLENT_MATCH|GOOD_MATCH|POTENTIAL|POOR_FIT>",
#     "feedback": "<1 concise HR-tone sentence>",
#     "skillMatch": ["<skill>", "<skill>"],
#     "skillMiss": ["<skill>", "<skill>"],
#     "learningPath": "<90-day roadmap string or null>"
#   }}
# ]"""

#     llm = ChatGoogleGenerativeAI(
#         model=settings.SCORING_GEMINI_MODEL,
#         temperature=0.0,
#         max_output_tokens=1500,
#         google_api_key=settings.GEMINI_API_KEY,
#     )

#     try:
#         response = await llm.ainvoke([HumanMessage(content=scoring_prompt)])
#         raw = response.content.strip()
#         if raw.startswith("```"):
#             raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
#         state["scored_jobs"] = json.loads(raw)
#         print(f"[Scoring] Scored {len(state['scored_jobs'])} positions (multi-dimensional).")
#     except Exception as e:
#         print(f"[Scoring Error] JSON parse failed: {e} — proceeding without scores.")
#         state["scored_jobs"] = None

#     return state


# # ---------------------------------------------------------------------------
# # Node 3 — Build prompts
# # ---------------------------------------------------------------------------

# def build_prompts_node(state: ChatState) -> ChatState:
#     """Assemble the system + user prompts, including multi-dimensional scores for jd_search."""
#     system_prompt, user_prompt = get_prompt_for_intent(
#         intent=state["intent"],
#         query=state["query"],
#         cv_context=state.get("cv_context", []),
#         jd_context=state.get("jd_context", []),
#         conversation_history=state.get("conversation_history", []),
#         scored_jobs=state.get("scored_jobs"),
#     )
#     state["system_prompt"] = system_prompt
#     state["user_prompt"] = user_prompt
#     return state


# # ---------------------------------------------------------------------------
# # Node 4 — LLM reasoning with tool support
# # ---------------------------------------------------------------------------

# async def llm_reasoning_node(state: ChatState) -> ChatState:
#     """Single LLM call that generates the full candidate-facing response."""
#     intent = state["intent"]
#     temperature = get_temperature_for_intent(intent)

#     llm = ChatGoogleGenerativeAI(
#         model=settings.GEMINI_MODEL,
#         temperature=temperature,
#         max_output_tokens=settings.GEMINI_MAX_TOKENS,
#         google_api_key=settings.GEMINI_API_KEY,
#     ).bind_tools(CANDIDATE_TOOLS)

#     messages = [
#         SystemMessage(content=state["system_prompt"]),
#         HumanMessage(content=state["user_prompt"]),
#     ]

#     response = await llm.ainvoke(messages)
#     state["llm_response"] = _extract_llm_text(response.content)
#     state["function_calls"] = None

#     if not response.tool_calls:
#         return state

#     print("[LLM] Tool calls detected:", response.tool_calls)
#     state["function_calls"] = []
#     tool_map = {t.name: t for t in CANDIDATE_TOOLS}
#     messages.append(response)

#     finalized_positions = []

#     for call in response.tool_calls:
#         tool_name = call["name"]
#         tool_args = call["args"]

#         if tool_name == "finalize_application":
#             pos_id = tool_args.get("position_id")
#             ref_map = state.get("position_ref_map", {})
#             applied_position_name = ref_map.get(pos_id, f"position #{pos_id}")

#             if not pos_id and state.get("scored_jobs"):
#                 # Resolve to the best-scoring allowed job
#                 allowed = [j for j in state["scored_jobs"] if j.get("overallStatus") in _APPLY_ALLOWED_STATUSES]
#                 if allowed:
#                     best = max(allowed, key=lambda j: j.get("technicalScore", 0) + j.get("experienceScore", 0))
#                     pos_id = best.get("positionId")
#                     tool_args = {**tool_args, "position_id": pos_id}
#                     applied_position_name = ref_map.get(pos_id, f"position #{pos_id}")
#                     print(f"[Tầng 1] Resolved position_id={pos_id} from scored_jobs cache.")

#             # Guardrail: block POOR_FIT applications
#             matched_job = next(
#                 (j for j in (state.get("scored_jobs") or []) if j.get("positionId") == pos_id),
#                 None,
#             )
#             if matched_job and matched_job.get("overallStatus") not in _APPLY_ALLOWED_STATUSES:
#                 skill_miss = matched_job.get("skillMiss", [])
#                 learning_path = matched_job.get("learningPath", "")
#                 block_msg = (
#                     f"Không thể nộp đơn vào vị trí **{applied_position_name}** "
#                     f"(Trạng thái: **POOR_FIT**).\n\n"
#                     f"**Kỹ năng còn thiếu:** {', '.join(skill_miss) if skill_miss else 'N/A'}\n\n"
#                     f"**Lộ trình cải thiện:** {learning_path or 'Chưa có gợi ý cụ thể.'}"
#                 )
#                 messages.append(ToolMessage(content=block_msg, tool_call_id=call["id"]))
#                 state["function_calls"].append({"name": tool_name, "arguments": tool_args, "result": block_msg})
#                 continue

#             try:
#                 # ISSUE-10: Normalize list→str for skill fields to prevent
#                 # LLM serializing List[str] as Python repr in API payload.
#                 normalized_args = {**tool_args}
#                 for skill_field in ("skill_match", "skill_miss"):
#                     val = normalized_args.get(skill_field)
#                     if isinstance(val, list):
#                         normalized_args[skill_field] = ", ".join(val)
#                     elif val is None:
#                         normalized_args[skill_field] = ""

#                 tool_res = await tool_map["finalize_application"].ainvoke({
#                     **normalized_args,
#                     "candidate_id": state["candidate_id"],
#                     "session_id":   state["session_id"],
#                 })
#                 state["function_calls"].append({"name": tool_name, "arguments": tool_args, "result": tool_res})
#                 messages.append(ToolMessage(content=str(tool_res), tool_call_id=call["id"]))

#                 if "thành công" in str(tool_res).lower() or "success" in str(tool_res).lower():
#                     finalized_positions.append(applied_position_name)
#             except Exception as e:
#                 messages.append(ToolMessage(content=f"Application error: {str(e)}", tool_call_id=call["id"]))

#         elif tool_name == "evaluate_cv_fit":
#             scored_summary = json.dumps(state.get("scored_jobs") or [], ensure_ascii=False)
#             state["function_calls"].append({"name": tool_name, "arguments": tool_args})
#             messages.append(ToolMessage(content=scored_summary, tool_call_id=call["id"]))

#         elif tool_name == "check_application_status":
#             enriched_args = {**tool_args, "candidate_id": state["candidate_id"]}
#             try:
#                 tool_res = await tool_map["check_application_status"].ainvoke(enriched_args)
#                 state["function_calls"].append({"name": tool_name, "arguments": enriched_args, "result": tool_res})
#                 messages.append(ToolMessage(content=str(tool_res), tool_call_id=call["id"]))
#             except Exception as e:
#                 messages.append(ToolMessage(content=f"Status check error: {str(e)}", tool_call_id=call["id"]))

#     if finalized_positions:
#         pos_list_str = ", ".join(f"**{name}**" for name in finalized_positions)
#         state["llm_response"] = (
#             f"I have successfully submitted your application for the following position(s): {pos_list_str}. "
#             "Please wait for our HR response!"
#         )
#         return state

#     if state["function_calls"]:
#         llm_no_tools = ChatGoogleGenerativeAI(
#             model=settings.GEMINI_MODEL,
#             temperature=temperature,
#             max_output_tokens=settings.GEMINI_MAX_TOKENS,
#             google_api_key=settings.GEMINI_API_KEY,
#         )
#         second_response = await llm_no_tools.ainvoke(messages)
#         state["llm_response"] = _extract_llm_text(second_response.content)

#     return state


# # ---------------------------------------------------------------------------
# # Node 4.5 — Persist turn to recruitment-service
# # ---------------------------------------------------------------------------

# async def save_turn_node(state: ChatState) -> ChatState:
#     """Save user + assistant messages to recruitment-service (MySQL) via internal API."""
#     try:
#         await recruitment_api.save_message(
#             session_id=state["session_id"],
#             role="USER",
#             content=state["query"],
#         )

#         function_call_payload: Optional[str] = None
#         raw_calls = state.get("function_calls")
#         if raw_calls:
#             function_call_payload = json.dumps(raw_calls, ensure_ascii=False)
#         elif state.get("scored_jobs"):
#             function_call_payload = json.dumps(
#                 {"scored_jobs": state["scored_jobs"]}, ensure_ascii=False
#             )

#         await recruitment_api.save_message(
#             session_id=state["session_id"],
#             role="ASSISTANT",
#             content=state["llm_response"],
#             function_call=function_call_payload,
#         )
#     except Exception as e:
#         print(f"[API Error] Could not save turn: {e}")

#     return state


# # ---------------------------------------------------------------------------
# # Node 5 — Format final response
# # ---------------------------------------------------------------------------

# def format_response_node(state: ChatState) -> ChatState:
#     """Package the final answer and metadata for the API layer."""
#     state["final_answer"] = state["llm_response"]
#     state["metadata"] = {
#         "intent":             state["intent"],
#         "intent_confidence":  state["intent_confidence"],
#         "domain":             state["domain"],
#         "is_apply_intent":    state.get("is_apply_intent", False),
#         "cv_chunks_used":     len(state.get("cv_context", [])),
#         "jd_docs_used":       len(state.get("jd_context", [])),
#         "temperature_used":   get_temperature_for_intent(state["intent"]),
#         "function_calls":     state.get("function_calls"),
#         "scored_jobs":        state.get("scored_jobs"),
#     }
#     return state


# # ---------------------------------------------------------------------------
# # Routing decision
# # ---------------------------------------------------------------------------

# def should_retrieve(state: ChatState) -> Literal["retrieve", "skip_retrieve"]:
#     intent = state["intent"]
#     if intent in ("jd_search", "jd_analysis", "cv_analysis"):
#         return "retrieve"
#     if intent == "general" and state.get("domain") == "general" and state["intent_confidence"] > 0.7:
#         return "skip_retrieve"
#     return "retrieve"


# # ---------------------------------------------------------------------------
# # Graph construction
# # ---------------------------------------------------------------------------

# def create_candidate_graph():
#     workflow = StateGraph(ChatState)

#     workflow.add_node("load_session_history", load_session_history_node)
#     workflow.add_node("classify_intent",      classify_intent_node)
#     workflow.add_node("retrieve_context",     retrieve_context_node)
#     workflow.add_node("scoring",              scoring_node)
#     workflow.add_node("build_prompts",        build_prompts_node)
#     workflow.add_node("llm_reasoning",        llm_reasoning_node)
#     workflow.add_node("save_turn",            save_turn_node)
#     workflow.add_node("format_response",      format_response_node)

#     workflow.set_entry_point("load_session_history")
#     workflow.add_edge("load_session_history", "classify_intent")

#     workflow.add_conditional_edges(
#         "classify_intent",
#         should_retrieve,
#         {"retrieve": "retrieve_context", "skip_retrieve": "build_prompts"},
#     )

#     workflow.add_edge("retrieve_context", "scoring")
#     workflow.add_edge("scoring",          "build_prompts")
#     workflow.add_edge("build_prompts",    "llm_reasoning")
#     workflow.add_edge("llm_reasoning",    "save_turn")
#     workflow.add_edge("save_turn",        "format_response")
#     workflow.add_edge("format_response",  END)

#     return workflow.compile()


# # ---------------------------------------------------------------------------
# # Public interface
# # ---------------------------------------------------------------------------

# class CandidateChatbot:
#     def __init__(self):
#         self.graph = create_candidate_graph()

#     async def chat(
#         self,
#         query: str,
#         session_id: str,
#         candidate_id: str,
#         cv_id: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         initial_state: ChatState = {
#             "query":                query,
#             "session_id":           session_id,
#             "candidate_id":         candidate_id,
#             "cv_id":                cv_id,
#             "jd_id":                None,
#             "conversation_history": [],
#             "active_position_ids":  [],
#             "position_ref_map":     {},
#             "cv_context":           [],
#             "jd_context":           [],
#             "retrieval_stats":      {},
#             "scored_jobs":          None,
#             "is_apply_intent":      False,
#             "system_prompt":        "",
#             "user_prompt":          "",
#             "llm_response":         "",
#             "function_calls":       None,
#             "final_answer":         "",
#             "metadata":             {},
#         }

#         final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 50})

#         return {
#             "answer":   final_state["final_answer"],
#             "metadata": final_state["metadata"],
#         }


# candidate_chatbot = CandidateChatbot()
