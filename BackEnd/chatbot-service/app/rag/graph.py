# """
# LangGraph workflow for career counseling chatbot
# Orchestrates: Intent Classification → Retrieval → LLM Reasoning → Response
# """

# from typing import TypedDict, Literal, Optional, List, Dict, Any
# from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import SystemMessage, HumanMessage

# from app.rag.intent import intent_classifier
# from app.rag.prompts import get_prompt_for_intent
# from app.services.retriever import retriever
# from app.config import get_settings

# settings = get_settings()


# # ============================================================
# # STATE DEFINITION 
# # ============================================================

# class ChatState(TypedDict):
#     """State passed between graph nodes"""
    
#     # Input
#     query: str
#     candidate_id: Optional[str]
#     cv_id: Optional[int]
#     jd_id: Optional[int]
#     conversation_history: Optional[List[Dict[str, str]]]
    
#     # Processing
#     intent: Literal["jd_search", "jd_analysis", "cv_analysis", "general"]
#     intent_confidence: float
#     domain: str
    
#     # Retrieved context
#     cv_context: List[Dict[str, Any]]
#     jd_context: List[Dict[str, Any]]
#     retrieval_stats: Dict[str, Any]
    
#     # LLM
#     system_prompt: str
#     user_prompt: str
#     llm_response: str
    
#     # Output
#     final_answer: str
#     metadata: Dict[str, Any]


# # ============================================================
# # HELPER: Intent-specific temperature
# # ============================================================

# def get_temperature_for_intent(intent: str) -> float:
#     """
#     Get optimal temperature based on intent
    
#     Lower temperature = more deterministic, factual
#     Higher temperature = more creative, varied
    
#     For career counseling, we want consistency and accuracy
#     """
#     temperatures = {
#         "cv_analysis": 0.2,    # Most deterministic - pure facts from CV
#         "jd_analysis": 0.25,   # Very deterministic - facts from JD
#         "jd_search": 0.3,      # Slightly more flexible for recommendations
#         "general": 0.4         # More conversational
#     }
    
#     return temperatures.get(intent, 0.3)


# # ============================================================
# # GRAPH NODES
# # ============================================================

# def classify_intent_node(state: ChatState) -> ChatState:
#     """Node 1: Classify user intent"""
#     query = state["query"]
    
#     result = intent_classifier.classify(query)
    
#     print(f"[Intent] {result['intent']} (confidence: {result['confidence']:.2f}, domain: {result['domain']})")
    
#     state["intent"] = result["intent"]
#     state["intent_confidence"] = result["confidence"]
#     state["domain"] = result["domain"]
    
#     return state


# async def retrieve_context_node(state: ChatState) -> ChatState:
#     """Node 2: Retrieve relevant context"""
#     query = state["query"]
#     intent = state["intent"]
    
#     print(f"[Retrieval] Intent: {intent}, Query: '{query[:50]}...'")
    
#     # retriever now uses adaptive top_k and threshold internally
#     result = await retriever.retrieve_for_intent(
#         query=query,
#         intent=intent,
#         candidate_id=state.get("candidate_id"),
#         cv_id=state.get("cv_id"),
#         jd_id=state.get("jd_id")
#         # No need to pass top_k, score_threshold - using adaptive defaults
#     )
    
#     state["cv_context"] = result.get("cv_context", [])
#     state["jd_context"] = result.get("jd_context", [])
#     state["retrieval_stats"] = result.get("retrieval_stats", {})
    
#     print(f"[Retrieval] Retrieved: {len(state['cv_context'])} CV chunks, {len(state['jd_context'])} JDs")
    
#     # Log retrieval quality
#     if "cv_quality" in state["retrieval_stats"]:
#         cv_quality = state["retrieval_stats"]["cv_quality"]
#         print(f"[Retrieval Quality] CV: {cv_quality['quality']} (score: {cv_quality['max_score']:.2f})")
    
#     if "jd_quality" in state["retrieval_stats"]:
#         jd_quality = state["retrieval_stats"]["jd_quality"]
#         print(f"[Retrieval Quality] JD: {jd_quality['quality']} (score: {jd_quality['max_score']:.2f})")
    
#     return state


# def build_prompts_node(state: ChatState) -> ChatState:
#     """Node 3: Build prompts from context"""
#     query = state["query"]
#     intent = state["intent"]
#     cv_context = state["cv_context"]
#     jd_context = state["jd_context"]
#     conversation_history = state.get("conversation_history")
    
#     print(f"[Prompts] Building for intent: {intent}")
    
#     system_prompt, user_prompt = get_prompt_for_intent(
#         intent=intent,
#         query=query,
#         cv_context=cv_context,
#         jd_context=jd_context,
#         conversation_history=conversation_history
#     )
    
#     state["system_prompt"] = system_prompt
#     state["user_prompt"] = user_prompt
    
#     # Log prompt lengths for monitoring
#     print(f"[Prompts] System: {len(system_prompt)} chars, User: {len(user_prompt)} chars")
    
#     return state


# async def llm_reasoning_node(state: ChatState) -> ChatState:
#     """
#     Node 4: LLM reasoning with context
#     TIER 1: Lower temperature for consistency
#     """
#     system_prompt = state["system_prompt"]
#     user_prompt = state["user_prompt"]
#     intent = state["intent"]

#     # Get intent-specific temperature
#     temperature = get_temperature_for_intent(intent)
    
#     print(f"[LLM] Calling Gemini ({settings.GEMINI_MODEL}) with temperature={temperature}")

#     llm = ChatGoogleGenerativeAI(
#         model=settings.GEMINI_MODEL,
#         temperature=temperature,  # Intent-specific, lower than before
#         max_output_tokens=settings.GEMINI_MAX_TOKENS,
#         google_api_key=settings.GEMINI_API_KEY
#     )

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_prompt)
#     ]

#     response = await llm.ainvoke(messages)

#     state["llm_response"] = response.content

#     print(f"[LLM] Response received: {len(response.content)} chars")

#     return state


# def format_response_node(state: ChatState) -> ChatState:
#     """Node 5: Format final response"""
#     llm_response = state["llm_response"]
    
#     state["final_answer"] = llm_response
    
#     # Enhanced metadata
#     state["metadata"] = {
#         "intent": state["intent"],
#         "intent_confidence": state["intent_confidence"],
#         "domain": state["domain"],
#         "cv_chunks_used": len(state["cv_context"]),
#         "jd_docs_used": len(state["jd_context"]),
#         "retrieval_stats": state["retrieval_stats"],
#         "temperature_used": get_temperature_for_intent(state["intent"]), 
#         "response_length": len(llm_response) 
#     }
    
#     print(f"[Response] Formatted: {len(llm_response)} chars")
    
#     return state


# # ============================================================
# # CONDITIONAL ROUTING 
# # ============================================================

# def should_retrieve(state: ChatState) -> Literal["retrieve", "skip_retrieve"]:
#     """Decide if retrieval is needed"""
#     intent = state["intent"]
#     domain = state.get("domain", "career")
#     confidence = state["intent_confidence"]
    
#     if intent in ["jd_search", "jd_analysis", "cv_analysis"]:
#         return "retrieve"
    
#     if intent == "general" and domain == "general" and confidence > 0.7:
#         return "skip_retrieve"
    
#     return "retrieve"


# # ============================================================
# # GRAPH CONSTRUCTION 
# # ============================================================

# def create_career_counselor_graph():
#     """Create the LangGraph workflow"""
    
#     workflow = StateGraph(ChatState)
    
#     workflow.add_node("classify_intent", classify_intent_node)
#     workflow.add_node("retrieve_context", retrieve_context_node)
#     workflow.add_node("build_prompts", build_prompts_node)
#     workflow.add_node("llm_reasoning", llm_reasoning_node)
#     workflow.add_node("format_response", format_response_node)
    
#     workflow.set_entry_point("classify_intent")
    
#     workflow.add_conditional_edges(
#         "classify_intent",
#         should_retrieve,
#         {
#             "retrieve": "retrieve_context",
#             "skip_retrieve": "build_prompts"
#         }
#     )
    
#     workflow.add_edge("retrieve_context", "build_prompts")
#     workflow.add_edge("build_prompts", "llm_reasoning")
#     workflow.add_edge("llm_reasoning", "format_response")
#     workflow.add_edge("format_response", END)
    
#     app = workflow.compile()
    
#     return app


# # ============================================================
# # CONVENIENCE WRAPPER (enhanced logging)
# # ============================================================

# class CareerChatbot:
#     """High-level wrapper for the career counseling chatbot"""
    
#     def __init__(self):
#         self.graph = create_career_counselor_graph()
    
#     async def chat(
#         self,
#         query: str,
#         candidate_id: Optional[str] = None,
#         cv_id: Optional[int] = None,
#         jd_id: Optional[int] = None,
#         conversation_history: Optional[List[Dict[str, str]]] = None
#     ) -> Dict[str, Any]:
#         """Main chat interface"""
        
#         initial_state = {
#             "query": query,
#             "candidate_id": candidate_id,
#             "cv_id": cv_id,
#             "jd_id": jd_id,
#             "conversation_history": conversation_history,
#             "cv_context": [],
#             "jd_context": [],
#             "retrieval_stats": {},
#         }
        
#         print(f"\n{'='*70}")
#         print(f"TIER 1 IMPROVED CHATBOT")
#         print(f"Query: {query[:60]}...")
#         print(f"Candidate: {candidate_id}, CV: {cv_id}, JD: {jd_id}")
#         print(f"{'='*70}\n")
        
#         final_state = await self.graph.ainvoke(initial_state)
        
#         print(f"\n{'='*70}")
#         print(f"Query processed successfully")
#         print(f"Intent: {final_state['metadata']['intent']} ({final_state['metadata']['intent_confidence']:.2f})")
#         print(f"Temperature: {final_state['metadata']['temperature_used']}")
#         print(f"Context: {final_state['metadata']['cv_chunks_used']} CV + {final_state['metadata']['jd_docs_used']} JD")
#         print(f"Response: {final_state['metadata']['response_length']} chars")
#         print(f"{'='*70}\n")
        
#         return {
#             "answer": final_state["final_answer"],
#             "metadata": final_state["metadata"]
#         }


# # Global instance
# chatbot = CareerChatbot()