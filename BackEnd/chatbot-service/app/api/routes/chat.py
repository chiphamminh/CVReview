# from fastapi import APIRouter, HTTPException, status, Header
# from fastapi.responses import StreamingResponse
# import time
# import json
# from typing import AsyncIterator, Optional

# from app.models.chat import ChatRequest, ChatResponse, ErrorResponse, ChatMetadata
# from app.rag.graph import chatbot

# router = APIRouter()


# # ============================================================
# # REGULAR CHAT ENDPOINT
# # ============================================================

# @router.post(
#     "/chat",
#     response_model=ChatResponse,
#     responses={
#         400: {"model": ErrorResponse, "description": "Bad Request"},
#         500: {"model": ErrorResponse, "description": "Internal Server Error"}
#     },
#     summary="Chat with Career Counselor",
#     description="""
#     Send a query to the AI career counselor chatbot.
    
#     The chatbot can:
#     - Analyze CV/resume content
#     - Search for suitable job positions
#     - Match candidates to job descriptions
#     - Provide career guidance
    
#     **Intent Detection:**
#     - `cv_analysis`: Questions about candidate's skills, experience, profile
#     - `jd_search`: Finding suitable job positions
#     - `cv_jd_match`: Matching a CV to specific job description
#     - `general`: General career advice
#     """
# )
# async def chat_endpoint(
#     request: ChatRequest,
#     x_user_id: str = Header(..., alias="X-User-Id"),
#     x_user_role: str = Header(..., alias="X-User-Role"),
#     x_user_phone: Optional[str] = Header(None, alias="X-User-Phone")
# ):
#     """
#     Main chat endpoint (non-streaming)
#     User info automatically injected from API Gateway via JWT
#     """
#     start_time = time.time()
    
#     if x_user_role and x_user_role.upper() == "HR":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="HR role cannot access chatbot service"
#         )
    
#     # Override candidate_id from JWT if not provided
#     if not request.candidate_id:
#         request.candidate_id = x_user_id
#         print(f"Using candidate_id from JWT: {x_user_id}")
    
#     try:
#         # Validate request
#         if not request.query or not request.query.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Query cannot be empty"
#             )
        
#         # Call chatbot
#         result = await chatbot.chat(
#             query=request.query,
#             candidate_id=request.candidate_id,
#             cv_id=request.cv_id,
#             jd_id=request.jd_id,
#             conversation_history=request.conversation_history 
#         )
        
#         # Calculate processing time
#         processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
#         # Build response
#         metadata = ChatMetadata(
#             intent=result["metadata"]["intent"],
#             intent_confidence=result["metadata"]["intent_confidence"],
#             cv_chunks_used=result["metadata"]["cv_chunks_used"],
#             jd_docs_used=result["metadata"]["jd_docs_used"],
#             retrieval_stats=result["metadata"]["retrieval_stats"],
#             processing_time_ms=processing_time
#         )
        
#         response = ChatResponse(
#             answer=result["answer"],
#             metadata=metadata
#         )
        
#         return response
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error in chat endpoint: {e}")
#         import traceback
#         traceback.print_exc()
        
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Internal server error: {str(e)}"
#         )


# # ============================================================
# # STREAMING CHAT ENDPOINT
# # ============================================================

# async def chat_stream_generator(
#     query: str,
#     candidate_id: str = None,
#     cv_id: int = None,
#     jd_id: int = None
# ) -> AsyncIterator[str]:
#     """
#     Generator for streaming chat responses
    
#     Yields Server-Sent Events (SSE) format:
#     data: {"type": "token", "content": "Hello"}\n\n
#     data: {"type": "done", "metadata": {...}}\n\n
#     """
#     start_time = time.time()
    
#     try:
#         # Import streaming-capable LLM
#         from langchain_openai import ChatOpenAI
#         from app.config import get_settings
#         from app.rag.intent import intent_classifier
#         from app.rag.prompts import get_prompt_for_intent
#         from app.services.retriever import retriever
        
#         settings = get_settings()
        
#         # Step 1: Intent Classification
#         yield f"data: {json.dumps({'type': 'status', 'content': 'Classifying intent...'})}\n\n"
        
#         intent_result = intent_classifier.classify(query)
#         intent = intent_result["intent"]
        
#         yield f"data: {json.dumps({'type': 'intent', 'intent': intent, 'confidence': intent_result['confidence']})}\n\n"
        
#         # Step 2: Retrieval
#         yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving context...'})}\n\n"
        
#         retrieval_result = await retriever.retrieve_for_intent(
#             query=query,
#             intent=intent,
#             candidate_id=candidate_id,
#             cv_id=cv_id,
#             jd_id=jd_id,
#             top_k=5 if intent == "cv_analysis" else 3,
#             score_threshold=0.5
#         )
        
#         cv_context = retrieval_result.get("cv_context", [])
#         jd_context = retrieval_result.get("jd_context", [])
        
#         yield f"data: {json.dumps({'type': 'retrieval', 'cv_chunks': len(cv_context), 'jd_docs': len(jd_context)})}\n\n"
        
#         # Step 3: Build prompts
#         system_prompt, user_prompt = get_prompt_for_intent(
#             intent=intent,
#             query=query,
#             cv_context=cv_context,
#             jd_context=jd_context
#         )
        
#         # Step 4: Stream LLM response
#         yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
        
#         llm = ChatOpenAI(
#             model=settings.OPENAI_MODEL,
#             temperature=settings.OPENAI_TEMPERATURE,
#             openai_api_key=settings.OPENAI_API_KEY,
#             streaming=True
#         )
        
#         # Stream tokens
#         full_response = ""
#         async for chunk in llm.astream([
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]):
#             if chunk.content:
#                 full_response += chunk.content
#                 yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
        
#         # Step 5: Send metadata
#         processing_time = (time.time() - start_time) * 1000
        
#         metadata = {
#             "intent": intent,
#             "intent_confidence": intent_result["confidence"],
#             "cv_chunks_used": len(cv_context),
#             "jd_docs_used": len(jd_context),
#             "retrieval_stats": retrieval_result.get("retrieval_stats", {}),
#             "processing_time_ms": processing_time
#         }
        
#         yield f"data: {json.dumps({'type': 'done', 'metadata': metadata})}\n\n"
        
#     except Exception as e:
#         error_msg = f"Error: {str(e)}"
#         yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"


# @router.post(
#     "/chat/stream",
#     summary="Chat with Streaming",
#     description="Stream the AI response token by token for better UX",
#     responses={
#         200: {
#             "description": "Server-Sent Events stream",
#             "content": {
#                 "text/event-stream": {
#                     "example": """data: {"type": "status", "content": "Classifying intent..."}

# data: {"type": "intent", "intent": "cv_analysis", "confidence": 0.85}

# data: {"type": "token", "content": "Based on"}

# data: {"type": "token", "content": " your CV,"}

# data: {"type": "done", "metadata": {...}}
# """
#                 }
#             }
#         }
#     }
# )

# async def chat_stream_endpoint(
#     request: ChatRequest,
#     x_user_id: str = Header(..., alias="X-User-Id"),
#     x_user_role: str = Header(..., alias="X-User-Role"),
#     x_user_phone: Optional[str] = Header(None, alias="X-User-Phone")
# ):
#     """
#     Streaming chat endpoint using Server-Sent Events (SSE)
#     User info automatically injected from API Gateway via JWT
#     """
#     if x_user_role and x_user_role.upper() == "HR":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="HR role cannot access chatbot service"
#         )
    
#     # Override candidate_id from JWT
#     if not request.candidate_id:
#         request.candidate_id = x_user_id
    
#     try:
#         if not request.query or not request.query.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Query cannot be empty"
#             )
        
#         return StreamingResponse(
#             chat_stream_generator(
#                 query=request.query,
#                 candidate_id=request.candidate_id,
#                 cv_id=request.cv_id,
#                 jd_id=request.jd_id
#             ),
#             media_type="text/event-stream",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "X-Accel-Buffering": "no"  # Disable nginx buffering
#             }
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error in streaming endpoint: {e}")
#         import traceback
#         traceback.print_exc()
        
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Internal server error: {str(e)}"
#         )