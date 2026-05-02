from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import health, candidate_chat, hr_chat
from app.services.embedding import embedding_service
from app.services.qdrant import qdrant_service
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown
    """
    # Startup
    print("Starting Chatbot Service...")
    
    # Warm up embedding model
    print("Warming up embedding model...")
    embedding_service._load_model()
    print("Embedding model ready")
    
    # Test Qdrant connection
    print("Testing Qdrant connection...")
    if qdrant_service.test_connection():
        print("Qdrant connected")
    else:
        print(" Warning: Qdrant connection failed")
    
    # Check collections
    try:
        cv_info = qdrant_service.get_collection_info(settings.CV_COLLECTION_NAME)
        jd_info = qdrant_service.get_collection_info(settings.JD_COLLECTION_NAME)
        
        if cv_info:
            print(f"CV collection: {cv_info['points_count']} points")
        if jd_info:
            print(f"JD collection: {jd_info['points_count']} points")
    except Exception as e:
        print(f" Warning: Could not check collections: {e}")
    
    print("Chatbot Service started successfully!")
    print(f"API docs: http://localhost:8085/docs")
    
    yield
    
    # Shutdown
    print("Shutting down Chatbot Service...")


# Create FastAPI app
app = FastAPI(
    title="Career Counselor Chatbot API",
    description="""
    AI-powered career counseling chatbot for job matching and career guidance.
    
    ## Features
    
    * **Job Search**: Find suitable positions based on candidate CV
    * **Job Analysis**: Detailed analysis of specific job descriptions
    * **Skill Gap Analysis**: Identify learning needs and improvement areas
    * **Career Guidance**: Context-aware career advice with conversation memory
    
    ## Technologies
    
    * LangGraph for workflow orchestration
    * LangChain for RAG pipeline
    * Qdrant for vector search
    * Gemini 2.5 Flash for LLM reasoning
    * BGE-small-en-v1.5 for embeddings
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candidate_chat.router, prefix="/chatbot")
app.include_router(hr_chat.router, prefix="/chatbot")
app.include_router(health.router, prefix="/chatbot")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "service": "CV Review Chatbot Service",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "candidate_session": "POST /chatbot/candidate/session",
            "candidate_chat": "POST /chatbot/candidate/chat",
            "hr_session": "POST /chatbot/hr/session",
            "hr_chat": "POST /chatbot/hr/chat",
            "health": "GET /chatbot/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8085,
        reload=True,
        log_level="info"
    )