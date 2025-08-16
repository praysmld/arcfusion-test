import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.api.endpoints import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Chat With PDF Backend")
    
    # Check for required environment variables
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not found - web search will use mock results")
    
    # Ensure data directories exist
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chat With PDF Backend")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="""
    A multi-agent RAG system for intelligent question-answering over academic papers.
    
    This system can:
    - Answer questions from PDF documents using retrieval-augmented generation
    - Perform web searches for current information not in the PDFs
    - Handle ambiguous questions with clarification requests
    - Maintain conversation memory within sessions
    
    The system uses LangGraph to orchestrate multiple specialized agents for optimal responses.
    """,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["Chat with PDF"])


@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Chat With PDF Backend",
        "version": settings.version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "error": str(exc) if settings.debug else "Internal server error"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    ) 