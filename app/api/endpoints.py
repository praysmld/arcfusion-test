import logging
from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    QuestionRequest, QuestionResponse, 
    ClearMemoryRequest, ClearMemoryResponse,
    HealthResponse, DocumentChunk, ChunksResponse
)
from app.agents.graph import multi_agent_system
from app.services.session import session_manager

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.version
    )


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the multi-agent RAG system"""
    try:
        logger.info(f"Received question: {request.question}")
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process the question through the multi-agent system
        response = await multi_agent_system.process_question(
            question=request.question,
            session_id=session_id
        )
        
        # Return structured response
        return QuestionResponse(
            answer=response["answer"],
            sources=response["sources"],
            session_id=response["session_id"],
            agent_used=response["agent_used"],
            confidence_score=response["confidence_score"]
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}"
        )


@router.post("/clear-memory", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest):
    """Clear conversation memory for a specific session"""
    try:
        logger.info(f"Clearing memory for session: {request.session_id}")
        
        success = multi_agent_system.clear_session(request.session_id)
        
        if success:
            return ClearMemoryResponse(
                success=True,
                message=f"Memory cleared successfully for session {request.session_id}"
            )
        else:
            return ClearMemoryResponse(
                success=False,
                message=f"Session {request.session_id} not found"
            )
            
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while clearing memory: {str(e)}"
        )


@router.get("/sessions")
async def get_sessions():
    """Get information about all active sessions"""
    try:
        sessions_info = session_manager.get_all_sessions()
        return JSONResponse(content=sessions_info)
        
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get information about a specific session"""
    try:
        session_metadata = session_manager.get_session_metadata(session_id)
        
        if not session_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return JSONResponse(content=session_metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving session: {str(e)}"
        )


@router.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str, limit: Optional[int] = None):
    """Get conversation history for a specific session"""
    try:
        history = session_manager.get_conversation_history(session_id, limit)
        
        return JSONResponse(content={
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation history for {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving conversation history: {str(e)}"
        )


@router.get("/system/info")
async def get_system_info():
    """Get system information and status"""
    try:
        system_info = multi_agent_system.get_system_info()
        return JSONResponse(content=system_info)
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving system information: {str(e)}"
        )


@router.post("/system/cleanup")
async def cleanup_expired_sessions(background_tasks: BackgroundTasks):
    """Clean up expired sessions"""
    try:
        def cleanup_task():
            try:
                removed_count = multi_agent_system.cleanup_expired_sessions()
                logger.info(f"Cleaned up {removed_count} expired sessions")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
        
        background_tasks.add_task(cleanup_task)
        
        return JSONResponse(content={
            "message": "Cleanup task started",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error starting cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while starting cleanup: {str(e)}"
        )


@router.get("/system/vectorstore")
async def get_vectorstore_info():
    """Get vector store information"""
    try:
        vectorstore_info = multi_agent_system.vectorstore_service.get_collection_info()
        return JSONResponse(content=vectorstore_info)
        
    except Exception as e:
        logger.error(f"Error getting vectorstore info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving vectorstore information: {str(e)}"
        )


@router.get("/chunks", response_model=ChunksResponse)
async def get_all_chunks(limit: Optional[int] = None):
    """Get all chunks stored in the vector database"""
    try:
        logger.info(f"Retrieving all chunks with limit: {limit}")
        
        chunks = multi_agent_system.vectorstore_service.get_all_chunks(limit=limit)
        
        return ChunksResponse(
            chunks=chunks,
            total_count=len(chunks),
            limit_applied=limit
        )
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving chunks: {str(e)}"
        )


# Note: Exception handlers should be defined in the main FastAPI app, not in routers 