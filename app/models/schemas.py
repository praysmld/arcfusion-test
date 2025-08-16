from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="The AI assistant's answer")
    sources: List[Dict[str, Any]] = Field(default=[], description="Sources used to generate the answer")
    session_id: str = Field(..., description="Session ID for this conversation")
    agent_used: str = Field(..., description="Which agent was used to generate the answer")
    confidence_score: Optional[float] = Field(None, description="Confidence score for the answer")


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to clear")


class ClearMemoryResponse(BaseModel):
    success: bool = Field(..., description="Whether the memory was successfully cleared")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")


class AgentState(BaseModel):
    """State for the multi-agent graph"""
    question: str = Field(..., description="Original user question")
    session_id: str = Field(..., description="Session ID")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")
    agent_actions: List[str] = Field(default_factory=list, description="Actions taken by agents")
    pdf_results: Optional[List[Dict[str, Any]]] = Field(None, description="Results from PDF search")
    web_results: Optional[List[Dict[str, Any]]] = Field(None, description="Results from web search")
    needs_clarification: bool = Field(False, description="Whether the question needs clarification")
    clarification_request: Optional[str] = Field(None, description="Clarification request")
    final_answer: Optional[str] = Field(None, description="Final answer to return")
    confidence_score: Optional[float] = Field(None, description="Confidence in the answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used")
    agent_used: str = Field("unknown", description="Which agent was used")
    next_agent: Optional[str] = Field(None, description="Next agent to route to")


class DocumentChunk(BaseModel):
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the chunk")
    similarity_score: Optional[float] = Field(None, description="Similarity score to query") 


class ChunksResponse(BaseModel):
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks")
    total_count: int = Field(..., description="Total number of chunks retrieved")
    limit_applied: Optional[int] = Field(None, description="Limit applied to the query") 