import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from app.config import settings
from app.models.schemas import AgentState
from app.agents.router import QueryRouter
from app.agents.pdf_agent import PDFAgent
from app.agents.web_agent import WebSearchAgent
from app.agents.clarification import ClarificationAgent
from app.services.vectorstore import VectorStoreService
from app.services.session import session_manager

logger = logging.getLogger(__name__)


class MultiAgentRAGSystem:
    """Multi-agent RAG system using LangGraph for orchestration"""
    
    def __init__(self):
        # Initialize services
        self.vectorstore_service = VectorStoreService()
        
        # Initialize agents
        self.router = QueryRouter()
        self.pdf_agent = PDFAgent(self.vectorstore_service)
        self.web_agent = WebSearchAgent()
        self.clarification_agent = ClarificationAgent()
        
        # Initialize the graph
        self.graph = self._create_graph()
        
        logger.info("Multi-agent RAG system initialized")
    
    def _create_graph(self) -> StateGraph:
        """Create and configure the LangGraph state graph"""
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("router", self._route_query)
        workflow.add_node("pdf_agent", self._call_pdf_agent)
        workflow.add_node("web_agent", self._call_web_agent)
        workflow.add_node("clarification_agent", self._call_clarification_agent)
        workflow.add_node("finalizer", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._determine_next_node,
            {
                "pdf_agent": "pdf_agent",
                "web_agent": "web_agent", 
                "clarification_agent": "clarification_agent"
            }
        )
        
        # All agents go to finalizer
        workflow.add_edge("pdf_agent", "finalizer")
        workflow.add_edge("web_agent", "finalizer")
        workflow.add_edge("clarification_agent", "finalizer")
        
        # Finalizer ends the workflow
        workflow.add_edge("finalizer", END)
        
        # Compile the graph
        memory = MemorySaver()
        compiled_graph = workflow.compile(checkpointer=memory)
        
        return compiled_graph
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Router node to determine which agent should handle the query"""
        try:
            logger.info("Routing query")
            
            # Check if this is a follow-up to a clarification request
            if (state.conversation_history and 
                self.clarification_agent.is_follow_up_clarification(
                    state.question, state.conversation_history
                )):
                # Route to PDF agent for follow-up after clarification
                state.agent_actions.append("Follow-up after clarification, routing to PDF agent")
                state.next_agent = "pdf_agent"
                return state
            
            # Use router to determine the next agent
            next_agent = self.router.route_query(state)
            state.next_agent = next_agent.lower()
            
            return state
            
        except Exception as e:
            logger.error(f"Error in router node: {e}")
            state.next_agent = "pdf_agent"  # Default fallback
            return state
    
    def _call_pdf_agent(self, state: AgentState) -> AgentState:
        """PDF agent node"""
        try:
            logger.info("Calling PDF agent")
            return self.pdf_agent.process_query(state)
        except Exception as e:
            logger.error(f"Error in PDF agent node: {e}")
            state.final_answer = f"An error occurred in the PDF agent: {str(e)}"
            state.confidence_score = 0.0
            state.agent_used = "pdf_agent"
            return state
    
    def _call_web_agent(self, state: AgentState) -> AgentState:
        """Web search agent node"""
        try:
            logger.info("Calling Web agent")
            return self.web_agent.process_query(state)
        except Exception as e:
            logger.error(f"Error in Web agent node: {e}")
            state.final_answer = f"An error occurred in the web search agent: {str(e)}"
            state.confidence_score = 0.0
            state.agent_used = "web_agent"
            return state
    
    def _call_clarification_agent(self, state: AgentState) -> AgentState:
        """Clarification agent node"""
        try:
            logger.info("Calling Clarification agent")
            return self.clarification_agent.process_query(state)
        except Exception as e:
            logger.error(f"Error in Clarification agent node: {e}")
            state.final_answer = "I need more information to answer your question effectively. Could you please provide more specific details?"
            state.confidence_score = 0.5
            state.agent_used = "clarification_agent"
            return state
    
    def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalizer node to prepare the final response"""
        try:
            logger.info("Finalizing response")
            
            # Log the agent actions for debugging
            state.agent_actions.append(f"Final answer generated by {state.agent_used}")
            
            # Ensure we have a final answer
            if not state.final_answer:
                state.final_answer = "I'm sorry, I couldn't generate an answer to your question."
                state.confidence_score = 0.0
            
            # Set default confidence if not set
            if state.confidence_score is None:
                state.confidence_score = 0.5
            
            # Ensure sources is a list
            if state.sources is None:
                state.sources = []
            
            return state
            
        except Exception as e:
            logger.error(f"Error in finalizer node: {e}")
            state.final_answer = "An error occurred while processing your request."
            state.confidence_score = 0.0
            state.sources = []
            return state
    
    def _determine_next_node(self, state: AgentState) -> str:
        """Determine which node to go to next based on router decision"""
        return state.next_agent
    
    async def process_question(
        self, 
        question: str, 
        session_id: str,
        thread_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a question through the multi-agent system"""
        try:
            logger.info(f"Processing question for session {session_id}: {question}")
            
            # Get conversation history
            conversation_history = session_manager.get_conversation_history(session_id)
            
            # Create initial state
            initial_state = AgentState(
                question=question,
                session_id=session_id,
                conversation_history=conversation_history,
                agent_actions=[],
                sources=[],
                agent_used="unknown"
            )
            
            # Configure thread for graph execution
            if not thread_config:
                thread_config = {"configurable": {"thread_id": session_id}}
            
            # Run the graph
            final_state = await self.graph.ainvoke(
                initial_state.dict(),
                config=thread_config
            )
            
            # Convert back to AgentState for type safety
            result_state = AgentState(**final_state)
            
            # Add messages to session
            session_manager.add_message(
                session_id, 
                "user", 
                question,
                {"timestamp": "now"}
            )
            session_manager.add_message(
                session_id,
                "assistant",
                result_state.final_answer,
                {
                    "agent_used": result_state.agent_used,
                    "confidence_score": result_state.confidence_score,
                    "sources_count": len(result_state.sources)
                }
            )
            
            # Prepare response
            response = {
                "answer": result_state.final_answer,
                "sources": result_state.sources,
                "session_id": session_id,
                "agent_used": result_state.agent_used,
                "confidence_score": result_state.confidence_score,
                "agent_actions": result_state.agent_actions,
                "needs_clarification": result_state.needs_clarification
            }
            
            logger.info(f"Question processed successfully by {result_state.agent_used}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            # Fallback response
            response = {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "agent_used": "error_handler",
                "confidence_score": 0.0,
                "agent_actions": [f"Error: {str(e)}"],
                "needs_clarification": False
            }
            
            return response
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system state"""
        try:
            vectorstore_info = self.vectorstore_service.get_collection_info()
            sessions_info = session_manager.get_all_sessions()
            
            return {
                "vectorstore": vectorstore_info,
                "sessions": sessions_info,
                "agents": {
                    "router": "QueryRouter",
                    "pdf_agent": "PDFAgent", 
                    "web_agent": "WebSearchAgent",
                    "clarification_agent": "ClarificationAgent"
                },
                "settings": {
                    "openai_model": settings.openai_model,
                    "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size,
                    "max_iterations": settings.max_iterations
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        return session_manager.clear_session(session_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        return session_manager.cleanup_expired_sessions()


# Global instance
multi_agent_system = MultiAgentRAGSystem() 