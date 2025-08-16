import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings
from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


class QueryRouter:
    """Router agent to determine which agent should handle the query"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query routing agent. Your job is to analyze user questions and determine the best approach to answer them.

Available agents:
1. PDF_AGENT: For questions that can be answered from academic papers about text-to-SQL and LLM capabilities
2. WEB_AGENT: For current events, recent releases, or information not in the PDFs
3. CLARIFICATION_AGENT: For ambiguous or underspecified questions that need clarification

Available PDFs contain research on:
- Text-to-SQL generation with large language models
- Prompting strategies for SQL generation
- Benchmarking LLM capabilities on SQL tasks
- Zero-shot and few-shot learning approaches

Routing Rules:
- Use PDF_AGENT if the question is about research findings, methodologies, experimental results, or specific papers mentioned
- Use WEB_AGENT if the question asks about recent events, current releases, or explicitly requests web search
- Use CLARIFICATION_AGENT if the question is vague, ambiguous, or lacks necessary context

Consider the conversation history to understand context.

Respond with ONLY one of: PDF_AGENT, WEB_AGENT, or CLARIFICATION_AGENT"""),
            ("human", "Question: {question}\n\nConversation History: {history}\n\nRoute to:")
        ])
    
    def route_query(self, state: AgentState) -> str:
        """Route the query to the appropriate agent"""
        try:
            # Prepare conversation history
            history_text = ""
            if state.conversation_history:
                history_items = []
                for msg in state.conversation_history[-5:]:  # Last 5 messages for context
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            else:
                history_text = "No previous conversation"
            
            # Get routing decision
            response = self.llm.invoke(
                self.routing_prompt.format(
                    question=state.question,
                    history=history_text
                )
            )
            
            route = response.content.strip().upper()
            
            # Validate route
            valid_routes = ["PDF_AGENT", "WEB_AGENT", "CLARIFICATION_AGENT"]
            if route not in valid_routes:
                logger.warning(f"Invalid route '{route}', defaulting to PDF_AGENT")
                route = "PDF_AGENT"
            
            logger.info(f"Query routed to: {route}")
            state.agent_actions.append(f"Routed to {route}")
            
            return route
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Default to PDF agent on error
            return "PDF_AGENT"
    
    def needs_clarification(self, question: str, history: list = None) -> tuple[bool, str]:
        """Check if a question needs clarification"""
        try:
            clarification_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the user's question to determine if it's too vague or ambiguous to answer effectively.

A question needs clarification if:
- It uses vague terms like "good", "enough", "best" without context
- It lacks specific parameters or constraints
- It refers to "it" or "this" without clear reference
- Multiple interpretations are possible

If clarification is needed, provide a helpful clarification request that asks for specific details.

Respond in this format:
NEEDS_CLARIFICATION: Yes/No
CLARIFICATION_REQUEST: [your request if needed, or "None"]"""),
                ("human", "Question: {question}\n\nHistory: {history}")
            ])
            
            history_text = ""
            if history:
                history_items = []
                for msg in history[-3:]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            
            response = self.llm.invoke(
                clarification_prompt.format(
                    question=question,
                    history=history_text or "No previous conversation"
                )
            )
            
            content = response.content.strip()
            lines = content.split('\n')
            
            needs_clarification = False
            clarification_request = ""
            
            for line in lines:
                if line.startswith("NEEDS_CLARIFICATION:"):
                    needs_clarification = "yes" in line.lower()
                elif line.startswith("CLARIFICATION_REQUEST:"):
                    clarification_request = line.replace("CLARIFICATION_REQUEST:", "").strip()
            
            return needs_clarification, clarification_request
            
        except Exception as e:
            logger.error(f"Error checking clarification needs: {e}")
            return False, ""


# Initialize global router instance
query_router = QueryRouter() 