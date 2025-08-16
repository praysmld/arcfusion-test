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
        # Main router LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Use GPT-4o mini for clarification detection
        self.clarification_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
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
- Use WEB_AGENT if the question asks about recent events, current releases, recent LLM models, or explicitly requests web search
- Use WEB_AGENT for questions about specific companies (like Meta, Google, OpenAI, Anthropic) and their recent products
- Use WEB_AGENT for any question that requires information about events or technologies from 2023 or later
- Use CLARIFICATION_AGENT if the question is vague, ambiguous, or lacks necessary context

Consider the conversation history to understand context.

Respond with ONLY one of: PDF_AGENT, WEB_AGENT, or CLARIFICATION_AGENT"""),
            ("human", "Question: {question}\n\nConversation History: {history}\n\nRoute to:")
        ])
        
        # Enhanced clarification detection prompt
        self.clarification_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at detecting ambiguous or vague questions that require clarification. Your job is to analyze questions and determine if they need clarification before they can be answered effectively.

A question needs clarification if:
1. It contains vague qualifiers ("good", "enough", "best", "high") without context
2. It lacks specific parameters or constraints needed to provide a precise answer
3. It contains unclear references ("it", "this", "that") without clear antecedents
4. It has multiple possible interpretations that could lead to different answers
5. It's overly broad or general without specific focus
6. It uses domain-specific terms without clarifying which aspect is being asked about

For technical questions about text-to-SQL or LLMs, look for:
- Missing dataset specifications
- Unclear evaluation metrics
- Ambiguous model references
- Undefined comparison criteria

Analyze the question carefully and determine if clarification is needed.

Respond in this format:
NEEDS_CLARIFICATION: Yes/No
CONFIDENCE: [1-10 scale, where 10 is very confident]
PRIMARY_ISSUE: [Main reason clarification is needed, or "None" if no clarification needed]
CLARIFICATION_REQUEST: [Specific clarification request, or "None" if no clarification needed]"""),
            ("human", "Question: {question}\n\nHistory: {history}")
        ])
    
    def route_query(self, state: AgentState) -> str:
        """Route the query to the appropriate agent"""
        try:
            # First check if clarification is needed using GPT-4o mini
            needs_clarification, confidence, _ = self._check_clarification_needed(state.question, state.conversation_history)
            
            # If high confidence that clarification is needed, route directly
            if needs_clarification and confidence >= 7:
                logger.info("Query determined to need clarification with high confidence")
                state.agent_actions.append("Routed to CLARIFICATION_AGENT based on ambiguity detection")
                return "CLARIFICATION_AGENT"
            
            # Check for explicit web search keywords next
            if self._should_use_web_agent(state.question):
                logger.info("Query contains web search keywords, routing to WEB_AGENT")
                state.agent_actions.append("Routed to WEB_AGENT based on keywords")
                return "WEB_AGENT"
            
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
            
            # Get routing decision from main LLM
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
            
            # If medium confidence that clarification is needed, prioritize it over the main routing decision
            if needs_clarification and confidence >= 5 and route != "CLARIFICATION_AGENT":
                logger.info(f"Overriding route {route} with CLARIFICATION_AGENT based on medium confidence")
                state.agent_actions.append(f"Route changed from {route} to CLARIFICATION_AGENT based on ambiguity detection")
                return "CLARIFICATION_AGENT"
            
            logger.info(f"Query routed to: {route}")
            state.agent_actions.append(f"Routed to {route}")
            
            return route
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Default to PDF agent on error
            return "PDF_AGENT"
    
    def _should_use_web_agent(self, question: str) -> bool:
        """Check if the question should be directly routed to web agent based on keywords"""
        question_lower = question.lower()
        
        # Direct check for Meta LLM questions - highest priority
        if ("meta" in question_lower and any(term in question_lower for term in ["llm", "language model", "ai model", "model"])) or \
           ("llama" in question_lower) or \
           ("meta" in question_lower and "launch" in question_lower):
            logger.info("Detected Meta LLM question, routing to web agent")
            return True
        
        # Keywords that strongly suggest web search
        web_keywords = [
            "recent", "latest", "current", "new", "2023", "2024", 
            "last month", "this year", "last week", "news", "announced",
            "released", "launched", "update", "upcoming"
        ]
        
        # Company names that suggest web search for recent information
        company_keywords = [
            "meta", "facebook", "openai", "anthropic", "google", "microsoft", 
            "claude", "gpt-4", "gpt-5", "llama", "gemini", "mistral"
        ]
        
        # Check for web keywords
        if any(keyword in question_lower for keyword in web_keywords):
            return True
            
        # Check for company names combined with certain verbs
        if any(company in question_lower for company in company_keywords):
            action_verbs = ["launch", "release", "announce", "introduce", "unveil", "develop", "create", "build"]
            if any(verb in question_lower for verb in action_verbs):
                return True
        
        return False
    
    def _check_clarification_needed(self, question: str, history: list = None) -> tuple[bool, int, str]:
        """Check if a question needs clarification using GPT-4o mini"""
        try:
            history_text = ""
            if history:
                history_items = []
                for msg in history[-3:]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            
            # Use GPT-4o mini for clarification detection
            response = self.clarification_llm.invoke(
                self.clarification_detection_prompt.format(
                    question=question,
                    history=history_text or "No previous conversation"
                )
            )
            
            content = response.content.strip()
            lines = content.split('\n')
            
            needs_clarification = False
            confidence = 5  # Default medium confidence
            primary_issue = ""
            clarification_request = ""
            
            for line in lines:
                if line.startswith("NEEDS_CLARIFICATION:"):
                    needs_clarification = "yes" in line.lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence_str = line.replace("CONFIDENCE:", "").strip()
                        confidence = int(confidence_str)
                    except ValueError:
                        pass
                elif line.startswith("PRIMARY_ISSUE:"):
                    primary_issue = line.replace("PRIMARY_ISSUE:", "").strip()
                elif line.startswith("CLARIFICATION_REQUEST:"):
                    clarification_request = line.replace("CLARIFICATION_REQUEST:", "").strip()
            
            logger.info(f"Clarification check: needed={needs_clarification}, confidence={confidence}, issue={primary_issue}")
            return needs_clarification, confidence, clarification_request
            
        except Exception as e:
            logger.error(f"Error checking clarification needs: {e}")
            return False, 0, ""
    
    def needs_clarification(self, question: str, history: list = None) -> tuple[bool, str]:
        """Check if a question needs clarification (legacy method)"""
        needs_clarification, _, clarification_request = self._check_clarification_needed(question, history)
        return needs_clarification, clarification_request


# Initialize global router instance
query_router = QueryRouter() 