import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.config import settings
from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


class ClarificationAgent:
    """Agent for handling ambiguous or underspecified queries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful clarification agent. Your job is to identify ambiguous or underspecified questions and request clarification to provide better answers.

When a question is vague or ambiguous, provide a helpful clarification request that:
1. Explains what makes the question unclear
2. Asks specific questions to gather needed details
3. Provides examples of how the question could be interpreted
4. Remains friendly and helpful

Focus on these types of ambiguity:
- Vague qualifiers ("good", "enough", "best", "high") without context
- Missing scope or constraints
- Unclear references ("it", "this", "that")
- Multiple possible interpretations
- Missing essential parameters

Context: The system can search academic papers on text-to-SQL and LLM research, or search the web for current information.

Previous Conversation:
{history}

Question: {question}

Clarification Request:"""),
            ("human", "{question}")
        ])
    
    def generate_clarification(self, question: str, history: str = "") -> str:
        """Generate a clarification request for an ambiguous question"""
        try:
            response = self.llm.invoke(
                self.clarification_prompt.format(
                    question=question,
                    history=history
                )
            )
            
            clarification = response.content.strip()
            logger.info("Generated clarification request")
            return clarification
            
        except Exception as e:
            logger.error(f"Error generating clarification: {e}")
            return "Could you please provide more specific details about your question? This will help me give you a more accurate and helpful answer."
    
    def analyze_ambiguity(self, question: str) -> Dict[str, Any]:
        """Analyze what makes a question ambiguous"""
        ambiguity_indicators = {
            'vague_qualifiers': [],
            'missing_context': False,
            'unclear_references': False,
            'multiple_interpretations': False,
            'scope_issues': False
        }
        
        # Check for vague qualifiers
        vague_terms = ['good', 'bad', 'best', 'worst', 'enough', 'sufficient', 'high', 'low', 'many', 'few', 'better', 'worse']
        for term in vague_terms:
            if term in question.lower():
                ambiguity_indicators['vague_qualifiers'].append(term)
        
        # Check for unclear references
        unclear_refs = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        for ref in unclear_refs:
            if ref in question.lower().split():
                ambiguity_indicators['unclear_references'] = True
                break
        
        # Check for scope issues
        scope_indicators = ['what', 'how', 'when', 'where', 'which', 'why']
        if not any(indicator in question.lower() for indicator in scope_indicators):
            ambiguity_indicators['scope_issues'] = True
        
        return ambiguity_indicators
    
    def suggest_specific_questions(self, question: str, domain: str = "text-to-sql") -> List[str]:
        """Suggest specific questions based on the domain"""
        suggestions = []
        
        if domain == "text-to-sql":
            if "accuracy" in question.lower():
                suggestions.extend([
                    "What specific dataset are you asking about (e.g., Spider, WikiSQL)?",
                    "Are you interested in execution accuracy or exact match accuracy?",
                    "Which model or approach are you comparing?"
                ])
            
            if "performance" in question.lower():
                suggestions.extend([
                    "Are you asking about speed/latency or accuracy metrics?",
                    "Which specific benchmark or evaluation criteria?",
                    "What model size or computational constraints?"
                ])
            
            if "prompt" in question.lower() or "prompting" in question.lower():
                suggestions.extend([
                    "Are you asking about zero-shot, few-shot, or in-context learning?",
                    "Which specific prompting technique (e.g., chain-of-thought, SQL generation)?",
                    "What target LLM are you interested in?"
                ])
        
        return suggestions
    
    def process_query(self, state: AgentState) -> AgentState:
        """Process a query that needs clarification"""
        try:
            logger.info(f"Clarification Agent processing query: {state.question}")
            
            # Prepare conversation history
            history_text = ""
            if state.conversation_history:
                history_items = []
                for msg in state.conversation_history[-5:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            
            # Analyze the ambiguity
            ambiguity_analysis = self.analyze_ambiguity(state.question)
            
            # Generate clarification request
            clarification_request = self.generate_clarification(state.question, history_text)
            
            # Add domain-specific suggestions
            suggestions = self.suggest_specific_questions(state.question)
            if suggestions:
                clarification_request += "\n\nSpecific questions that might help:\n"
                for suggestion in suggestions[:3]:  # Limit to 3 suggestions
                    clarification_request += f"• {suggestion}\n"
            
            # Update state
            state.needs_clarification = True
            state.clarification_request = clarification_request
            state.final_answer = clarification_request
            state.confidence_score = 0.8  # High confidence in clarification request
            state.sources = []
            state.agent_used = "clarification_agent"
            state.agent_actions.append("Generated clarification request")
            
            # Store ambiguity analysis in metadata
            state.agent_actions.append(f"Ambiguity analysis: {ambiguity_analysis}")
            
            logger.info("Clarification Agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Clarification Agent: {e}")
            state.final_answer = "I need more information to answer your question effectively. Could you please provide more specific details?"
            state.confidence_score = 0.5
            state.sources = []
            state.agent_used = "clarification_agent"
            state.needs_clarification = True
            return state
    
    def is_follow_up_clarification(self, question: str, history: List[Dict[str, Any]]) -> bool:
        """Check if this is a follow-up that provides clarification"""
        if not history:
            return False
        
        # Check if the last assistant message was a clarification request
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                clarification_phrases = [
                    "could you please",
                    "can you specify",
                    "what do you mean",
                    "clarification",
                    "more specific",
                    "which"
                ]
                if any(phrase in content for phrase in clarification_phrases):
                    return True
                break
        
        return False 