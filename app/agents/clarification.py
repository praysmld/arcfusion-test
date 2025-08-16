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
        # Use GPT-4o mini for clarification
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using GPT-4o mini for clarification
            temperature=0.2,  # Lower temperature for more consistent clarifications
            api_key=settings.openai_api_key
        )
        
        # Enhanced clarification prompt based on clarification prompting guide
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized clarification agent designed to identify ambiguous or underspecified questions and request clarification to provide better answers.

Follow this 4-step process for clarification:

1. IDENTIFY AMBIGUITY: Pinpoint specific areas where the question is unclear, vague, or open to multiple interpretations.
   - Vague qualifiers ("good", "enough", "best", "high") without context
   - Missing scope or constraints
   - Unclear references ("it", "this", "that")
   - Multiple possible interpretations
   - Missing essential parameters
   - Overly broad questions

2. PROMPT FOR INFORMATION: Ask specific, targeted questions to gather the necessary details.
   - Make your questions clear and focused on the identified areas of ambiguity
   - Provide options when appropriate to help the user understand what you need
   - Be concise but thorough in your requests

3. EXPLAIN WHY: Briefly explain why this clarification will help provide a better answer.
   - Show how different interpretations could lead to different answers
   - Explain how additional context will improve the response quality

4. STRUCTURE YOUR RESPONSE:
   - Begin with "I'd like to help you with this question about [topic]."
   - Explain: "To provide the most accurate answer, I need to clarify a few things:"
   - List your specific questions
   - End with "Once you provide these details, I'll be able to give you a more precise and helpful answer."

Context: The system can search academic papers on text-to-SQL and LLM research, or search the web for current information.

Previous Conversation:
{history}

Question: {question}

Clarification Request:"""),
            ("human", "{question}")
        ])
        
        # Secondary analysis prompt for deeper understanding of ambiguity
        self.ambiguity_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing questions for ambiguity and vagueness. 
            
Examine the following question and identify all potential sources of ambiguity or vagueness.

Analyze the question for:
1. Vague qualifiers or terms
2. Missing context or parameters
3. Unclear references
4. Multiple possible interpretations
5. Domain-specific ambiguities

Return your analysis in this format:
AMBIGUITY_SCORE: [1-10, where 10 is extremely ambiguous]
PRIMARY_ISSUE: [Main source of ambiguity]
SPECIFIC_AMBIGUITIES: [List specific ambiguous elements]
CLARIFICATION_NEEDS: [What specific information is needed]"""),
            ("human", "Question: {question}\n\nContext: {context}")
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
    
    def analyze_ambiguity(self, question: str, context: str = "text-to-sql research and LLM capabilities") -> Dict[str, Any]:
        """Perform a detailed analysis of question ambiguity using GPT-4o mini"""
        try:
            # Use the LLM for a more sophisticated ambiguity analysis
            response = self.llm.invoke(
                self.ambiguity_analysis_prompt.format(
                    question=question,
                    context=context
                )
            )
            
            analysis_text = response.content.strip()
            
            # Parse the structured response
            analysis = {
                "ambiguity_score": 5,  # Default value
                "primary_issue": "",
                "specific_ambiguities": [],
                "clarification_needs": []
            }
            
            # Extract structured information from the response
            for line in analysis_text.split('\n'):
                if line.startswith("AMBIGUITY_SCORE:"):
                    try:
                        score = int(line.replace("AMBIGUITY_SCORE:", "").strip())
                        analysis["ambiguity_score"] = score
                    except ValueError:
                        pass
                elif line.startswith("PRIMARY_ISSUE:"):
                    analysis["primary_issue"] = line.replace("PRIMARY_ISSUE:", "").strip()
                elif line.startswith("SPECIFIC_AMBIGUITIES:"):
                    ambiguities = line.replace("SPECIFIC_AMBIGUITIES:", "").strip()
                    if ambiguities:
                        analysis["specific_ambiguities"] = [a.strip() for a in ambiguities.split(',')]
                elif line.startswith("CLARIFICATION_NEEDS:"):
                    needs = line.replace("CLARIFICATION_NEEDS:", "").strip()
                    if needs:
                        analysis["clarification_needs"] = [n.strip() for n in needs.split(',')]
            
            logger.info(f"Ambiguity analysis completed with score: {analysis['ambiguity_score']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in ambiguity analysis: {e}")
            
            # Fall back to basic analysis
            basic_analysis = self._basic_ambiguity_check(question)
            return {
                "ambiguity_score": basic_analysis["score"],
                "primary_issue": basic_analysis["primary_issue"],
                "specific_ambiguities": basic_analysis["ambiguous_terms"],
                "clarification_needs": []
            }
    
    def _basic_ambiguity_check(self, question: str) -> Dict[str, Any]:
        """Perform a basic rule-based ambiguity check as fallback"""
        result = {
            "score": 0,
            "primary_issue": "",
            "ambiguous_terms": []
        }
        
        # Check for vague qualifiers
        vague_terms = ['good', 'bad', 'best', 'worst', 'enough', 'sufficient', 'high', 'low', 'many', 'few', 'better', 'worse']
        found_vague = [term for term in vague_terms if term in question.lower().split()]
        if found_vague:
            result["score"] += 3
            result["ambiguous_terms"].extend(found_vague)
            if not result["primary_issue"]:
                result["primary_issue"] = "Vague qualifiers"
        
        # Check for unclear references
        unclear_refs = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        found_refs = [ref for ref in unclear_refs if ref in question.lower().split()]
        if found_refs:
            result["score"] += 2
            result["ambiguous_terms"].extend(found_refs)
            if not result["primary_issue"]:
                result["primary_issue"] = "Unclear references"
        
        # Check for question length (very short questions often lack context)
        if len(question.split()) < 5:
            result["score"] += 2
            if not result["primary_issue"]:
                result["primary_issue"] = "Too brief"
        
        # Check for missing specific parameters in technical questions
        tech_terms = ['accuracy', 'performance', 'model', 'dataset', 'benchmark', 'compare']
        if any(term in question.lower() for term in tech_terms) and len(question.split()) < 10:
            result["score"] += 2
            if not result["primary_issue"]:
                result["primary_issue"] = "Missing technical parameters"
        
        return result
    
    def suggest_specific_questions(self, question: str, ambiguity_analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific questions based on the ambiguity analysis"""
        suggestions = []
        
        # Add suggestions based on clarification needs from analysis
        if "clarification_needs" in ambiguity_analysis and ambiguity_analysis["clarification_needs"]:
            for need in ambiguity_analysis["clarification_needs"]:
                if need and len(need) > 3:  # Basic validation
                    suggestions.append(f"Could you specify {need}?")
        
        # Add domain-specific suggestions
        if "text-to-sql" in question.lower() or "sql" in question.lower():
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
        
        # Add suggestions for LLM-related questions
        if "llm" in question.lower() or "language model" in question.lower():
            suggestions.extend([
                "Which specific model or model family are you referring to?",
                "Are you interested in capabilities, limitations, or comparisons?",
                "What specific task or application are you considering?"
            ])
        
        # Limit to avoid overwhelming the user
        return suggestions[:4]
    
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
            
            # Analyze the ambiguity with enhanced analysis
            context = "The system can search academic papers on text-to-SQL and LLM research, or search the web for current information."
            ambiguity_analysis = self.analyze_ambiguity(state.question, context)
            
            # Generate clarification request
            clarification_request = self.generate_clarification(state.question, history_text)
            
            # Add domain-specific suggestions
            suggestions = self.suggest_specific_questions(state.question, ambiguity_analysis)
            if suggestions:
                clarification_request += "\n\nHere are some specific questions that might help:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    clarification_request += f"{i}. {suggestion}\n"
            
            # Update state
            state.needs_clarification = True
            state.clarification_request = clarification_request
            state.final_answer = clarification_request
            state.confidence_score = 0.9  # High confidence in clarification request
            state.sources = []
            state.agent_used = "clarification_agent"
            state.agent_actions.append("Generated clarification request using GPT-4o mini")
            
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
        if not history or len(history) < 2:
            return False
        
        # Check if the last assistant message was a clarification request
        assistant_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                assistant_message = msg.get("content", "").lower()
                break
        
        if not assistant_message:
            return False
        
        # Check for clarification indicators in the assistant's message
        clarification_phrases = [
            "could you please", "can you specify", "what do you mean",
            "clarification", "more specific", "which", "to clarify",
            "i need to clarify", "please provide", "can you tell me more",
            "i'd like to help you", "to provide the most accurate answer"
        ]
        
        if any(phrase in assistant_message for phrase in clarification_phrases):
            # Now check if the user's current question seems to be providing clarification
            # This is a simple heuristic - the actual determination would be more complex
            if len(question.split()) > 5:  # Not just a simple yes/no
                return True
        
        return False 