import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import httpx
from tavily import TavilyClient

from app.config import settings
from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Agent for searching the web when information is not available in PDFs"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        
        # Initialize Tavily client if API key is available
        self.tavily_client = None
        if settings.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=settings.tavily_api_key)
                logger.info("Tavily client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Tavily client: {e}")
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant. Your job is to synthesize information from web search results to answer user questions.

Guidelines:
1. Use the provided search results to answer the question comprehensively
2. Cite sources with URLs when possible
3. Be factual and objective
4. If the search results don't contain enough information, say so clearly
5. Prioritize recent and authoritative sources
6. Distinguish between facts and opinions/speculation

Search Results:
{search_results}

Previous Conversation:
{history}

Question: {question}

Synthesized Answer:"""),
            ("human", "{question}")
        ])
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using Tavily API"""
        try:
            if not self.tavily_client:
                logger.warning("Tavily client not available, using mock results")
                return self._get_mock_results(query)
            
            # Perform search with Tavily
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            results = []
            if 'results' in response:
                for result in response['results']:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'content': result.get('content', ''),
                        'published_date': result.get('published_date', ''),
                        'score': result.get('score', 0.0)
                    })
            
            logger.info(f"Found {len(results)} web search results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return self._get_mock_results(query)
    
    def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Provide mock search results when web search is not available"""
        return [
            {
                'title': f'Mock result for: {query}',
                'url': 'https://example.com/mock-result',
                'content': f'This is a mock search result for the query: {query}. In a real implementation, this would be replaced with actual web search results from Tavily API.',
                'published_date': '2024-01-01',
                'score': 0.8
            }
        ]
    
    def synthesize_answer(self, search_results: List[Dict[str, Any]], question: str, history: str = "") -> tuple[str, float, List[Dict[str, Any]]]:
        """Synthesize an answer from search results"""
        try:
            if not search_results:
                return "I couldn't find relevant information through web search.", 0.1, []
            
            # Format search results for the prompt
            formatted_results = []
            sources = []
            
            for i, result in enumerate(search_results):
                formatted_results.append(
                    f"[Source {i+1}] {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Content: {result['content']}\n"
                    f"Published: {result.get('published_date', 'Unknown')}\n"
                )
                
                sources.append({
                    'title': result['title'],
                    'url': result['url'],
                    'content': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    'published_date': result.get('published_date', 'Unknown'),
                    'score': result.get('score', 0.0)
                })
            
            search_results_text = "\n\n".join(formatted_results)
            
            # Generate synthesized answer
            response = self.llm.invoke(
                self.synthesis_prompt.format(
                    search_results=search_results_text,
                    history=history,
                    question=question
                )
            )
            
            answer = response.content.strip()
            
            # Calculate confidence based on search results quality
            confidence = self._calculate_confidence(search_results, answer)
            
            logger.info(f"Synthesized web search answer with confidence: {confidence:.2f}")
            return answer, confidence, sources
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return f"An error occurred while processing web search results: {str(e)}", 0.0, []
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score for web search answer"""
        try:
            confidence = 0.6  # Base confidence for web results
            
            # Increase confidence based on result quality
            if len(search_results) >= 3:
                confidence += 0.1
            
            # Check for high-scoring results
            high_score_results = [r for r in search_results if r.get('score', 0) > 0.8]
            if high_score_results:
                confidence += 0.1
            
            # Check for recent results
            recent_results = [r for r in search_results if '2024' in r.get('published_date', '')]
            if recent_results:
                confidence += 0.1
            
            # Decrease confidence for uncertain language
            uncertain_phrases = ["unclear", "unknown", "might", "possibly", "could not find"]
            if any(phrase in answer.lower() for phrase in uncertain_phrases):
                confidence -= 0.2
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating web search confidence: {e}")
            return 0.5
    
    def process_query(self, state: AgentState) -> AgentState:
        """Process a query using web search"""
        try:
            logger.info(f"Web Agent processing query: {state.question}")
            
            # Enhance query for better search results
            enhanced_query = self._enhance_search_query(state.question)
            
            # Perform web search
            search_results = self.search_web(enhanced_query, max_results=5)
            
            # Store web results in state
            state.web_results = search_results
            
            # Prepare conversation history
            history_text = ""
            if state.conversation_history:
                history_items = []
                for msg in state.conversation_history[-3:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            
            # Synthesize answer from search results
            answer, confidence, sources = self.synthesize_answer(
                search_results, state.question, history_text
            )
            
            # Update state
            state.final_answer = answer
            state.confidence_score = confidence
            state.sources = sources
            state.agent_used = "web_agent"
            state.agent_actions.append(f"Web search found {len(search_results)} results")
            
            logger.info("Web Agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Web Agent: {e}")
            state.final_answer = f"An error occurred while performing web search: {str(e)}"
            state.confidence_score = 0.0
            state.sources = []
            state.agent_used = "web_agent"
            return state
    
    def _enhance_search_query(self, query: str) -> str:
        """Enhance the search query for better results"""
        try:
            # Add current year for recent results
            if "recent" in query.lower() or "latest" in query.lower() or "current" in query.lower():
                return f"{query} 2024"
            
            # Add specific terms for AI/ML queries
            ai_terms = ["ai", "artificial intelligence", "machine learning", "llm", "gpt", "language model"]
            if any(term in query.lower() for term in ai_terms):
                return f"{query} AI ML 2024"
            
            return query
            
        except Exception as e:
            logger.error(f"Error enhancing search query: {e}")
            return query 