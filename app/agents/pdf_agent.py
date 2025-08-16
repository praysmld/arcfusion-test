import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from app.config import settings
from app.models.schemas import AgentState, DocumentChunk
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


class PDFAgent:
    """Agent for retrieving and answering questions from PDF documents"""
    
    def __init__(self, vectorstore_service: VectorStoreService):
        self.vectorstore = vectorstore_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant specializing in text-to-SQL and large language model research. 
Your job is to answer questions based on the provided research paper excerpts.

Guidelines:
1. Base your answers strictly on the provided context from the research papers
2. Cite specific papers, authors, and findings when possible
3. If the question cannot be answered from the provided context, clearly state this
4. Be precise with numbers, metrics, and experimental results
5. Maintain scientific objectivity and accuracy
6. When referencing specific results, include the source paper information

Context from Research Papers:
{context}

Previous Conversation:
{history}

Question: {question}

Answer:"""),
            ("human", "{question}")
        ])
    
    def search_documents(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Search for relevant documents in the vector store"""
        try:
            chunks = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(chunks)} relevant document chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def extract_answer_with_sources(self, chunks: List[DocumentChunk], question: str, history: str = "") -> tuple[str, float, List[Dict[str, Any]]]:
        """Extract answer from document chunks with confidence scoring"""
        try:
            if not chunks:
                return "I couldn't find relevant information in the available research papers to answer your question.", 0.0, []
            
            # Prepare context from chunks
            context_parts = []
            sources = []
            
            for i, chunk in enumerate(chunks):
                context_parts.append(f"[Source {i+1}] {chunk.content}")
                
                # Prepare source information
                source_info = {
                    "chunk_id": i + 1,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "metadata": chunk.metadata,
                    "similarity_score": chunk.similarity_score
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate answer
            response = self.llm.invoke(
                self.qa_prompt.format(
                    context=context,
                    history=history,
                    question=question
                )
            )
            
            answer = response.content.strip()
            
            # Simple confidence scoring based on answer characteristics
            confidence = self._calculate_confidence(answer, chunks)
            
            logger.info(f"Generated answer with confidence: {confidence:.2f}")
            return answer, confidence, sources
            
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return f"An error occurred while processing your question: {str(e)}", 0.0, []
    
    def _calculate_confidence(self, answer: str, chunks: List[DocumentChunk]) -> float:
        """Calculate confidence score for the answer"""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on answer characteristics
            if "according to" in answer.lower() or "found that" in answer.lower():
                confidence += 0.2
            
            if any(chunk.similarity_score and chunk.similarity_score > 0.8 for chunk in chunks):
                confidence += 0.2
            
            if len(chunks) >= 3:
                confidence += 0.1
            
            # Decrease confidence for uncertain language
            uncertain_phrases = ["might", "could", "possibly", "unclear", "cannot be determined"]
            if any(phrase in answer.lower() for phrase in uncertain_phrases):
                confidence -= 0.2
            
            if "couldn't find" in answer.lower() or "no information" in answer.lower():
                confidence = 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def process_query(self, state: AgentState) -> AgentState:
        """Process a query using PDF retrieval and generation"""
        try:
            logger.info(f"PDF Agent processing query: {state.question}")
            
            # Search for relevant documents
            chunks = self.search_documents(state.question, k=5)
            
            if not chunks:
                state.final_answer = "I couldn't find relevant information in the available research papers to answer your question. You might want to try a web search for more current information."
                state.confidence_score = 0.1
                state.sources = []
                state.agent_used = "pdf_agent"
                return state
            
            # Store PDF results in state
            state.pdf_results = [
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "similarity_score": chunk.similarity_score
                }
                for chunk in chunks
            ]
            
            # Prepare conversation history
            history_text = ""
            if state.conversation_history:
                history_items = []
                for msg in state.conversation_history[-3:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    history_items.append(f"{role}: {content}")
                history_text = "\n".join(history_items)
            
            # Generate answer
            answer, confidence, sources = self.extract_answer_with_sources(
                chunks, state.question, history_text
            )
            
            # Update state
            state.final_answer = answer
            state.confidence_score = confidence
            state.sources = sources
            state.agent_used = "pdf_agent"
            state.agent_actions.append(f"PDF search found {len(chunks)} relevant chunks")
            
            logger.info("PDF Agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in PDF Agent: {e}")
            state.final_answer = f"An error occurred while searching the research papers: {str(e)}"
            state.confidence_score = 0.0
            state.sources = []
            state.agent_used = "pdf_agent"
            return state
    
    def search_by_author_or_paper(self, author_name: str = None, paper_title: str = None) -> List[DocumentChunk]:
        """Search for documents by specific author or paper title"""
        try:
            metadata_filter = {}
            
            if author_name:
                # Search in file names or content for author references
                chunks = self.vectorstore.similarity_search(f"author:{author_name}", k=10)
                return chunks
            
            if paper_title:
                chunks = self.vectorstore.similarity_search(paper_title, k=10)
                return chunks
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching by author/paper: {e}")
            return [] 