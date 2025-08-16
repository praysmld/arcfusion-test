import logging
import warnings
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

# Suppress deprecation warnings for Chroma
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.vectorstores")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.config import settings
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector embeddings and similarity search"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            # Ensure directory exists
            import os
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            
            # Configure ChromaDB settings
            chroma_settings = ChromaSettings(
                persist_directory=settings.chroma_persist_directory,
                anonymized_telemetry=False
            )
            
            # Initialize client first to ensure connection
            client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=chroma_settings
            )
            
            # Check if collection exists, create it if not
            collection_name = "langchain"
            try:
                collection = client.get_or_create_collection(name=collection_name)
                logger.info(f"Using collection '{collection_name}' with {collection.count()} documents")
            except Exception as coll_error:
                logger.warning(f"Error with collection: {coll_error}")
            
            # Initialize Chroma vector store
            self.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embeddings,
                client_settings=chroma_settings,
                collection_name=collection_name
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents to add to vector store")
                return False
            
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            
            # Persist the vector store
            self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Perform similarity search and return ranked results"""
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                return []
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            # Convert to DocumentChunk objects
            chunks = []
            for doc, score in results:
                chunk = DocumentChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=1 - score  # Convert distance to similarity
                )
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} similar documents for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection"""
        try:
            if not self.vectorstore:
                return {"error": "Vector store not initialized"}
            
            # Get collection statistics
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "embedding_model": settings.embedding_model,
                "persist_directory": settings.chroma_persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            if self.vectorstore:
                try:
                    # Try to delete the collection if it exists
                    self.vectorstore.delete_collection()
                    logger.info("Existing collection deleted")
                except Exception as delete_error:
                    # Collection might not exist yet, which is fine
                    logger.info(f"Collection deletion (expected if new): {delete_error}")
                
                # Reinitialize
                self._initialize_vectorstore()
                
                logger.info("Vector store collection cleared/reinitialized")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def search_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        limit: int = 10
    ) -> List[DocumentChunk]:
        """Search documents by metadata filters"""
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                return []
            
            # Use Chroma's where filter
            results = self.vectorstore.get(
                where=metadata_filter,
                limit=limit
            )
            
            chunks = []
            if results and 'documents' in results:
                for i, content in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if 'metadatas' in results else {}
                    chunk = DocumentChunk(
                        content=content,
                        metadata=metadata,
                        similarity_score=None
                    )
                    chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} documents matching metadata filter")
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return [] 
    
    def get_all_chunks(self, limit: Optional[int] = None) -> List[DocumentChunk]:
        """Retrieve all chunks from the vector store"""
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                self._initialize_vectorstore()  # Try to reinitialize
                if not self.vectorstore:
                    return []
            
            # Get all documents from the collection
            # Use a large limit if none specified, or the specified limit
            max_limit = limit if limit else 10000  # Reasonable upper bound
            
            try:
                # Try using the langchain interface first
                results = self.vectorstore.get(limit=max_limit)
                
                chunks = []
                if results and 'documents' in results:
                    for i, content in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if 'metadatas' in results else {}
                        chunk = DocumentChunk(
                            content=content,
                            metadata=metadata,
                            similarity_score=None
                        )
                        chunks.append(chunk)
                
                logger.info(f"Retrieved {len(chunks)} chunks from vector store")
                return chunks
                
            except Exception as inner_e:
                logger.error(f"Error getting chunks: {inner_e}")
                return []
            
        except Exception as e:
            logger.error(f"Error retrieving all chunks: {e}")
            return [] 