import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)


class PDFIngestionService:
    """Service for ingesting and processing PDF documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file and return documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": "pdf"
                })
            
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content)
                })
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return []
    
    def load_all_pdfs(self, directory: str = None) -> List[Document]:
        """Load all PDF files from the specified directory"""
        if directory is None:
            directory = settings.pdf_directory
        
        pdf_dir = Path(directory)
        if not pdf_dir.exists():
            logger.error(f"PDF directory {directory} does not exist")
            return []
        
        all_documents = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        for pdf_file in pdf_files:
            documents = self.load_pdf(str(pdf_file))
            all_documents.extend(documents)
        
        # Chunk all documents
        chunked_documents = self.chunk_documents(all_documents)
        
        logger.info(f"Processed {len(pdf_files)} PDF files into {len(chunked_documents)} chunks")
        return chunked_documents
    
    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract metadata summary from processed documents"""
        metadata = {
            "total_chunks": len(documents),
            "files": {},
            "total_characters": 0
        }
        
        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            
            if file_name not in metadata["files"]:
                metadata["files"][file_name] = {
                    "chunks": 0,
                    "pages": set(),
                    "characters": 0
                }
            
            metadata["files"][file_name]["chunks"] += 1
            metadata["files"][file_name]["characters"] += len(doc.page_content)
            metadata["total_characters"] += len(doc.page_content)
            
            if "page" in doc.metadata:
                metadata["files"][file_name]["pages"].add(doc.metadata["page"])
        
        # Convert sets to lists for JSON serialization
        for file_info in metadata["files"].values():
            file_info["pages"] = sorted(list(file_info["pages"]))
        
        return metadata 