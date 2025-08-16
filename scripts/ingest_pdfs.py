#!/usr/bin/env python3
"""
PDF Ingestion Script

This script processes PDF files and ingests them into the vector database.
It can be run independently or called from the main application.

Usage:
    python scripts/ingest_pdfs.py [--directory path/to/pdfs] [--clear-existing]
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.pdf_ingestion import PDFIngestionService
from app.services.vectorstore import VectorStoreService
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest PDF files into vector database")
    parser.add_argument(
        "--directory",
        type=str,
        default=settings.pdf_directory,
        help=f"Directory containing PDF files (default: {settings.pdf_directory})"
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing documents from vector store before ingesting"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Text chunk size (default: {settings.chunk_size})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Text chunk overlap (default: {settings.chunk_overlap})"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate directory
        pdf_directory = Path(args.directory)
        if not pdf_directory.exists():
            logger.error(f"Directory {args.directory} does not exist")
            return 1
        
        logger.info(f"Starting PDF ingestion from {args.directory}")
        
        # Initialize services
        pdf_service = PDFIngestionService()
        vectorstore_service = VectorStoreService()
        
        # Clear existing documents if requested
        if args.clear_existing:
            logger.info("Clearing existing documents from vector store")
            try:
                if vectorstore_service.clear_collection():
                    logger.info("Successfully cleared existing documents")
                else:
                    logger.warning("Could not clear collection (may be new)")
            except Exception as e:
                logger.warning(f"Collection clearing issue (continuing): {e}")
        
        # Get vector store info before ingestion
        before_info = vectorstore_service.get_collection_info()
        logger.info(f"Vector store before ingestion: {before_info}")
        
        # Load and process PDFs
        logger.info("Loading PDF documents...")
        documents = pdf_service.load_all_pdfs(args.directory)
        
        if not documents:
            logger.warning("No documents found to ingest")
            return 0
        
        # Extract metadata
        metadata = pdf_service.extract_metadata(documents)
        logger.info(f"Processed documents metadata: {metadata}")
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        success = vectorstore_service.add_documents(documents)
        
        if not success:
            logger.error("Failed to add documents to vector store")
            return 1
        
        # Get vector store info after ingestion
        after_info = vectorstore_service.get_collection_info()
        logger.info(f"Vector store after ingestion: {after_info}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {len(metadata['files'])}")
        logger.info(f"Total chunks created: {metadata['total_chunks']}")
        logger.info(f"Total characters: {metadata['total_characters']:,}")
        logger.info(f"Documents in vector store: {after_info.get('document_count', 'unknown')}")
        
        # File breakdown
        logger.info("\nFile breakdown:")
        for filename, file_info in metadata['files'].items():
            logger.info(f"  {filename}:")
            logger.info(f"    Chunks: {file_info['chunks']}")
            logger.info(f"    Pages: {len(file_info['pages'])}")
            logger.info(f"    Characters: {file_info['characters']:,}")
        
        logger.info("\nIngestion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        return 1


def test_ingestion():
    """Test the ingestion with a sample query"""
    try:
        logger.info("Testing ingestion with sample query...")
        
        vectorstore_service = VectorStoreService()
        
        # Test query
        test_query = "text-to-SQL generation accuracy"
        results = vectorstore_service.similarity_search(test_query, k=3)
        
        logger.info(f"Test query: '{test_query}'")
        logger.info(f"Found {len(results)} results:")
        
        for i, chunk in enumerate(results):
            logger.info(f"\nResult {i+1}:")
            logger.info(f"  Similarity: {chunk.similarity_score:.3f}")
            logger.info(f"  Source: {chunk.metadata.get('file_name', 'unknown')}")
            logger.info(f"  Content preview: {chunk.content[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        # Run a test query
        logger.info("\n" + "=" * 60)
        logger.info("TESTING INGESTION")
        logger.info("=" * 60)
        test_success = test_ingestion()
        
        if test_success:
            logger.info("Test completed successfully!")
        else:
            logger.warning("Test failed - check logs for details")
            exit_code = 1
    
    sys.exit(exit_code) 