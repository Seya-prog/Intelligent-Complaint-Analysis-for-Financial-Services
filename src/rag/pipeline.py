"""
RAG pipeline that combines retrieval and generation.
"""
import logging
from pathlib import Path
from typing import Optional, List, Iterator

from .retriever import Retriever
from .generator import Generator, GeneratorConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG pipeline that combines retrieval and generation components.
    """
    
    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        generator_config: Optional[GeneratorConfig] = None
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_store_path: Path to the vector store. If None, uses default path.
            generator_config: Configuration for the generator. If None, uses defaults.
        """
        # Set default vector store path if none provided
        if vector_store_path is None:
            vector_store_path = Path("data/vector_store")  # Changed from "../data/vector_store"
        
        logger.info("Initializing RAG pipeline")
        logger.info(f"Using vector store at: {vector_store_path}")
        
        # Initialize components
        self.retriever = Retriever(vector_store_path=vector_store_path)
        self.generator = Generator(config=generator_config)
        
        logger.info("RAG pipeline initialized successfully")
    
    def process_query(self, query: str, num_chunks: int = 5) -> str:
        """Process a query through the RAG pipeline.
        
        Args:
            query: The query to process
            num_chunks: Number of chunks to retrieve
            
        Returns:
            Generated response based on retrieved context
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, k=num_chunks)
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        
        # Generate response
        response = self.generator.generate(chunks=chunks, question=query)
        logger.info("Generated response successfully")
        
        return response
        
    def stream_process_query(self, query: str, num_chunks: int = 5) -> Iterator[str]:
        """Process a query through the RAG pipeline with streaming response.
        
        Args:
            query: The query to process
            num_chunks: Number of chunks to retrieve
            
        Yields:
            Generated response tokens/deltas based on retrieved context
        """
        logger.info(f"Processing query with streaming: {query}")
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, k=num_chunks)
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        
        # Stream generate response
        for token in self.generator.stream_generate(chunks=chunks, question=query):
            yield token 