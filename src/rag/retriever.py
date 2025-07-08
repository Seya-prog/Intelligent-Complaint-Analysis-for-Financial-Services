"""
Retriever component for finding relevant complaint chunks.
"""
from pathlib import Path
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """Data class for retrieved chunks with their metadata and similarity scores."""
    content: str
    metadata: Dict[str, Any]
    score: float

class Retriever:
    """Handles retrieval of relevant complaint chunks based on user queries."""
    
    def __init__(
        self,
        vector_store_path: Path,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = None,
        top_k: int = 5
    ):
        """
        Initialize the Retriever.
        
        Args:
            vector_store_path: Path to the vector store directory
            model_name: Name of the sentence transformer model to use
            device: Device to use for computation ('cuda' or 'cpu')
            top_k: Number of chunks to retrieve
        """
        self.vector_store_path = Path(vector_store_path)
        self.top_k = top_k
        self.device = device
        
        # Load the embedding model
        logger.info(f"Loading embedding model {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Load the vector store components
        self._load_vector_store()
        
        logger.info(f"Initialized Retriever with {len(self.chunks)} chunks")
    
    def _load_vector_store(self):
        """Load the vector store components from disk."""
        try:
            # Load the FAISS index
            index_path = self.vector_store_path / "complaints.index"
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks and metadata from pickle file
            chunks_path = self.vector_store_path / "chunks.pkl"
            with open(chunks_path, 'rb') as f:
                chunks_data = pickle.load(f)
                self.chunks = chunks_data['chunks']
                self.metadata = chunks_data['metadata']
            
            logger.info(f"Loaded {len(self.chunks)} chunks from vector store")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def retrieve(self, query: str, k: int = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: The user's question
            k: Optional number of chunks to retrieve, overrides top_k if provided
            
        Returns:
            List of RetrievedChunk objects containing relevant chunks and their metadata
        """
        try:
            # Use provided k or fall back to self.top_k
            k = k if k is not None else self.top_k
            
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query}")
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            
            # Perform similarity search
            logger.debug("Performing similarity search")
            distances, indices = self.index.search(
                query_embedding.cpu().numpy().astype('float32'),
                k=k
            )
            
            # Format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                # Check if index is valid
                if 0 <= idx < len(self.chunks):
                    chunk = RetrievedChunk(
                        content=self.chunks[idx],
                        metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                        score=float(1 - dist)
                    )
                    results.append(chunk)
                else:
                    logger.warning(f"Retrieved invalid index {idx}, skipping")
            
            logger.info(f"Retrieved {len(results)} chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise 