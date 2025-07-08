"""
Vector store module for managing FAISS index and metadata.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm

class VectorStore:
    """Handles FAISS index operations and metadata management."""
    
    def __init__(self, dimension: int):
        """
        Initialize the VectorStore.
        
        Args:
            dimension (int): Dimension of the vectors to store
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
    def add_vectors(self, vectors: np.ndarray):
        """
        Add vectors to the FAISS index.
        
        Args:
            vectors (np.ndarray): Matrix of vectors to add
        """
        self.index.add(vectors)
        
    def search(self, query_vector: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (np.ndarray): Query vector
            k (int): Number of results to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices)
        """
        return self.index.search(query_vector.reshape(1, -1), k)
    
    def save(self, vector_dir: Path, chunks: List[str], metadata: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Save the index, chunks, and metadata.
        
        Args:
            vector_dir (Path): Directory to save files in
            chunks (List[str]): List of text chunks
            metadata (List[Dict]): List of chunk metadata
            config (Dict): Configuration parameters
        """
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = vector_dir / 'complaints.index'
        faiss.write_index(self.index, str(faiss_path))
        
        # Save metadata and configuration
        index_metadata = {
            'config': config,
            'index_metadata': {
                'total_vectors': self.index.ntotal,
                'dimension': self.dimension,
                'index_type': 'IndexFlatL2'
            },
            'chunks_metadata': metadata
        }
        
        metadata_path = vector_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(index_metadata, f, indent=2)
            
        # Save chunks for reference
        chunks_path = vector_dir / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'metadata': metadata
            }, f)
    
    @classmethod
    def load(cls, vector_dir: Path) -> Tuple['VectorStore', Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """
        Load a saved vector store.
        
        Args:
            vector_dir (Path): Directory containing saved files
            
        Returns:
            Tuple: (VectorStore instance, config dict, chunks list, metadata list)
        """
        # Load metadata
        with open(vector_dir / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
            
        # Load chunks
        with open(vector_dir / 'chunks.pkl', 'rb') as f:
            chunks_dict = pickle.load(f)
            
        # Load index
        index = faiss.read_index(str(vector_dir / 'complaints.index'))
        
        # Create instance
        instance = cls(metadata_dict['index_metadata']['dimension'])
        instance.index = index
        
        return (
            instance,
            metadata_dict['config'],
            chunks_dict['chunks'],
            chunks_dict['metadata']
        ) 