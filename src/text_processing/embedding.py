"""
Embedding generation module for complaint narratives.
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingGenerator:
    """Handles generation of embeddings for text chunks."""
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
            device (str): Device to use for computation ('cuda' or 'cpu')
            batch_size (int): Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            np.ndarray: Matrix of embeddings (n_texts x embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def get_embedding_info(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Get information about the embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            
        Returns:
            Dict[str, Any]: Dictionary containing embedding information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_embeddings': embeddings.shape[0],
            'device': self.device,
            'dtype': str(embeddings.dtype)
        }
    
    def compute_similarities(self, embeddings: np.ndarray, indices: List[tuple]) -> List[float]:
        """
        Compute cosine similarities between pairs of embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            indices (List[tuple]): List of (i,j) pairs to compute similarities for
            
        Returns:
            List[float]: List of similarity scores
        """
        similarities = []
        for i, j in indices:
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(float(similarity))
        return similarities 