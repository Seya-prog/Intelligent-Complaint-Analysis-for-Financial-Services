"""
Text chunking module for processing complaint narratives.
"""
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm

class TextChunker:
    """Handles text chunking of complaint narratives."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        return self.text_splitter.split_text(text)
    
    def process_complaints(self, df: pd.DataFrame, text_column: str = 'cleaned_narrative') -> tuple:
        """
        Process all complaints and create chunks with metadata.
        
        Args:
            df (pd.DataFrame): DataFrame containing complaints
            text_column (str): Name of the column containing text to chunk
            
        Returns:
            tuple: (list of chunks, list of chunk metadata)
        """
        all_chunks = []
        chunk_metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking narratives"):
            if pd.isna(row[text_column]):
                continue
                
            chunks = self.chunk_text(row[text_column])
            
            # Store chunks and their metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'complaint_id': row['Complaint ID'],
                    'product': row['Product'],
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                    'original_length': len(row[text_column])
                })
        
        return all_chunks, chunk_metadata
    
    def get_chunk_statistics(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Calculate statistics about the chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            Dict[str, Any]: Dictionary containing statistics
        """
        lengths = [len(chunk) for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(chunks) if chunks else 0,
            'min_length': min(lengths) if chunks else 0,
            'max_length': max(lengths) if chunks else 0
        } 