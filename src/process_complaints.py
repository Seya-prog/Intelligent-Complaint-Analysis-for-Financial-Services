"""
Main script for processing complaints through chunking, embedding, and indexing.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
from text_processing import TextChunker, EmbeddingGenerator, VectorStore

def main(args):
    # Configuration
    config = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'model_name': args.model_name,
        'device': 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu',
        'batch_size': args.batch_size
    }
    
    print(f"Using device: {config['device']}")
    
    # Set up paths
    data_dir = Path(args.data_dir)
    input_file = data_dir / 'processed' / 'filtered_complaints.csv'
    vector_dir = data_dir / 'vector_store'
    
    # Load data
    print(f"Loading complaints from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} complaints")
    
    # Initialize processors
    chunker = TextChunker(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    
    embedder = EmbeddingGenerator(
        model_name=config['model_name'],
        device=config['device'],
        batch_size=config['batch_size']
    )
    
    # Process chunks
    print("\nChunking narratives...")
    chunks, chunk_metadata = chunker.process_complaints(df)
    chunk_stats = chunker.get_chunk_statistics(chunks)
    print(f"Created {chunk_stats['total_chunks']} chunks")
    print(f"Average chunk length: {chunk_stats['avg_length']:.1f} characters")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = embedder.generate_embeddings(chunks)
    embedding_info = embedder.get_embedding_info(embeddings)
    print(f"Generated {embedding_info['num_embeddings']} embeddings")
    print(f"Embedding dimension: {embedding_info['embedding_dim']}")
    
    # Create and save vector store
    print("\nCreating vector store...")
    vector_store = VectorStore(dimension=embedding_info['embedding_dim'])
    vector_store.add_vectors(embeddings)
    
    print("\nSaving vector store...")
    vector_store.save(vector_dir, chunks, chunk_metadata, config)
    print(f"Vector store saved to {vector_dir}")
    
    # Test search functionality
    if args.test_search:
        print("\nTesting search functionality...")
        test_idx = 0
        distances, indices = vector_store.search(embeddings[test_idx], k=3)
        
        print(f"\nQuery chunk: {chunks[test_idx][:200]}...")
        print("\nTop 3 similar chunks:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            print(f"\n{i}. Distance: {dist:.3f}")
            print(f"Text: {chunks[idx][:200]}...")
            print(f"Metadata: {chunk_metadata[idx]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process complaints for semantic search')
    
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the data')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Maximum size of each chunk in characters')
    parser.add_argument('--chunk_overlap', type=int, default=128,
                        help='Number of characters to overlap between chunks')
    parser.add_argument('--model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Name of the sentence-transformers model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--test_search', action='store_true',
                        help='Run a test search after indexing')
    
    args = parser.parse_args()
    main(args) 