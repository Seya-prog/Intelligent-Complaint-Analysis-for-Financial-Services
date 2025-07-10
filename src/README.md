# Source Code Directory

This directory contains the core implementation of the Intelligent Complaint Analysis system.

## Directory Structure

### `text_processing/`
Text processing utilities for the RAG pipeline:
- `chunking.py`: Document chunking strategies
- `embedding.py`: Text embedding generation
- `vector_store.py`: Vector storage and retrieval

### `rag/`
RAG pipeline implementation:
- `generator.py`: Response generation using LLM
- `pipeline.py`: Main RAG pipeline orchestration
- `retriever.py`: Context retrieval from vector store

### Core Scripts
- `evaluate_rag.py`: Evaluation tools and metrics
- `process_complaints.py`: Complaint data processing utilities

## Implementation Details

### Text Processing
- Implements efficient document chunking strategies
- Handles text embedding generation using transformer models
- Manages vector storage and similarity search

### RAG Pipeline
- Retrieves relevant context based on user queries
- Generates responses using retrieved context
- Ensures responses are grounded in actual complaint data

### Evaluation
The evaluation system measures:
- Response accuracy and relevance
- Context retrieval quality
- Response completeness
- Source citation accuracy

## Development Guidelines

1. Code Style
   - Follow PEP 8 guidelines
   - Use type hints for all functions
   - Add docstrings for all modules and functions

2. Testing
   - Write unit tests for new functionality
   - Ensure tests pass before committing
   - Use pytest for testing

3. Documentation
   - Keep README files updated
   - Document complex algorithms
   - Include usage examples

4. Version Control
   - Create feature branches for new work
   - Write clear commit messages
   - Review code before merging 