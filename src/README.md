# Scripts Directory

This directory contains Python scripts for the Intelligent Complaint Analysis project.

## Structure

- `preprocessing/`: Contains scripts for data preprocessing
  - `data_loader.py`: Functions to load and validate the CFPB dataset
  - `text_cleaner.py`: Functions to clean and normalize complaint text
  - `data_filter.py`: Functions to filter the dataset by product categories
  
- `embedding/`: Contains scripts for text embedding and chunking
  - `text_chunker.py`: Functions to split long narratives into chunks
  - `embedder.py`: Functions to generate embeddings from text
  
- `vector_db/`: Contains scripts for vector database operations
  - `db_manager.py`: Functions to create and manage the vector database
  - `indexer.py`: Functions to index embeddings in the vector database
  - `retriever.py`: Functions to retrieve relevant documents from the vector database
  
- `rag/`: Contains scripts for the RAG agent
  - `agent.py`: Implementation of the RAG agent
  - `prompt_templates.py`: Templates for LLM prompts
  - `llm_interface.py`: Interface to the language model
  
- `api/`: Contains scripts for the API and interface
  - `app.py`: Streamlit application for the user interface
  - `api.py`: FastAPI implementation for the backend API

## Usage

Most scripts are designed to be imported as modules, but some can be run directly:

```
# Run the data preprocessing pipeline
python scripts/preprocessing/run_pipeline.py

# Start the Streamlit interface
python scripts/api/app.py
```

## Development Guidelines

1. Follow PEP 8 style guidelines
2. Add docstrings to all functions and classes
3. Include type hints for function parameters and return values
4. Write unit tests for all functions in the `test/` directory 