# Notebooks Directory

This directory contains Jupyter notebooks for the Intelligent Complaint Analysis project.

## Structure

- `1_exploratory_data_analysis.ipynb`: Initial exploration of the CFPB dataset
  - Data loading and inspection
  - Statistical analysis of complaint distribution
  - Text length analysis
  - Visualization of complaint categories
  
- `2_data_preprocessing.ipynb`: Data cleaning and preparation
  - Filtering by product categories
  - Text cleaning and normalization
  - Handling missing values
  - Saving processed data
  
- `3_text_embedding.ipynb`: Text embedding and chunking
  - Text chunking strategies
  - Embedding model selection and evaluation
  - Visualization of embeddings
  
- `4_vector_database.ipynb`: Vector database setup and querying
  - Vector database initialization
  - Document indexing
  - Query testing and evaluation
  
- `5_rag_agent_development.ipynb`: RAG agent development and testing
  - Prompt engineering
  - LLM integration
  - Agent evaluation
  
- `6_performance_evaluation.ipynb`: System evaluation
  - Query response time analysis
  - Accuracy evaluation
  - User satisfaction metrics

## Usage

These notebooks are designed to be run in sequence, but each can also be run independently if the necessary data files are available.

To run a notebook:
1. Start Jupyter: `jupyter notebook`
2. Navigate to the notebooks directory
3. Open the desired notebook
4. Run cells in sequence

## Dependencies

All notebooks require the dependencies listed in the project's `requirements.txt` file. 