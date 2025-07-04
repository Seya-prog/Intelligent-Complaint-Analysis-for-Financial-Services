# Intelligent Complaint Analysis for Financial Services

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) powered chatbot to transform customer complaint data into actionable insights for CrediTrust Financial, a digital finance company serving East African markets.

## Folder Structure
- **data/**: Contains raw and processed datasets
  - Raw CFPB complaint data
  - Filtered and cleaned complaint data
  - Vector embeddings and indices
- **notebooks/**: Jupyter notebooks for data exploration, model development and analysis
  - EDA notebooks
  - Data preprocessing notebooks
  - Model evaluation notebooks
- **scripts/**: Python scripts for the RAG pipeline components
  - Data preprocessing scripts
  - Text embedding and chunking utilities
  - Vector database management
  - RAG agent implementation
  - API and interface code
- **test/**: Unit and integration tests for the system components

## Project Goals
1. Build a RAG agent that allows internal users to query customer complaints using natural language
2. Decrease complaint trend identification time from days to minutes
3. Empower non-technical teams to get insights without data analyst support
4. Enable proactive problem identification based on customer feedback

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Download the CFPB complaint dataset
3. Run the data preprocessing notebook/script
4. Set up the vector database
5. Launch the chatbot interface

## Technologies Used
- Python 3.11+
- Sentence transformers for text embedding
- FAISS/ChromaDB for vector similarity search
- LangChain for RAG pipeline orchestration
- Streamlit for user interface
- Pandas/NumPy for data processing 