# Test Directory

This directory contains test files for the Intelligent Complaint Analysis project.

## Structure

- `unit/`: Contains unit tests for individual components
  - `test_preprocessing.py`: Tests for data preprocessing functions
  - `test_embedding.py`: Tests for text embedding functions
  - `test_vector_store.py`: Tests for vector database operations
  - `test_rag_agent.py`: Tests for the RAG agent components
  
- `integration/`: Contains integration tests that verify multiple components working together
  - `test_end_to_end.py`: End-to-end test of the complete RAG pipeline
  - `test_api.py`: Tests for the API endpoints

- `fixtures/`: Contains test data and fixtures
  - `sample_complaints.csv`: Small sample of complaint data for testing
  - `mock_embeddings.pkl`: Mock embeddings for testing vector search

## Running Tests

To run all tests:
```
pytest
```

To run a specific test file:
```
pytest test/unit/test_preprocessing.py
```

To run tests with coverage report:
```
pytest --cov=scripts
```

## Test Strategy

1. Unit tests verify the correctness of individual functions and classes
2. Integration tests verify that components work together correctly
3. End-to-end tests verify the complete pipeline from data input to chatbot response 