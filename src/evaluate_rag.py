"""
Script to evaluate the RAG pipeline performance.
"""
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from rag import RAGPipeline, GeneratorConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test questions
TEST_QUESTIONS = [
    "What are the most common types of credit card complaints?",
    "How do customers typically describe issues with unauthorized charges?",
    "What patterns emerge in complaints about credit card fees?",
    "How do banks typically respond to billing disputes?",
    "What are common complaints about credit card application processes?",
    "How do customers express dissatisfaction with customer service?",
    "What issues do customers face with credit card rewards programs?",
    "How are payment posting delays typically reported?",
    "What concerns do customers raise about interest rates?",
    "How do customers describe problems with credit limit changes?"
]

def main():
    """Main evaluation function."""
    logger.info("Starting RAG pipeline evaluation")
    
    # Initialize pipeline with default config
    pipeline = RAGPipeline()
    
    # Process each question
    results = []
    for question in tqdm(TEST_QUESTIONS, desc="Processing questions"):
        try:
            # Retrieve chunks and generate response to capture sources
            chunks = pipeline.retriever.retrieve(question, k=5)
            response = pipeline.generator.generate(question=question, chunks=chunks)
            # Take up to first 2 chunk contents as retrieved sources
            retrieved_sources = " || ".join(chunk.content.replace("\n", " ")[:300] for chunk in chunks[:2])
            results.append({
                "question": question,
                "response": response,
                "retrieved_sources": retrieved_sources,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            results.append({
                "question": question,
                "response": str(e),
                "status": "error"
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path("data/evaluation_results.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main() 