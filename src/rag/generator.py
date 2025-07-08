"""
Generator component for producing responses using an LLM.
"""
from typing import List, Optional
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from .retriever import RetrievedChunk

@dataclass
class GeneratorConfig:
    """Configuration for the Generator component."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_huggingface_token() -> str:
    """Load HuggingFace token from environment or .env files.
    
    Returns:
        The HuggingFace token if found.
        
    Raises:
        ValueError: If token cannot be found in any location.
    """
    # Try loading from environment first
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
        
    # Try loading from .env file in current or parent directories
    current_dir = Path.cwd()
    env_paths = [current_dir / ".env", current_dir.parent / ".env"]
    
    for env_path in env_paths:
        if env_path.exists():
            logger.info(f"Found .env file at {env_path}")
            load_dotenv(env_path)
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token:
                logger.info("Successfully loaded token from .env file")
                return token
                
    raise ValueError(
        "HuggingFace token not found in environment or .env files. "
        "Please set HUGGINGFACE_TOKEN environment variable."
    )

class Generator:
    """Generator component that produces responses using an LLM."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize the generator.

        Args:
            config: Optional GeneratorConfig to override defaults.
        """
        # If no configuration is provided, use defaults
        self.config = config or GeneratorConfig()
        
        logger.info(f"Initializing Generator with model: {self.config.model_name}")
        
        # Load HuggingFace token
        self.token = load_huggingface_token()
        
        # Initialize client
        self.client = InferenceClient(
            model=self.config.model_name,
            token=self.token
        )
        
        # Define the prompt template for RAG
        self.prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer,
state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string."""
        return "\n\n".join(f"[Score: {chunk.score:.2f}] {chunk.content}" for chunk in chunks)

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Generate a response for the given question using retrieved chunks as context."""
        try:
            # Format the context and create the prompt
            context = self._format_context(chunks)
            prompt = self.prompt_template.format(context=context, question=question)
            
            logger.info(f"Generating response for question: {question}")
            logger.info(f"Using {len(chunks)} chunks as context")
            
            # Create chat messages format
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful financial analyst assistant that provides accurate information based on the given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate response using chat completion
            result = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                
            )
            # Extract assistant message text
            if hasattr(result, "choices") and result.choices:
                return result.choices[0].message["content"]
            return str(result)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 