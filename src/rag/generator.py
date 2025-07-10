"""
Generator component for producing responses using an LLM.
"""
from typing import List, Optional, Iterator, Any, Dict
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from .retriever import RetrievedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneratorConfig:
    """Configuration for the Generator component."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    system_prompt: str = "You are a helpful financial analyst assistant that provides accurate information based on the given context."
    rag_prompt_template: str = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer,
state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

def load_huggingface_token() -> str:
    """Load HuggingFace token from environment or .env files."""
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

@dataclass
class Generator:
    """Generator component that produces responses using an LLM."""
    
    config: GeneratorConfig
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize the generator with optional config override."""
        self.config = config or GeneratorConfig()
        
        logger.info(f"Initializing Generator with model: {self.config.model_name}")
        
        # Load HuggingFace token
        self.token = load_huggingface_token()
        
        # Initialize client
        self.client = InferenceClient(
            model=self.config.model_name,
            token=self.token
        )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string."""
        return "\n\n".join(f"[Score: {chunk.score:.2f}] {chunk.content}" for chunk in chunks)

    def _create_messages(self, question: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
        """Create the message list for the chat completion."""
        context = self._format_context(chunks)
        user_prompt = self.config.rag_prompt_template.format(context=context, question=question)
        
        return [
            {
                "role": "system",
                "content": self.config.system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Generate a response for the given question using retrieved chunks as context."""
        try:
            logger.info(f"Generating response for question: {question}")
            logger.info(f"Using {len(chunks)} chunks as context")
            
            # Create chat messages
            messages = self._create_messages(question, chunks)
            
            # Generate response using chat completion
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p
            )
            
            content = response.choices[0].message.content
            return content if content is not None else ""
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def stream_generate(self, question: str, chunks: List[RetrievedChunk]) -> Iterator[str]:
        """Stream generate a response for the given question using retrieved chunks as context.
        
        Args:
            question: The question to answer
            chunks: Retrieved context chunks to use
            
        Yields:
            String tokens/deltas from the model response
        """
        try:
            logger.info(f"Stream generating response for question: {question}")
            logger.info(f"Using {len(chunks)} chunks as context")
            
            # Create chat messages
            messages = self._create_messages(question, chunks)
            
            # Stream response using chat completion
            for chunk in self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                stream=True
            ):
                # Extract content from the chunk
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
                    
        except Exception as e:
            logger.error(f"Error stream generating response: {str(e)}")
            raise 