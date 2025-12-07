"""
Configuration module for loading environment variables and API keys.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variables.
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return api_key
