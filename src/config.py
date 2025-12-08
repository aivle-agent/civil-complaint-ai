"""
Configuration module for loading environment variables and API keys.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# LLM 모델 설정 (전역)
# 모델 변경 시 이 값만 수정하면 전체 프로젝트에 적용됨
# ============================================

LLM_MODEL = "gpt-5.1"


def get_llm_model() -> str:
    """
    사용할 LLM 모델명 반환
    
    변경 방법: 위의 LLM_MODEL 상수를 수정
    예시: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
    """
    return LLM_MODEL


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
