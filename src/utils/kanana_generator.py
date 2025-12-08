# -*- coding: utf-8 -*-
"""
LLM 텍스트 생성 모듈 (OpenAI API 사용)
로컬 Kanana 모델 대신 OpenAI GPT-4o-mini API를 사용
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from src.config import get_openai_api_key

logger = logging.getLogger(__name__)

# 전역 싱글톤 (lazy loading)
_llm_instance = None


def get_llm() -> ChatOpenAI:
    """
    OpenAI LLM 인스턴스 반환 (최초 호출 시 생성)
    
    Returns:
        ChatOpenAI 인스턴스
    """
    global _llm_instance
    
    if _llm_instance is None:
        logger.info("[LLM] Initializing OpenAI GPT-4o-mini")
        _llm_instance = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=get_openai_api_key()
        )
    
    return _llm_instance


def llm_generate(
    prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
) -> str:
    """
    OpenAI API로 텍스트 생성
    
    Args:
        prompt: 입력 프롬프트
        max_tokens: 최대 생성 토큰 수 (현재 미사용, API 제한)
        temperature: 샘플링 온도
        
    Returns:
        생성된 텍스트
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=get_openai_api_key()
        )
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"[LLM] Generation failed: {e}")
        raise


def is_model_available() -> bool:
    """
    LLM이 사용 가능한지 확인 (테스트/CI 환경용)
    """
    try:
        api_key = get_openai_api_key()
        return bool(api_key)
    except Exception:
        return False


# 호환성을 위한 별칭
kanana_generate = llm_generate
get_kanana_model = get_llm
