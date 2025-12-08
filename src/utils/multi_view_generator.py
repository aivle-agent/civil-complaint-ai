# -*- coding: utf-8 -*-
"""
Multi-View 생성기 모듈
문서를 3가지 관점(법령/사례/종합)으로 분류하고 LLM으로 요약 생성
"""

import logging
from typing import List, Dict

from src.utils.kanana_generator import llm_generate

logger = logging.getLogger(__name__)


def llm_summarize_block(title: str, question: str, docs: List[str]) -> str:
    """
    주어진 문서들을 7줄 이내로 요약
    
    Args:
        title: 요약 블록 제목 (예: "관련 법령 및 조항")
        question: 원본 질문
        docs: 요약할 문서 목록
        
    Returns:
        요약된 텍스트
    """
    if not docs:
        return ""
    
    # 상위 8개 문서, 각 800자 제한
    ctx = "\n\n".join(d[:800] for d in docs[:8])
    
    prompt = f"""아래 자료는 '{title}'에 해당하는 참고자료입니다.
민원 답변을 작성하는 데 필요한 핵심 내용만 7줄 이내로 요약하십시오.

[질문]
{question}

[참고자료]
{ctx}

[요약]"""
    
    try:
        result = llm_generate(prompt, temperature=0.5)
        return result.strip()
    except Exception as e:
        logger.warning(f"[MULTI-VIEW] Summarization failed: {e}")
        return ""


def build_views(question: str, docs: List[str]) -> Dict[str, str]:
    """
    문서를 law/case/mixed 3가지 view로 분류 및 요약
    
    Args:
        question: 원본 질문
        docs: 검색된 문서 목록
        
    Returns:
        Dict with keys: "law", "case", "mixed"
    """
    # 법령 관련 문서 필터링
    law_docs = [
        d for d in docs 
        if any(k in d for k in ["법", "조례", "시행령", "조항", "규정"])
    ]
    
    # 사례 관련 문서 필터링
    case_docs = [
        d for d in docs 
        if any(k in d for k in ["민원", "사례", "판결", "판례", "신청인", "피신청인"])
    ]
    
    # 종합 문서 (상위 12개)
    mixed_docs = docs[:min(len(docs), 12)]
    
    logger.info(f"[MULTI-VIEW] Building views - law:{len(law_docs)}, case:{len(case_docs)}, mixed:{len(mixed_docs)}")
    print(f"[MULTI-VIEW] Building views - law:{len(law_docs)}, case:{len(case_docs)}, mixed:{len(mixed_docs)}")
    
    # 각 view 요약 생성
    print("[MULTI-VIEW] Generating law summary...")
    law_summary = llm_summarize_block("관련 법령 및 조항", question, law_docs)
    
    print("[MULTI-VIEW] Generating case summary...")
    case_summary = llm_summarize_block("유사 사례 및 판례", question, case_docs)
    
    print("[MULTI-VIEW] Generating mixed summary...")
    mixed_summary = llm_summarize_block("종합 참고자료", question, mixed_docs)
    
    return {
        "law": law_summary,
        "case": case_summary,
        "mixed": mixed_summary,
    }
