# -*- coding: utf-8 -*-
"""
Draft Reply Node
Multi-View + Critic + Self-Refine 알고리즘을 사용한 민원 답변 생성
"""

import logging

from src.models.state import CivilComplaintState
from src.utils.kanana_retriever import retrieve_with_kanana
from src.utils.multi_view_generator import build_views
from src.utils.answer_generator import generate_best_answer

logger = logging.getLogger(__name__)


def draft_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Multi-View + Critic + Self-Refine 파이프라인으로 민원 답변 생성
    
    파이프라인:
    1. RAG 검색 (Kanana retriever)
    2. Multi-View 요약 생성 (법령/사례/종합)
    3. 다중 후보 답변 생성 (3가지 전략)
    4. Critic 평가 및 정렬
    5. Self-Refine 2회
    
    Args:
        state: Current state containing refined_question
        
    Returns:
        Updated state with draft_answer, views, candidates
    """
    print("---DRAFT REPLY NODE (Multi-View)---")
    
    refined_question = state.get("refined_question")
    if not refined_question:
        refined_question = state.get("user_question", "")
    
    # 1. RAG 검색
    print(f"[1/5] Retrieving documents for: {refined_question}...")
    try:
        retrieved_docs = retrieve_with_kanana(refined_question, top_k=5)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        for i, doc in enumerate(retrieved_docs):
            print(f"[{i+1}] {doc['doc'][:100]}... (Meta: {doc.get('meta')})")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        retrieved_docs = []
    
    # 문서가 없으면 fallback
    if not retrieved_docs:
        print("No documents retrieved, using fallback response.")
        return {
            "draft_answer": "관련 문서를 찾을 수 없어 답변을 생성할 수 없습니다.",
            "retrieved_documents": [],
            "views": {},
            "candidates": [],
            "rag_context": "",
        }
    
    # 문서 텍스트 추출
    docs = [d["doc"] for d in retrieved_docs]
    
    # 2. Multi-View 요약 생성
    print("[2/5] Building multi-view summaries...")
    try:
        views = build_views(refined_question, docs)
        print(f"Views generated - law: {len(views.get('law', ''))} chars, "
              f"case: {len(views.get('case', ''))} chars, "
              f"mixed: {len(views.get('mixed', ''))} chars")
    except Exception as e:
        print(f"View generation failed: {e}")
        views = {"law": "", "case": "", "mixed": ""}
    
    # 3-5. 후보 생성 → Critic 평가 → Self-Refine
    print("[3-5/5] Generating candidates, scoring, and refining...")
    try:
        final_answer, scored_candidates = generate_best_answer(refined_question, views)
        print(f"Generated {len(scored_candidates)} candidates, final answer: {len(final_answer)} chars")
    except Exception as e:
        print(f"Answer generation failed: {e}")
        final_answer = "답변 생성 중 오류가 발생했습니다."
        scored_candidates = []
    
    # RAG context 구성
    context_text = ""
    if retrieved_docs:
        context_text = "\n\n[참고 자료]\n"
        for i, doc in enumerate(retrieved_docs[:5]):  # 상위 5개만
            context_text += f"{i+1}. {doc['doc'][:200]}...\n"
    
    print(f"---DRAFT REPLY NODE COMPLETE---")
    print(f"Final answer preview: {final_answer[:100]}...")
    
    return {
        "draft_answer": final_answer,
        "retrieved_documents": retrieved_docs,
        "views": views,
        "candidates": scored_candidates,
        "rag_context": context_text,
    }
