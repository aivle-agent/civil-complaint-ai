# -*- coding: utf-8 -*-
"""
답변 생성기 모듈
다중 후보 생성, Critic 평가, Self-Refine 로직
"""

import logging
from typing import List, Dict, Any, Tuple

from src.utils.kanana_generator import llm_generate
from src.utils.prompts import (
    GEN_STRATEGIES,
    build_answer_prompt,
    strip_to_answer,
)

logger = logging.getLogger(__name__)


def generate_candidates(question: str, views: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    3가지 전략으로 후보 답변 생성
    
    Args:
        question: 원본 질문
        views: Multi-view 요약 (law, case, mixed)
        
    Returns:
        후보 답변 목록 [{"strategy": str, "view": str, "answer": str}, ...]
    """
    candidates = []
    
    for strat in GEN_STRATEGIES:
        view_name = strat["view"]
        view_summary = views.get(view_name, "")
        prompt = build_answer_prompt(question, view_summary)
        
        print(f"[GEN] Generating candidate with strategy: {strat['name']}...")
        
        try:
            full_text = llm_generate(
                prompt,
                temperature=strat["temp"],
            )
            answer = strip_to_answer(full_text, "")
            
            candidates.append({
                "strategy": strat["name"],
                "view": view_name,
                "answer": answer,
            })
            print(f"[GEN] Generated answer: {len(answer)} chars")
        except Exception as e:
            logger.warning(f"[GEN] Candidate generation failed for {strat['name']}: {e}")
            print(f"[GEN] Failed: {e}")
            continue
    
    logger.info(f"[GEN] Generated {len(candidates)} candidate answers")
    return candidates


def critic_score(question: str, answer: str) -> Tuple[float, str]:
    """
    답변을 0.0~1.0으로 평가
    
    Args:
        question: 원본 질문
        answer: 평가할 답변
        
    Returns:
        (점수, 이유) 튜플
    """
    prompt = f"""다음 민원 답변을 0.0~1.0 사이 점수로 평가하십시오.

[질문]
{question}

[답변]
{answer}

평가 기준:
- 공공 민원 답변 형식(검토내용 1,2,3,4 구조) 준수 여부
- 민원 요지 반영 정도
- 검토 과정 및 조치 계획의 구체성
- 관련 법령·조례·규정 등의 적절한 언급 여부
- 공무원 답변 톤·표현의 적절성

출력 형식:
점수: <0.00~1.00>
설명: <짧은 이유>"""
    
    try:
        print("[CRITIC] Evaluating answer...")
        result = llm_generate(prompt, temperature=0.3)
        
        score = 0.5
        reason = ""
        
        for line in result.splitlines():
            if "점수" in line:
                try:
                    val = float(line.split(":", 1)[1].strip().split()[0])
                    score = max(0.0, min(1.0, val))
                except Exception:
                    pass
            if line.startswith("설명"):
                reason = line.split(":", 1)[1].strip()
        
        print(f"[CRITIC] Score: {score:.2f}")
        return score, reason
    except Exception as e:
        logger.warning(f"[CRITIC] Scoring failed: {e}")
        return 0.5, "평가 실패"


def self_refine(question: str, base_answer: str, other_answers: List[str]) -> str:
    """
    base 답변을 다른 후보 참고하여 정제
    
    Args:
        question: 원본 질문
        base_answer: 정제할 기본 답변
        other_answers: 참고할 다른 후보 답변들
        
    Returns:
        정제된 답변
    """
    others_block = "\n\n".join(
        f"- {a[:300].replace(chr(10), ' ')}" for a in other_answers if a
    )
    
    prompt = f"""아래는 민원에 대한 1차 답변입니다. 그리고 참고할 수 있는 다른 후보 답변 일부입니다.
이들을 참고하여 더 완성도 높고 공공 민원 양식에 부합하는 최종 답변으로 수정하십시오.

[질문]
{question}

[1차 답변]
{base_answer}

[다른 후보 답변 요약 일부]
{others_block if others_block else "별도 후보 없음"}

요구사항:
- '검토내용'으로 시작하는 1,2,3,4 구조를 유지하십시오.
- 민원의 현황·문제점, 검토 결과, 향후 조치, 문의처를 명확히 구분하십시오.
- 예시 문장이나 지시문은 포함하지 마십시오.

검토내용"""
    
    try:
        print("[REFINE] Refining answer...")
        result = llm_generate(prompt, temperature=0.6)
        refined = strip_to_answer(result, "")
        print(f"[REFINE] Refined: {len(refined)} chars")
        return refined
    except Exception as e:
        logger.warning(f"[REFINE] Self-refine failed: {e}")
        return base_answer


def generate_best_answer(
    question: str, 
    views: Dict[str, str]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    전체 파이프라인: 후보 생성 → Critic 평가 → Self-Refine 2회
    
    Args:
        question: 원본 질문
        views: Multi-view 요약
        
    Returns:
        (최종 답변, 평가된 후보 목록) 튜플
    """
    # 1. 후보 답변 생성
    print("[PIPELINE] Step 1: Generating candidates...")
    candidates = generate_candidates(question, views)
    
    if not candidates:
        logger.error("[PIPELINE] No candidates generated")
        return "답변 생성에 실패했습니다.", []
    
    # 2. Critic 점수 부여
    print("[PIPELINE] Step 2: Scoring candidates...")
    scored_candidates = []
    for c in candidates:
        score, reason = critic_score(question, c["answer"])
        scored_candidates.append({
            **c,
            "critic_score": score,
            "critic_reason": reason,
        })
    
    # 점수 기준 정렬
    scored_candidates.sort(key=lambda x: x["critic_score"], reverse=True)
    
    # 3. Best 선택 및 Self-Refine 2회
    best = scored_candidates[0]
    base_answer = best["answer"]
    other_answers = [c["answer"] for c in scored_candidates[1:]]
    
    print(f"[PIPELINE] Best candidate: {best['strategy']} (score: {best['critic_score']:.2f})")
    
    print("[PIPELINE] Step 3: Self-refine round 1...")
    refined_1 = self_refine(question, base_answer, other_answers)
    
    print("[PIPELINE] Step 4: Self-refine round 2...")
    refined_2 = self_refine(question, refined_1, other_answers)
    
    final_answer = refined_2.strip()
    
    print("[PIPELINE] Complete!")
    return final_answer, scored_candidates
