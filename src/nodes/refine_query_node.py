import re
from typing import Dict
from src.models.state import CivilComplaintState


def compute_question_quality(question: str) -> Dict[str, float]:
    """
    Compute question quality metrics using lightweight heuristics.
    Returns scores for: clarity, specific, fact_ratio, noise, focus, law_situation_sim
    All scores are in range [0.0, 1.0]
    """
    if not question or not question.strip():
        return {
            "clarity": 0.0,
            "specific": 0.0,
            "fact_ratio": 0.0,
            "noise": 1.0,  # High noise for empty
            "focus": 0.0,
            "law_situation_sim": 0.0,
        }

    # Clarity: Check for key question indicators (who/when/where/what)
    clarity_keywords = ["언제", "어디", "누가", "무엇", "어떻게", "왜"]
    clarity_count = sum(1 for kw in clarity_keywords if kw in question)
    clarity_score = min(clarity_count / 3.0, 1.0)  # Normalize to max 1.0

    # Specific: Count numbers and dates
    numbers = len(re.findall(r"\d+", question))
    dates = len(
        re.findall(
            r"\d{4}년|\d{1,2}월|\d{1,2}일|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", question
        )
    )
    specific_score = min((numbers + dates * 2) / 5.0, 1.0)

    # Fact ratio: Check for emotional keywords vs factual
    emotional_keywords = ["화가", "짜증", "억울", "불쾌", "무례", "최악", "!!!"]
    factual_keywords = ["발생", "신청", "요청", "문의", "확인", "처리", "개선"]
    emotional_count = sum(1 for kw in emotional_keywords if kw in question)
    factual_count = sum(1 for kw in factual_keywords if kw in question)
    total_indicators = emotional_count + factual_count
    fact_ratio = (
        factual_count / total_indicators if total_indicators > 0 else 0.5
    )  # Default neutral

    # Noise: Detect excessive punctuation/caps
    exclamation_count = question.count("!")
    question_marks = question.count("?")
    noise_score = min(
        (exclamation_count + question_marks) / 10.0, 1.0
    )  # More = higher noise

    # Focus: Check sentence count (fewer sentences = more focused)
    sentences = re.split(r"[.!?。]", question)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    focus_score = 1.0 if sentence_count <= 2 else max(0.0, 1.0 - (sentence_count / 10.0))

    # Law situation sim: Mock score based on legal keywords
    legal_keywords = ["법", "조례", "규정", "행정", "민원", "제도", "개선", "신고"]
    legal_count = sum(1 for kw in legal_keywords if kw in question)
    law_sim_score = min(legal_count / 3.0, 1.0)

    return {
        "clarity": round(clarity_score, 2),
        "specific": round(specific_score, 2),
        "fact_ratio": round(fact_ratio, 2),
        "noise": round(noise_score, 2),
        "focus": round(focus_score, 2),
        "law_situation_sim": round(law_sim_score, 2),
    }


def generate_refinement_suggestions(
    question: str, scores: Dict[str, float]
) -> str:
    """
    Generate refinement suggestions based on quality scores.
    """
    suggestions = []

    if scores["clarity"] < 0.5:
        suggestions.append(
            "명확성 개선: 상황을 더 구체적으로 설명해주세요 (언제, 어디서, 무엇을, 어떻게)."
        )

    if scores["specific"] < 0.3:
        suggestions.append("구체성 개선: 구체적인 날짜, 금액, 횟수 등을 추가해주세요.")

    if scores["fact_ratio"] < 0.4:
        suggestions.append(
            "객관성 개선: 감정 표현보다는 사실 중심으로 작성하면 더 효과적입니다."
        )

    if scores["noise"] > 0.5:
        suggestions.append(
            "간결성 개선: 불필요한 문장부호를 줄이고 핵심만 전달해주세요."
        )

    if scores["focus"] < 0.5:
        suggestions.append(
            "집중도 개선: 하나의 주요 문제에 집중하여 작성하면 더 좋습니다."
        )

    if not suggestions:
        suggestions.append("질문이 잘 작성되었습니다.")

    return " ".join(suggestions)


def refine_query_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Refines the user's question by analyzing quality and providing suggestions.
    """
    print("---REFINE QUERY NODE---")
    user_question = state["user_question"]

    # Compute quality scores
    quality_scores = compute_question_quality(user_question)
    print(f"Quality Scores: {quality_scores}")

    # Generate refinement suggestions
    suggestions = generate_refinement_suggestions(user_question, quality_scores)
    refined_question = f"{user_question}\n\n[개선 제안: {suggestions}]"

    return {"refined_question": refined_question, "quality_scores": quality_scores}
