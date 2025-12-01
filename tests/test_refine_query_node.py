from src.models.state import CivilComplaintState
from src.nodes.refine_query_node import (
    refine_query_node,
    compute_question_quality,
    generate_refinement_suggestions,
)


def test_refine_query_node_basic():
    """Test basic functionality of refine_query_node"""
    # Given
    initial_state: CivilComplaintState = {
        "user_question": "아파트 층간소음 민원은 어떻게 넣나요?",
        "retry_count": 0,
    }

    # When
    result = refine_query_node(initial_state)

    # Then
    assert "refined_question" in result
    assert "quality_scores" in result
    assert result["refined_question"]  # Not empty
    assert isinstance(result["quality_scores"], dict)


def test_quality_scores_data_types():
    """Test that quality_scores has correct structure and data types"""
    # Given
    question = "2024년 11월 15일 아파트 층간소음 문제를 신고하고 싶습니다."

    # When
    scores = compute_question_quality(question)

    # Then
    # Check all expected keys are present
    expected_keys = {
        "clarity",
        "specific",
        "fact_ratio",
        "noise",
        "focus",
        "law_situation_sim",
    }
    assert set(scores.keys()) == expected_keys

    # Check all values are floats
    for key, value in scores.items():
        assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    # Check all values are in range [0.0, 1.0]
    for key, value in scores.items():
        assert (
            0.0 <= value <= 1.0
        ), f"{key} should be in [0.0, 1.0], got {value}"


def test_quality_scores_clarity():
    """Test clarity scoring"""
    # High clarity (has who/when/where/what)
    clear_question = "2024년 11월 언제 어디서 누가 무엇을 어떻게 했나요?"
    scores_clear = compute_question_quality(clear_question)
    assert scores_clear["clarity"] >= 0.5

    # Low clarity (lacks details)
    unclear_question = "문제가 있어요"
    scores_unclear = compute_question_quality(unclear_question)
    assert scores_unclear["clarity"] < 0.5


def test_quality_scores_specific():
    """Test specific scoring"""
    # High specificity (has numbers and dates)
    specific_question = "2024년 11월 15일 오후 3시에 100만원 결제했어요"
    scores_specific = compute_question_quality(specific_question)
    assert scores_specific["specific"] > 0.0

    # Low specificity (no numbers/dates)
    vague_question = "문제가 발생했어요"
    scores_vague = compute_question_quality(vague_question)
    assert scores_vague["specific"] == 0.0


def test_quality_scores_fact_ratio():
    """Test fact_ratio scoring"""
    # High fact ratio (factual language)
    factual_question = "신청 처리가 확인되지 않습니다. 문의드립니다."
    scores_factual = compute_question_quality(factual_question)
    assert scores_factual["fact_ratio"] >= 0.5

    # Low fact ratio (emotional language)
    emotional_question = "정말 화가나고 짜증나고 억울해요!!!"
    scores_emotional = compute_question_quality(emotional_question)
    assert scores_emotional["fact_ratio"] < 0.5


def test_quality_scores_noise():
    """Test noise scoring"""
    # High noise (excessive punctuation)
    noisy_question = "왜!!!!! 안되나요????!!!! 정말!!!!!"
    scores_noisy = compute_question_quality(noisy_question)
    assert scores_noisy["noise"] > 0.5

    # Low noise (clean)
    clean_question = "처리 상태를 확인하고 싶습니다."
    scores_clean = compute_question_quality(clean_question)
    assert scores_clean["noise"] < 0.3


def test_quality_scores_focus():
    """Test focus scoring"""
    # High focus (1-2 sentences)
    focused_question = "아파트 층간소음 민원 처리 방법을 알려주세요."
    scores_focused = compute_question_quality(focused_question)
    assert scores_focused["focus"] >= 0.9

    # Low focus (many sentences)
    unfocused_question = (
        "문제가 있어요. 저기요. 그리고 또 이것도 문제고. "
        "저것도 문제고. 이것도 안되고. 저것도 안되고. "
        "정말 많은 문제가 발생했어요. 어떻게 해야 하나요."
    )
    scores_unfocused = compute_question_quality(unfocused_question)
    assert scores_unfocused["focus"] < 0.5


def test_quality_scores_law_similarity():
    """Test law_situation_sim scoring"""
    # High legal similarity
    legal_question = "행정 규정에 따른 민원 신고 제도 개선을 요청합니다."
    scores_legal = compute_question_quality(legal_question)
    assert scores_legal["law_situation_sim"] > 0.0

    # Low legal similarity
    casual_question = "이것 좀 도와주세요"
    scores_casual = compute_question_quality(casual_question)
    assert scores_casual["law_situation_sim"] == 0.0


def test_empty_question():
    """Test handling of empty question"""
    # Given
    empty_question = ""

    # When
    scores = compute_question_quality(empty_question)

    # Then
    assert scores["clarity"] == 0.0
    assert scores["specific"] == 0.0
    assert scores["fact_ratio"] == 0.0
    assert scores["noise"] == 1.0  # High noise for empty
    assert scores["focus"] == 0.0
    assert scores["law_situation_sim"] == 0.0


def test_very_long_question():
    """Test handling of very long question"""
    # Given
    long_question = "질문입니다. " * 100

    # When
    scores = compute_question_quality(long_question)

    # Then
    # Should still return valid scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # Focus should be low due to many sentences
    assert scores["focus"] < 0.5


def test_special_characters():
    """Test handling of special characters"""
    # Given
    special_question = "민원@#$%신청 방법은???"

    # When
    scores = compute_question_quality(special_question)

    # Then
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_refinement_suggestions_low_clarity():
    """Test that suggestions are generated for low clarity"""
    # Given
    question = "문제가 있어요"
    scores = compute_question_quality(question)

    # When
    suggestions = generate_refinement_suggestions(question, scores)

    # Then
    assert "명확성 개선" in suggestions


def test_refinement_suggestions_low_specific():
    """Test that suggestions are generated for low specificity"""
    # Given
    question = "문제가 발생했어요"
    scores = compute_question_quality(question)

    # When
    suggestions = generate_refinement_suggestions(question, scores)

    # Then
    assert "구체성 개선" in suggestions


def test_refinement_suggestions_good_quality():
    """Test that positive feedback is given for good quality"""
    # Given
    question = "2024년 11월 15일 오후 3시 아파트 층간소음 신고 처리 확인 요청합니다."
    scores = compute_question_quality(question)

    # When
    suggestions = generate_refinement_suggestions(question, scores)

    # Then
    assert "잘 작성되었습니다" in suggestions or len(suggestions) < 50
