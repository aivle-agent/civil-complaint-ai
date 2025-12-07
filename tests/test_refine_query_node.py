from src.models.state import CivilComplaintState
from src.nodes.refine_query_node import refine_query_node

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
    assert isinstance(result["refined_question"], str), "refined_question must be a string"
    assert len(result["refined_question"]) > 0, "refined_question must not be empty"
    
    # Check that quality_scores and strategy are also generated
    assert "quality_scores" in result, "Result must contain 'quality_scores' key"
    assert "strategy" in result, "Result must contain 'strategy' key"
    assert isinstance(result["strategy"], str), "strategy must be a string"
    
    print(f"✓ refined_question: {result['refined_question'][:50]}...")
    print(f"✓ strategy generated: {len(result['strategy'])} chars")
