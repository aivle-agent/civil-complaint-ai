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
    assert result["refined_question"] == initial_state["user_question"]
