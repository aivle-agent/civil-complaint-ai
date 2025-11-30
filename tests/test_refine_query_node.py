import pytest
from src.models.state import CivilComplaintState
from src.nodes.refine_query_node import refine_query_node

def test_refine_query_node():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "retry_count": 0
    }
    
    # When
    result = refine_query_node(state)
    
    # Then
    assert "refined_question" in result
    assert result["refined_question"] == "Refined: Test Question"
