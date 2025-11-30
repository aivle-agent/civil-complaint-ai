import pytest
from src.models.state import CivilComplaintState
from src.nodes.final_answer_node import final_answer_node

def test_final_answer_node():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "draft_answer": "Test Draft",
        "retry_count": 0
    }
    
    # When
    result = final_answer_node(state)
    
    # Then
    assert "final_answer" in result
    assert result["final_answer"] == "Final Answer: Test Draft"
