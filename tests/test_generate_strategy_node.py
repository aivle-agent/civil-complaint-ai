import pytest
from src.models.state import CivilComplaintState
from src.nodes.generate_strategy_node import generate_strategy_node

def test_generate_strategy_node():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "retry_count": 0
    }
    
    # When
    result = generate_strategy_node(state)
    
    # Then
    assert "strategy" in result
    assert result["strategy"] == "Strategy for: Refined: Test Question"
