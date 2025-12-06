from src.models.state import CivilComplaintState
from src.nodes.civil_complaint_node import civil_complaint_node


def test_civi_complaint_node_with_input():
    # Given
    initial_state: CivilComplaintState = {
        "user_question": "Test Question",
        "retry_count": 0,
    }

    # When
    result = civil_complaint_node(initial_state)

    # Then
    assert result["retry_count"] == 0
    # Should not return user_question if it was already present
    assert "user_question" not in result


def test_civi_complaint_node_without_input():
    # Given
    initial_state: CivilComplaintState = {
        "user_question": "",
        "retry_count": 0,
    }

    # When
    result = civil_complaint_node(initial_state)

    # Then
    assert result["retry_count"] == 0
    # Should return a user_question loaded from CSV
    assert "user_question" in result
    assert result["user_question"]  # Should not be empty
    print(f"Loaded question: {result['user_question']}")
