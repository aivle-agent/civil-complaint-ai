from src.models.state import CivilComplaintState
from src.nodes.civi_complaint_node import civil_complaint_node


def test_civi_complaint_node():
    # Given
    initial_state: CivilComplaintState = {
        "user_question": "Test Question",
        "retry_count": 0,
    }

    # When
    result = civil_complaint_node(initial_state)

    # Then
    assert result["retry_count"] == 0
    # Add more assertions if the node does more logic
