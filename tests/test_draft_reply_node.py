from src.models.state import CivilComplaintState
from src.nodes.draft_reply_node import draft_reply_node


def test_draft_reply_node():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "strategy": "Test Strategy",
        "retry_count": 0,
    }

    # When
    result = draft_reply_node(state)

    # Then
    assert "draft_answer" in result
    assert result["draft_answer"] == "Draft Answer based on Test Strategy"
