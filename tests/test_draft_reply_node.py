from unittest.mock import patch, MagicMock
from src.models.state import CivilComplaintState
from src.nodes.draft_reply_node import draft_reply_node


def test_draft_reply_node():
    """Test that draft_reply_node properly generates draft answer and updates state."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "strategy": "Test Strategy",
        "retry_count": 0,
        "is_verified": False,
    }

    # Mock the OpenAI API call and config
    mock_response = MagicMock()
    mock_response.content = "Mock draft answer based on the test strategy"
    
    with patch("src.nodes.draft_reply_node.get_openai_api_key", return_value="mock-api-key"), \
         patch("src.nodes.draft_reply_node.ChatOpenAI") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        # When
        result = draft_reply_node(state)

    # Then - Validate state structure and content
    # 1. Check that 'draft_answer' key exists in result
    assert "draft_answer" in result, "Result must contain 'draft_answer' key"
    
    # 2. Check that draft_answer is a non-empty string
    assert isinstance(result["draft_answer"], str), "Draft answer must be a string"
    assert len(result["draft_answer"]) > 0, "Draft answer must not be empty"
    
    # 3. Verify the draft answer contains meaningful content
    assert result["draft_answer"] == "Mock draft answer based on the test strategy"
    
    # 4. Ensure no unexpected keys are added
    assert set(result.keys()) == {"draft_answer"}, "Result should only contain 'draft_answer' key"
    
    print("✓ State validation passed: draft_answer properly generated and stored")
