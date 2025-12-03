from unittest.mock import patch, MagicMock
from src.models.state import CivilComplaintState
from src.nodes.verify_reply_node import verify_reply_node


def test_verify_reply_node_pass():
    """Test that verify_reply_node passes verification and properly updates state."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "strategy": "Test Strategy",
        "draft_answer": "Test Draft Answer that is comprehensive and helpful",
        "retry_count": 0,
        "is_verified": False,
    }

    # Mock the OpenAI API call and config to return a passing verification
    mock_response = MagicMock()
    mock_response.content = "판정: 통과\n피드백: 답변이 명확하고 친절합니다."
    
    with patch("src.nodes.verify_reply_node.get_openai_api_key", return_value="mock-api-key"), \
         patch("src.nodes.verify_reply_node.ChatOpenAI") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        # When
        result = verify_reply_node(state)

    # Then - Validate state structure and content
    # 1. Check that all required keys exist
    assert "is_verified" in result, "Result must contain 'is_verified' key"
    assert "verification_feedback" in result, "Result must contain 'verification_feedback' key"
    assert "retry_count" in result, "Result must contain 'retry_count' key"
    
    # 2. Check that is_verified is a boolean and True
    assert isinstance(result["is_verified"], bool), "is_verified must be a boolean"
    assert result["is_verified"] is True, "Verification should pass"
    
    # 3. Check that verification_feedback is a non-empty string
    assert isinstance(result["verification_feedback"], str), "Feedback must be a string"
    assert len(result["verification_feedback"]) > 0, "Feedback must not be empty"
    assert "통과" in result["verification_feedback"], "Feedback should indicate pass"
    
    # 4. Check that retry_count is properly incremented
    assert isinstance(result["retry_count"], int), "retry_count must be an integer"
    assert result["retry_count"] == 1, "retry_count should be incremented to 1"
    
    # 5. Ensure no unexpected keys are added
    expected_keys = {"is_verified", "verification_feedback", "retry_count"}
    assert set(result.keys()) == expected_keys, f"Result should only contain {expected_keys}"
    
    print("✓ State validation passed: verification PASSED with proper state updates")


def test_verify_reply_node_fail():
    """Test that verify_reply_node fails verification and properly updates state."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "strategy": "Test Strategy",
        "draft_answer": "Short answer",  # Intentionally short/poor answer
        "retry_count": 0,
        "is_verified": False,
    }

    # Mock the OpenAI API call and config to return a failing verification
    mock_response = MagicMock()
    mock_response.content = "판정: 재작성필요\n피드백: 답변이 너무 짧고 구체적이지 않습니다."
    
    with patch("src.nodes.verify_reply_node.get_openai_api_key", return_value="mock-api-key"), \
         patch("src.nodes.verify_reply_node.ChatOpenAI") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        # When
        result = verify_reply_node(state)

    # Then - Validate state structure and content
    # 1. Check that all required keys exist
    assert "is_verified" in result, "Result must contain 'is_verified' key"
    assert "verification_feedback" in result, "Result must contain 'verification_feedback' key"
    assert "retry_count" in result, "Result must contain 'retry_count' key"
    
    # 2. Check that is_verified is a boolean and False
    assert isinstance(result["is_verified"], bool), "is_verified must be a boolean"
    assert result["is_verified"] is False, "Verification should fail"
    
    # 3. Check that verification_feedback is a non-empty string with failure indication
    assert isinstance(result["verification_feedback"], str), "Feedback must be a string"
    assert len(result["verification_feedback"]) > 0, "Feedback must not be empty"
    assert "재작성" in result["verification_feedback"] or "실패" in result["verification_feedback"], \
        "Feedback should indicate failure/retry needed"
    
    # 4. Check that retry_count is properly incremented
    assert isinstance(result["retry_count"], int), "retry_count must be an integer"
    assert result["retry_count"] == 1, "retry_count should be incremented to 1"
    
    # 5. Ensure no unexpected keys are added
    expected_keys = {"is_verified", "verification_feedback", "retry_count"}
    assert set(result.keys()) == expected_keys, f"Result should only contain {expected_keys}"
    
    print("✓ State validation passed: verification FAILED with proper state updates")
