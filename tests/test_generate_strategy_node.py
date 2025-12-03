from unittest.mock import patch, MagicMock
from src.models.state import CivilComplaintState
from src.nodes.generate_strategy_node import generate_strategy_node


def test_generate_strategy_node():
    """Test that generate_strategy_node properly generates strategy and updates state."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "retry_count": 0,
        "is_verified": False,
    }

    # Mock the OpenAI API call and config
    mock_response = MagicMock()
    mock_response.content = "Mock strategy for answering the test question"
    
    with patch("src.nodes.generate_strategy_node.get_openai_api_key", return_value="mock-api-key"), \
         patch("src.nodes.generate_strategy_node.ChatOpenAI") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        # When
        result = generate_strategy_node(state)

    # Then - Validate state structure and content
    # 1. Check that 'strategy' key exists in result
    assert "strategy" in result, "Result must contain 'strategy' key"
    
    # 2. Check that strategy is a non-empty string
    assert isinstance(result["strategy"], str), "Strategy must be a string"
    assert len(result["strategy"]) > 0, "Strategy must not be empty"
    
    # 3. Verify the strategy contains meaningful content
    assert result["strategy"] == "Mock strategy for answering the test question"
    
    # 4. Ensure no unexpected keys are added
    assert set(result.keys()) == {"strategy"}, "Result should only contain 'strategy' key"
    
    print("✓ State validation passed: strategy properly generated and stored")
