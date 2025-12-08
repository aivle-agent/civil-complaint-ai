from unittest.mock import patch, MagicMock
from src.models.state import CivilComplaintState
from src.nodes.draft_reply_node import draft_reply_node


def test_draft_reply_node():
    """Test that draft_reply_node properly generates draft answer and updates state."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "retry_count": 0,
    }

    # Mock the OpenAI API call and config
    mock_response = MagicMock()
    mock_response.content = "Mock draft answer based on the test question"
    
    with patch("src.nodes.draft_reply_node.get_openai_api_key", return_value="mock-api-key"), \
         patch("src.nodes.draft_reply_node.ChatOpenAI") as mock_llm, \
         patch("src.nodes.draft_reply_node.retrieve_with_kanana") as mock_retriever:
        
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        # Mock retriever return value
        mock_retriever.return_value = [
            {"doc": "Test Document 1", "meta": {"source": "test"}},
            {"doc": "Test Document 2", "meta": {"source": "test"}}
        ]
        
        # When
        result = draft_reply_node(state)

    # Then - Validate state structure and content
    # 1. Check that 'draft_answer' key exists in result
    assert "draft_answer" in result, "Result must contain 'draft_answer' key"
    
    # 2. Check that draft_answer is a non-empty string
    assert isinstance(result["draft_answer"], str), "Draft answer must be a string"
    assert len(result["draft_answer"]) > 0, "Draft answer must not be empty"
    
    # 3. Check for new keys and content
    assert "retrieved_documents" in result, "Result must contain 'retrieved_documents' key"
    assert "rag_context" in result, "Result must contain 'rag_context' key"
    
    # Verify retrieved_documents content
    docs = result["retrieved_documents"]
    assert isinstance(docs, list), "retrieved_documents must be a list"
    assert len(docs) == 2, "Should have retrieved 2 documents"
    
    print(f"✓ State validation passed: draft_answer generated ({len(result['draft_answer'])} chars)")
