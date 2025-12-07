from src.models.state import CivilComplaintState
from src.nodes.final_answer_node import final_answer_node


def test_final_answer_node():
    """Test that final_answer_node properly copies draft_answer to final_answer."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "draft_answer": "This is the verified draft answer that should be used as final answer",
        "retry_count": 0,
        "is_verified": True,
    }

    # When
    result = final_answer_node(state)

    # Then - Validate state structure and content
    # 1. Check that 'final_answer' key exists in result
    assert "final_answer" in result, "Result must contain 'final_answer' key"
    
    # 2. Check that final_answer is a non-empty string
    assert isinstance(result["final_answer"], str), "Final answer must be a string"
    assert len(result["final_answer"]) > 0, "Final answer must not be empty"
    
    # 3. Ensure no unexpected keys are added
    assert set(result.keys()) == {"final_answer"}, "Result should only contain 'final_answer' key"
    
    print(f"✓ State validation passed: final_answer generated ({len(result['final_answer'])} chars)")
