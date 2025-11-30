import pytest
from unittest.mock import patch
from src.models.state import CivilComplaintState
from src.nodes.verify_reply_node import verify_reply_node

def test_verify_reply_node_pass():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "draft_answer": "Test Draft",
        "retry_count": 0
    }
    
    # When
    # Mock random to return > 0.5 (e.g., 0.6) for Pass
    with patch('random.random', return_value=0.6):
        result = verify_reply_node(state)
    
    # Then
    assert result["is_verified"] is True
    assert result["verification_feedback"] == "Verification Passed."
    assert result["retry_count"] == 1

def test_verify_reply_node_fail():
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "draft_answer": "Test Draft",
        "retry_count": 0
    }
    
    # When
    # Mock random to return <= 0.5 (e.g., 0.4) for Fail
    with patch('random.random', return_value=0.4):
        result = verify_reply_node(state)
    
    # Then
    assert result["is_verified"] is False
    assert result["verification_feedback"] == "Verification Failed. Please retry."
    assert result["retry_count"] == 1
