# -*- coding: utf-8 -*-
"""
Test for draft_reply_node with Multi-View algorithm
"""
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

    # Mock the Kanana retriever and generator
    mock_retrieved_docs = [
        {"doc": "Test Document 1 - 법률 관련 내용", "meta": {"source": "test"}},
        {"doc": "Test Document 2 - 민원 사례", "meta": {"source": "test"}}
    ]
    
    mock_views = {
        "law": "법령 요약 내용",
        "case": "사례 요약 내용",
        "mixed": "종합 요약 내용",
    }
    
    mock_candidates = [
        {"strategy": "law_focus", "view": "law", "answer": "검토내용\n1. 테스트 답변", "critic_score": 0.8, "critic_reason": "좋음"},
    ]
    
    mock_final_answer = "검토내용\n1. 최종 테스트 답변"
    
    with patch("src.nodes.draft_reply_node.retrieve_with_kanana") as mock_retriever, \
         patch("src.nodes.draft_reply_node.build_views") as mock_build_views, \
         patch("src.nodes.draft_reply_node.generate_best_answer") as mock_gen_answer:
        
        mock_retriever.return_value = mock_retrieved_docs
        mock_build_views.return_value = mock_views
        mock_gen_answer.return_value = (mock_final_answer, mock_candidates)
        
        # When
        result = draft_reply_node(state)

    # Then - Validate state structure and content
    # 1. Check that 'draft_answer' key exists in result
    assert "draft_answer" in result, "Result must contain 'draft_answer' key"
    
    # 2. Check that draft_answer is a non-empty string
    assert isinstance(result["draft_answer"], str), "Draft answer must be a string"
    assert len(result["draft_answer"]) > 0, "Draft answer must not be empty"
    
    # 3. Check for new keys
    assert "retrieved_documents" in result, "Result must contain 'retrieved_documents' key"
    assert "views" in result, "Result must contain 'views' key"
    assert "candidates" in result, "Result must contain 'candidates' key"
    assert "rag_context" in result, "Result must contain 'rag_context' key"
    
    # 4. Verify retrieved_documents content
    docs = result["retrieved_documents"]
    assert isinstance(docs, list), "retrieved_documents must be a list"
    assert len(docs) == 2, "Should have retrieved 2 documents"
    
    # 5. Verify views content
    views = result["views"]
    assert isinstance(views, dict), "views must be a dict"
    assert "law" in views, "views must contain 'law' key"
    assert "case" in views, "views must contain 'case' key"
    assert "mixed" in views, "views must contain 'mixed' key"
    
    # 6. Verify candidates content
    candidates = result["candidates"]
    assert isinstance(candidates, list), "candidates must be a list"
    
    print(f"✓ State validation passed: draft_answer generated ({len(result['draft_answer'])} chars)")
    print(f"✓ Views: law={len(views['law'])} chars, case={len(views['case'])} chars, mixed={len(views['mixed'])} chars")
    print(f"✓ Candidates: {len(candidates)} generated")


def test_draft_reply_node_no_documents():
    """Test fallback when no documents are retrieved."""
    # Given
    state: CivilComplaintState = {
        "user_question": "Test Question",
        "refined_question": "Refined: Test Question",
        "retry_count": 0,
    }
    
    with patch("src.nodes.draft_reply_node.retrieve_with_kanana") as mock_retriever:
        mock_retriever.return_value = []
        
        # When
        result = draft_reply_node(state)
    
    # Then - Fallback response
    assert "draft_answer" in result
    assert "문서를 찾을 수 없어" in result["draft_answer"]
    assert result["retrieved_documents"] == []
    assert result["views"] == {}
    assert result["candidates"] == []
    
    print("✓ Fallback test passed: proper response when no documents")
