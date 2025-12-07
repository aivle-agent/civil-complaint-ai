from unittest.mock import patch, MagicMock
from src.models.state import CivilComplaintState
from src.nodes.generate_strategy_node import generate_strategy_node

from unittest.mock import patch
from langchain_core.documents import Document


def test_generate_strategy_node():
    # ① 더미 retriever 준비
    class DummyRetriever:
        def invoke(self, query):
            return [
                Document(page_content="dummy document", metadata={"source_file": "test.csv"})
            ]

    # ② generate_strategy_node 내부 retriever를 더미로 교체
    with patch(
        "src.nodes.generate_strategy_node.get_retriever",
        return_value=DummyRetriever()
    ):
        # Given
        state: CivilComplaintState = {
            "user_question": "Test Question",
            "refined_question": "Refined: Test Question",
            "retry_count": 0,
        }

        # When
        result = generate_strategy_node(state)

        # Then
        # 1. Check that 'strategy' key exists in result
        assert "strategy" in result, "Result must contain 'strategy' key"
        
        # 2. Check that strategy is a non-empty string
        assert isinstance(result["strategy"], str), "Strategy must be a string"
        assert len(result["strategy"]) > 0, "Strategy must not be empty"
        
        # 3. Check that 'rag_context' key exists in result
        assert "rag_context" in result, "Result must contain 'rag_context' key"
        assert isinstance(result["rag_context"], str), "rag_context must be a string"
        
        print(f"✓ Strategy generated: {result['strategy'][:50]}...")
        print(f"✓ RAG context generated: {len(result['rag_context'])} chars")
