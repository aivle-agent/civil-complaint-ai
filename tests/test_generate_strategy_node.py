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
        assert "strategy" in result
        assert result["strategy"] == "Strategy for: Refined: Test Question using 1 documents."
        assert "dummy document" in result["rag_context"]
