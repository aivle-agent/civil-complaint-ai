from src.models.state import CivilComplaintState
import transformers
import torch
from langchain_community.llms import HuggingFacePipeline


def generate_strategy_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    RAG 결과와 질문을 참고해서 답변 전략을 생성하는 노드.
    """
    print("---GENERATE STRATEGY NODE---")
    refined_question = state.get("refined_question", "")
    rag_context = state.get("rag_context", "")

    # Mock strategy generation
    strategy = f"Strategy for: {refined_question}"
    return {"strategy": strategy}
