from langchain_openai import ChatOpenAI
from src.models.state import CivilComplaintState
from src.tools.civil_rag_tool import get_retriever


def generate_strategy_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    질문을 참고해서 RAG을 통해 관련 문서를 가져오고,
    답변 전략을 생성하는 노드.
    """
    print("---GENERATE STRATEGY NODE---")
    refined_question = state.get("refined_question", "")

    retriever = get_retriever()
    docs = retriever.invoke(refined_question)
    rag_context = "\n\n".join(
        f"[{i+1}] sorce: {doc.metadata.get('source_file', 'unknown')}\n{doc.page_content}"
        for i , doc in enumerate(docs)
        )

    # Mock strategy generation
    # rag_context를 활용하여 실제로 전략을 생성하는 로직이 들어가야 함
    strategy = f"Strategy for: {refined_question} using {len(docs)} documents."
    return {"strategy": strategy, "rag_context": rag_context}



# if __name__ == "__main__":
#     # 간단한 테스트 실행
#     test_state: CivilComplaintState = {
#         "user_question": "불법주정차 단속에 대해 알려줘",
#         "refined_question": "불법주정차 단속",
#         "retry_count": 0,
#     }
#     result = generate_strategy_node(test_state)
#     print("Generated Strategy:", result["strategy"])
#     print("RAG Context:", result["rag_context"])
