from src.models.state import CivilComplaintState
from src.tools.civil_rag_tool import get_civil_retriever

retriever = get_civil_retriever()

def retrieve_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    refined question을 바탕으로 RAG를 통해 관련 문서를 검색하는 노드.

    """
    print("---RETRIEVE NODE---")
    refined_question = state.get("refined_question", "")

    docs = retriever.invoke(refined_question)
    rag_context = "\n\n".join(
        f"[{i+1}] sorce: {doc.metadata.get('source_file', 'unknown')}\n{doc.page_content}"
        for i , doc in enumerate(docs)
        )
    return {"rag_context": rag_context}