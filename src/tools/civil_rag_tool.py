from functools import lru_cache
from langchain_core.tools.retriever import create_retriever_tool
from src.utils.rag_helper_chroma import get_civil_retriever

@lru_cache(maxsize=1)
def get_retriever():
    """
    민원 관련 RAG 리트리버 생성
    """
    retriever = get_civil_retriever()
    return retriever


@lru_cache(maxsize=1)
def get_civil_rag_tool():
    """
    민원 관련 RAG 도구 생성
    """
    retriever = get_retriever()
    return create_retriever_tool(
        retriever = retriever,
        name="retrieve_civil_docs",
        description="민원 질문과 유사한 과거 민원/기관 정보를 검색하는 도구."
    )
    
def get_tools():
    return [get_civil_rag_tool()]