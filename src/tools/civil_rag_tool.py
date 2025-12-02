from langchain_core.tools.retriever import create_retriever_tool
from src.utils.rag_helper_chroma import get_civil_retriever

retriever = get_civil_retriever()

civil_rag_tool = create_retriever_tool(
    retriever = retriever,
    name="retrieve_civil_docs",
    description="민원 질문과 유사한 과거 민원/기관 정보를 검색하는 도구."
)

tools = [civil_rag_tool]

# 테스트용
# test = civil_rag_tool.invoke({"query": "불법주정차가 너무 심해. 어디로 신고해야해?"})
# print(test)

# test = retriever.invoke("불법주정차가 너무 심해. 어디로 신고해야해?")
# for doc in test:
#     print(doc.metadata.keys())
    # print(doc.page_content)