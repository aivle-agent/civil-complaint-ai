from langchain_openai import ChatOpenAI
from src.models.state import CivilComplaintState
from src.config import get_openai_api_key
from src.utils.kanana_retriever import retrieve_with_kanana


def draft_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Drafts a reply to the civil complaint using OpenAI GPT-4o mini based on the strategy.
    
    Args:
        state: Current state containing strategy and refined_question
        
    Returns:
        Updated state with draft_answer field
    """
    print("---DRAFT REPLY NODE---")
    
    user_question = state.get("user_question")
    refined_question = state.get("refined_question")
    
    # RAG Retrieval
    print(f"Retrieving documents for: {refined_question[:50]}...")
    try:
        retrieved_docs = retrieve_with_kanana(refined_question, top_k=3)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        for i, doc in enumerate(retrieved_docs):
            print(f"[{i+1}] {doc['doc'][:100]}... (Meta: {doc.get('meta')})")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        retrieved_docs = []
    
    # Format retrieved context
    context_text = ""
    if retrieved_docs:
        context_text = "\n\n[참고 자료]\n"
        for i, doc in enumerate(retrieved_docs):
            context_text += f"{i+1}. {doc['doc']}\n"
            if doc.get('meta'):
                context_text += f"   (출처: {doc['meta']})\n"
    
    try:
        # Initialize OpenAI GPT-4o mini
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=get_openai_api_key()
        )
        
        # Create prompt for draft answer generation
        prompt = f"""당신은 민원에 답변하는 전문 상담원입니다.

민원 질문: {refined_question}
{context_text}
"""
        
        prompt += """
위 질문과 참고 자료를 바탕으로 민원인에게 제공할 상세하고 친절한 답변을 작성해주세요.
참고 자료가 있다면 적극적으로 활용하여 전문적인 답변을 해주세요.

답변 작성 시 다음 사항을 준수하세요:
1. 정중하고 공손한 어조 사용
2. 명확하고 이해하기 쉬운 설명
3. 구체적인 절차나 방법 안내
4. 필요시 관련 부서나 연락처 안내

답변:"""

        # Generate draft answer using LLM
        response = llm.invoke(prompt)
        draft_answer = response.content.strip()
        
        print(f"Generated Draft Answer: {draft_answer[:100]}...")
        
        return {
            "draft_answer": draft_answer,
            "retrieved_documents": retrieved_docs,
            "rag_context": context_text
        }
        
    except Exception as e:
        # Fallback to mock draft if LLM fails
        print(f"Error generating draft answer with LLM: {e}")
        print("Falling back to mock draft...")
        draft_answer = f"답변 초안: {user_question}에 대한 상세 답변"
        return {
            "draft_answer": draft_answer,
            "retrieved_documents": retrieved_docs,
            "rag_context": context_text
        }
