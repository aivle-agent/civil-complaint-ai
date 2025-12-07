from langchain_openai import ChatOpenAI
from src.models.state import CivilComplaintState
from src.config import get_openai_api_key


def draft_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Drafts a reply to the civil complaint using OpenAI GPT-4o mini based on the strategy.
    
    Args:
        state: Current state containing strategy and refined_question
        
    Returns:
        Updated state with draft_answer field
    """
    print("---DRAFT REPLY NODE---")
    
    strategy = state.get("strategy")
    user_question = state.get("user_question")
    refined_question = state.get("refined_question")
    verification_feedback = state.get("verification_feedback")
    
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

답변 전략: {strategy}
"""
        
        # Add verification feedback if this is a retry
        if verification_feedback and "Failed" in verification_feedback:
            prompt += f"\n이전 답변 검토 피드백: {verification_feedback}\n위 피드백을 반영하여 답변을 개선해주세요.\n"
        
        prompt += """
위 전략을 바탕으로 민원인에게 제공할 상세하고 친절한 답변을 작성해주세요.

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
        
        return {"draft_answer": draft_answer}
        
    except Exception as e:
        # Fallback to mock draft if LLM fails
        print(f"Error generating draft answer with LLM: {e}")
        print("Falling back to mock draft...")
        draft_answer = f"답변 초안: {user_question}에 대한 상세 답변 (전략 기반: {strategy[:50]}...)"
        return {"draft_answer": draft_answer}
