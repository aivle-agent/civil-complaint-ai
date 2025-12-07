from langchain_openai import ChatOpenAI
from src.models.state import CivilComplaintState
from src.config import get_openai_api_key


def verify_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Verifies the drafted reply using OpenAI GPT-4o mini.
    Provides intelligent feedback on the quality of the answer.
    
    Args:
        state: Current state containing draft_answer, refined_question, and strategy
        
    Returns:
        Updated state with verification_feedback and is_verified fields
    """
    print("---VERIFY REPLY NODE---")
    
    draft_answer = state.get("draft_answer", "")
    refined_question = state.get("refined_question", state.get("user_question", ""))
    strategy = state.get("strategy", "")
    
    try:
        # Initialize OpenAI GPT-4o mini
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,  # Lower temperature for more consistent verification
            api_key=get_openai_api_key()
        )
        
        # Create prompt for verification
        prompt = f"""당신은 민원 답변의 품질을 검증하는 전문가입니다.

다음 민원 질문과 전략, 그리고 작성된 답변을 검토하고 품질을 평가해주세요.

[민원 질문]
{refined_question}

[답변 전략]
{strategy}

[작성된 답변]
{draft_answer}

다음 기준으로 답변을 평가해주세요:
1. 질문에 대한 답변이 적절한가?
2. 답변이 명확하고 이해하기 쉬운가?
3. 친절하고 공손한 어조인가?
4. 필요한 정보가 모두 포함되어 있는가?

평가 결과를 다음 형식으로 작성해주세요:
판정: [통과 또는 재작성필요]
피드백: [구체적인 피드백 내용]
"""
        
        # Invoke the LLM
        response = llm.invoke(prompt)
        feedback = response.content
        
        # Parse the result to determine if verified
        is_verified = "통과" in feedback and "재작성" not in feedback
        
        print(f"Verification: {'PASSED' if is_verified else 'FAILED'}")
        
        return {
            "verification_feedback": feedback,
            "is_verified": is_verified,
            "retry_count": state.get("retry_count", 0) + 1,
        }
        
    except Exception as e:
        print(f"Verification error: {e}")
        # On error, mark as not verified to trigger retry
        return {
            "verification_feedback": f"검증 중 오류 발생: {str(e)}",
            "is_verified": False,
            "retry_count": state.get("retry_count", 0) + 1,
        }
