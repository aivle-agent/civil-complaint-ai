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

민원 질문: {refined_question}

답변 전략: {strategy}

작성된 답변 초안:
{draft_answer}

위 답변 초안을 다음 기준으로 평가해주세요:
1. 질문에 대한 직접적이고 명확한 답변 제공 여부
2. 정중하고 친절한 어조 사용 여부
3. 구체적인 절차나 방법 안내 제공 여부
4. 오류나 부적절한 내용 유무

평가 결과를 다음 형식으로 제공해주세요:
판정: [통과/재작성필요]
피드백: [구체적인 피드백 내용]

평가:"""

        # Generate verification using LLM
        response = llm.invoke(prompt)
        verification_result = response.content.strip()
        
        # Parse the result to determine if verification passed
        is_verified = "통과" in verification_result.split("\n")[0]
        
        if is_verified:
            feedback = f"검증 통과\n{verification_result}"
            print("Verification: PASSED")
        else:
            feedback = f"검증 실패 - 재작성 필요\n{verification_result}"
            print("Verification: FAILED (Retrying...)")
        
        print(f"Verification Feedback: {feedback[:100]}...")
        
        return {
            "verification_feedback": feedback,
            "is_verified": is_verified,
            "retry_count": state.get("retry_count", 0) + 1,
        }
        
    except Exception as e:
        # Fallback to simple verification if LLM fails
        print(f"Error verifying with LLM: {e}")
        print("Falling back to basic verification...")
        
        # Simple length-based check as fallback
        is_verified = len(draft_answer) > 50
        
        if is_verified:
            feedback = "검증 통과 (기본 검증)"
            print("Verification: PASSED (Fallback)")
        else:
            feedback = "검증 실패 - 답변이 너무 짧습니다. 재작성이 필요합니다."
            print("Verification: FAILED (Fallback)")
        
        return {
            "verification_feedback": feedback,
            "is_verified": is_verified,
            "retry_count": state.get("retry_count", 0) + 1,
        }
