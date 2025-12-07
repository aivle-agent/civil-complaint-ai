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

    # 0.5 probability
    is_verified = random.random() > 0.5

    if is_verified:
        feedback = "Verification Passed."
        print("Verification: PASSED")
    else:
        feedback = "Verification Failed. Please retry."
        print("Verification: FAILED (Retrying...)")

    return {
        "verification_feedback": feedback,
        "is_verified": is_verified,
        "retry_count": state.get("retry_count", 0) + 1,
    }
    
