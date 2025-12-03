from langchain_openai import ChatOpenAI
from src.models.state import CivilComplaintState
from src.config import get_openai_api_key


def generate_strategy_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Generates a strategy to answer the civil complaint using OpenAI GPT-4o mini.
    
    Args:
        state: Current state containing refined_question
        
    Returns:
        Updated state with strategy field
    """
    print("---GENERATE STRATEGY NODE---")
    
    refined_question = state.get("refined_question", state.get("user_question", ""))
    
    try:
        # Initialize OpenAI GPT-4o mini
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=get_openai_api_key()
        )
        
        # Create prompt for strategy generation
        prompt = f"""당신은 민원 답변 전략을 수립하는 전문가입니다.

다음 민원 질문에 대한 답변 전략을 수립해주세요:

질문: {refined_question}

다음 항목들을 포함한 상세한 답변 전략을 작성해주세요:
1. 민원의 핵심 쟁점 파악
2. 필요한 정보 및 근거 자료
3. 답변 접근 방법
4. 주의사항

전략:"""

        # Generate strategy using LLM
        response = llm.invoke(prompt)
        strategy = response.content.strip()
        
        print(f"Generated Strategy: {strategy[:100]}...")
        
        return {"strategy": strategy}
        
    except Exception as e:
        # Fallback to mock strategy if LLM fails
        print(f"Error generating strategy with LLM: {e}")
        print("Falling back to mock strategy...")
        strategy = f"전략: {refined_question}에 대한 체계적 답변 전략 수립 필요"
        return {"strategy": strategy}
