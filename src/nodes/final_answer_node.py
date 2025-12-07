from src.models.state import CivilComplaintState


def final_answer_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Produces the final answer by using the verified draft_answer.
    
    Since the draft_answer has already been verified by verify_reply_node,
    we simply use it as the final_answer without additional processing.
    
    Args:
        state: Current state containing draft_answer
        
    Returns:
        Updated state with final_answer field
    """
    print("---FINAL ANSWER NODE---")
    
    draft_answer = state.get("draft_answer", "")
    
    # Use the verified draft_answer as the final_answer
    final_answer = draft_answer
    
    print(f"Final Answer: {final_answer[:100]}...")
    
    return {"final_answer": final_answer}
