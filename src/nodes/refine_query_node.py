from src.models.state import CivilComplaintState

def refine_query_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Basic refine query node.
    Currently just passes the user question through as the refined question.
    """
    print("---REFINE QUERY NODE---")
    user_question = state.get("user_question", "")
    
    # Minimal logic: just use the original question
    return {"refined_question": user_question}
