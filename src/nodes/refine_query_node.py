from src.models.state import CivilComplaintState


def refine_query_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Refines the user's question for better clarity.
    """
    print("---REFINE QUERY NODE---")
    user_question = state["user_question"]
    # Mock refinement
    refined = f"Refined: {user_question}"
    return {"refined_question": refined}
